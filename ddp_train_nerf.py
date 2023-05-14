import torch
# import torch.nn as nn
import torch.optim
import torch.distributed
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing
import os
import sys
import pdb
from collections import OrderedDict
from ddp_model import NerfNetWithAutoExpo
import time
from data_loader_split import load_data_split
import numpy as np
from tensorboardX import SummaryWriter
from utils import img2mse, mse2psnr, img_HWC2CHW, colorize, TINY_NUMBER, save_image
import logging
import json
import mcubes
from demo_projSH_rotSH import Rotation
import imageio


logger = logging.getLogger(__package__)


class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child

    """

    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin


def setup_logger():
    # create logger
    logger = logging.getLogger(__package__)
    # logger.setLevel(logging.DEBUG)
    logger.setLevel(logging.INFO)

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # create formatter
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')

    # add formatter to ch
    ch.setFormatter(formatter)

    # add ch to logger
    logger.addHandler(ch)


def intersect_sphere(ray_o, ray_d):
    '''
    ray_o, ray_d: [..., 3]
    compute the depth of the intersection point between this ray and unit sphere
    '''
    # note: d1 becomes negative if this mid point is behind camera
    d1 = -torch.sum(ray_d * ray_o, dim=-1) / torch.sum(ray_d * ray_d, dim=-1)
    p = ray_o + d1.unsqueeze(-1) * ray_d
    # consider the case where the ray does not intersect the sphere
    ray_d_cos = 1. / torch.norm(ray_d, dim=-1)
    p_norm_sq = torch.sum(p * p, dim=-1)
    if (p_norm_sq >= 1.).any():
        raise Exception(
            'Not all your cameras are bounded by the unit sphere; please make sure the cameras are normalized properly!')
    d2 = torch.sqrt(1. - p_norm_sq) * ray_d_cos

    return d1 + d2


def perturb_samples(z_vals):
    # get intervals between samples
    mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
    upper = torch.cat([mids, z_vals[..., -1:]], dim=-1)
    lower = torch.cat([z_vals[..., 0:1], mids], dim=-1)
    # uniform samples in those intervals
    t_rand = torch.rand_like(z_vals)
    z_vals = lower + (upper - lower) * t_rand  # [N_rays, N_samples]

    return z_vals


def sample_pdf(bins, weights, N_samples, det=False):
    '''
    :param bins: tensor of shape [..., M+1], M is the number of bins
    :param weights: tensor of shape [..., M]
    :param N_samples: number of samples along each ray
    :param det: if True, will perform deterministic sampling
    :return: [..., N_samples]
    '''
    # Get pdf
    weights = weights + TINY_NUMBER  # prevent nans
    pdf = weights / torch.sum(weights, dim=-1, keepdim=True)  # [..., M]
    cdf = torch.cumsum(pdf, dim=-1)  # [..., M]
    cdf = torch.cat([torch.zeros_like(cdf[..., 0:1]), cdf], dim=-1)  # [..., M+1]

    # Take uniform samples
    dots_sh = list(weights.shape[:-1])
    M = weights.shape[-1]

    min_cdf = 0.00
    max_cdf = 1.00  # prevent outlier samples

    if det:
        u = torch.linspace(min_cdf, max_cdf, N_samples, device=bins.device)
        u = u.view([1] * len(dots_sh) + [N_samples]).expand(dots_sh + [N_samples, ])  # [..., N_samples]
    else:
        sh = dots_sh + [N_samples]
        u = torch.rand(*sh, device=bins.device) * (max_cdf - min_cdf) + min_cdf  # [..., N_samples]

    # Invert CDF
    # [..., N_samples, 1] >= [..., 1, M] ----> [..., N_samples, M] ----> [..., N_samples,]
    above_inds = torch.sum(u.unsqueeze(-1) >= cdf[..., :M].unsqueeze(-2), dim=-1).long()

    # random sample inside each bin
    below_inds = torch.clamp(above_inds - 1, min=0)
    inds_g = torch.stack((below_inds, above_inds), dim=-1)  # [..., N_samples, 2]

    cdf = cdf.unsqueeze(-2).expand(dots_sh + [N_samples, M + 1])  # [..., N_samples, M+1]
    cdf_g = torch.gather(input=cdf, dim=-1, index=inds_g)  # [..., N_samples, 2]

    bins = bins.unsqueeze(-2).expand(dots_sh + [N_samples, M + 1])  # [..., N_samples, M+1]
    bins_g = torch.gather(input=bins, dim=-1, index=inds_g)  # [..., N_samples, 2]

    # fix numeric issue
    denom = cdf_g[..., 1] - cdf_g[..., 0]  # [..., N_samples]
    denom = torch.where(denom < TINY_NUMBER, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom

    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0] + TINY_NUMBER)

    return samples


def render_single_image(rank, world_size, models, ray_sampler, chunk_size, iteration, rot_angle=0, img_name=None):
    ##### parallel rendering of a single image
    ray_batch = ray_sampler.get_all()

    fixed = 0
    if (ray_batch['ray_d'].shape[0] // world_size) * world_size != ray_batch['ray_d'].shape[0]:
        fixed = world_size - (ray_batch['ray_d'].shape[0] % world_size)
        for p in ray_batch:
            if ray_batch[p] is not None:
                ray_batch[p] = torch.cat((ray_batch[p], ray_batch[p][-fixed:]), dim=0)
    #     raise Exception('Number of pixels in the image is not divisible by the number of GPUs!\n\t# pixels: {}\n\t# GPUs: {}'.format(ray_batch['ray_d'].shape[0],
    #                                                                                                                                  world_size))

    # split into ranks; make sure different processes don't overlap
    rank_split_sizes = [ray_batch['ray_d'].shape[0] // world_size, ] * world_size
    rank_split_sizes[-1] = ray_batch['ray_d'].shape[0] - sum(rank_split_sizes[:-1])
    for key in ray_batch:
        if torch.is_tensor(ray_batch[key]):
            ray_batch[key] = torch.split(ray_batch[key], rank_split_sizes)[rank].to(rank)

    # split into chunks and render inside each process
    ray_batch_split = OrderedDict()
    for key in ray_batch:
        if torch.is_tensor(ray_batch[key]):
            ray_batch_split[key] = torch.split(ray_batch[key], chunk_size)

    # forward and backward
    ret_merge_chunk = [OrderedDict() for _ in range(models['cascade_level'])]
    for s in range(len(ray_batch_split['ray_d'])):
        ray_o = ray_batch_split['ray_o'][s]
        ray_d = ray_batch_split['ray_d'][s]
        min_depth = ray_batch_split['min_depth'][s]

        dots_sh = list(ray_d.shape[:-1])
        for m in range(models['cascade_level']):
            net = models['net_{}'.format(m)]
            # sample depths
            N_samples = models['cascade_samples'][m]
            if m == 0:
                # foreground depth
                fg_far_depth = intersect_sphere(ray_o, ray_d)  # [...,]
                fg_near_depth = min_depth  # [..., ]
                step = (fg_far_depth - fg_near_depth) / (N_samples - 1)
                fg_depth = torch.stack([fg_near_depth + i * step for i in range(N_samples)], dim=-1)  # [..., N_samples]

                # background depth
                bg_depth = torch.linspace(0., 1., N_samples).view(
                    [1, ] * len(dots_sh) + [N_samples, ]).expand(dots_sh + [N_samples, ]).to(rank)

                # delete unused memory
                del fg_near_depth
                del step
                torch.cuda.empty_cache()
            else:
                # sample pdf and concat with earlier samples
                fg_weights = ret['fg_weights'].clone().detach()
                fg_depth_mid = .5 * (fg_depth[..., 1:] + fg_depth[..., :-1])  # [..., N_samples-1]
                fg_weights = fg_weights[..., 1:-1]  # [..., N_samples-2]
                fg_depth_samples = sample_pdf(bins=fg_depth_mid, weights=fg_weights,
                                              N_samples=N_samples, det=True)  # [..., N_samples]
                fg_depth, _ = torch.sort(torch.cat((fg_depth, fg_depth_samples), dim=-1))

                # sample pdf and concat with earlier samples
                bg_weights = ret['bg_weights'].clone().detach()
                bg_depth_mid = .5 * (bg_depth[..., 1:] + bg_depth[..., :-1])
                bg_weights = bg_weights[..., 1:-1]  # [..., N_samples-2]
                bg_depth_samples = sample_pdf(bins=bg_depth_mid, weights=bg_weights,
                                              N_samples=N_samples, det=True)  # [..., N_samples]
                bg_depth, _ = torch.sort(torch.cat((bg_depth, bg_depth_samples), dim=-1))

                # delete unused memory
                del fg_weights
                del fg_depth_mid
                del fg_depth_samples
                del bg_weights
                del bg_depth_mid
                del bg_depth_samples
                torch.cuda.empty_cache()

            with torch.no_grad():
                ret = net(ray_o, ray_d, fg_far_depth, fg_depth, bg_depth, iteration, img_name=ray_sampler.img_path if img_name is None else img_name,
                          rot_angle=rot_angle, save_memory4validation=True)

            for key in ret:
                if key not in ['fg_weights', 'bg_weights']:
                    if torch.is_tensor(ret[key]):
                        if key not in ret_merge_chunk[m]:
                            ret_merge_chunk[m][key] = [ret[key].cpu(), ]
                        else:
                            ret_merge_chunk[m][key].append(ret[key].cpu())

                        ret[key] = None

            # clean unused memory
            torch.cuda.empty_cache()

    # merge results from different chunks
    for m in range(len(ret_merge_chunk)):
        for key in ret_merge_chunk[m]:
            ret_merge_chunk[m][key] = torch.cat(ret_merge_chunk[m][key], dim=0)

    # merge results from different processes
    if rank == 0:
        ret_merge_rank = [OrderedDict() for _ in range(len(ret_merge_chunk))]
        for m in range(len(ret_merge_chunk)):
            for key in ret_merge_chunk[m]:
                # generate tensors to store results from other processes
                sh = list(ret_merge_chunk[m][key].shape[1:])
                ret_merge_rank[m][key] = [torch.zeros(*[size, ] + sh, dtype=torch.float32) for size in rank_split_sizes]
                torch.distributed.gather(ret_merge_chunk[m][key], ret_merge_rank[m][key])
                ret_merge_rank[m][key] = torch.cat(ret_merge_rank[m][key], dim=0)
                if fixed > 0:
                    ret_merge_rank[m][key] = ret_merge_rank[m][key][:-fixed]
                ret_merge_rank[m][key] = ret_merge_rank[m][key].reshape(
                    (ray_sampler.H, ray_sampler.W, -1)).squeeze()
                # print(m, key, ret_merge_rank[m][key].shape)
    else:  # send results to main process
        for m in range(len(ret_merge_chunk)):
            for key in ret_merge_chunk[m]:
                torch.distributed.gather(ret_merge_chunk[m][key])

    # only rank 0 program returns
    if rank == 0:
        return ret_merge_rank
    else:
        return None

def relight_rotation_single_image(rank, world_size, models, ray_sampler, chunk_size, iteration, rot_angle=0, img_name=None):
    ##### parallel rendering of a single image
    ray_batch = ray_sampler.get_all()

    fixed = 0
    if (ray_batch['ray_d'].shape[0] // world_size) * world_size != ray_batch['ray_d'].shape[0]:
        fixed = world_size - (ray_batch['ray_d'].shape[0] % world_size)
        for p in ray_batch:
            if ray_batch[p] is not None:
                ray_batch[p] = torch.cat((ray_batch[p], ray_batch[p][-fixed:]), dim=0)
    #     raise Exception('Number of pixels in the image is not divisible by the number of GPUs!\n\t# pixels: {}\n\t# GPUs: {}'.format(ray_batch['ray_d'].shape[0],
    #                                                                                                                                  world_size))

    # split into ranks; make sure different processes don't overlap
    rank_split_sizes = [ray_batch['ray_d'].shape[0] // world_size, ] * world_size
    rank_split_sizes[-1] = ray_batch['ray_d'].shape[0] - sum(rank_split_sizes[:-1])
    for key in ray_batch:
        if torch.is_tensor(ray_batch[key]):
            ray_batch[key] = torch.split(ray_batch[key], rank_split_sizes)[rank].to(rank)

    # split into chunks and render inside each process
    ray_batch_split = OrderedDict()
    for key in ray_batch:
        if torch.is_tensor(ray_batch[key]):
            ray_batch_split[key] = torch.split(ray_batch[key], chunk_size)

    # forward and backward
    ret_merge_chunk = [OrderedDict() for _ in range(models['cascade_level'])]
    for s in range(len(ray_batch_split['ray_d'])):
        ray_o = ray_batch_split['ray_o'][s]
        ray_d = ray_batch_split['ray_d'][s]
        min_depth = ray_batch_split['min_depth'][s]

        dots_sh = list(ray_d.shape[:-1])
        for m in range(models['cascade_level']):
            if m == 0: continue
            net = models['net_{}'.format(m)]
            # sample depths
            N_samples = models['cascade_samples'][m]
            if m == 0:
                # foreground depth
                fg_far_depth = intersect_sphere(ray_o, ray_d)  # [...,]
                fg_near_depth = min_depth  # [..., ]
                step = (fg_far_depth - fg_near_depth) / (N_samples - 1)
                fg_depth = torch.stack([fg_near_depth + i * step for i in range(N_samples)], dim=-1)  # [..., N_samples]

                # background depth
                bg_depth = torch.linspace(0., 1., N_samples).view(
                    [1, ] * len(dots_sh) + [N_samples, ]).expand(dots_sh + [N_samples, ]).to(rank)

                # delete unused memory
                del fg_near_depth
                del step
                torch.cuda.empty_cache()
            else:
                # sample pdf and concat with earlier samples
                fg_weights = ret['fg_weights'].clone().detach()
                fg_depth_mid = .5 * (fg_depth[..., 1:] + fg_depth[..., :-1])  # [..., N_samples-1]
                fg_weights = fg_weights[..., 1:-1]  # [..., N_samples-2]
                fg_depth_samples = sample_pdf(bins=fg_depth_mid, weights=fg_weights,
                                              N_samples=N_samples, det=True)  # [..., N_samples]
                fg_depth, _ = torch.sort(torch.cat((fg_depth, fg_depth_samples), dim=-1))

                # sample pdf and concat with earlier samples
                bg_weights = ret['bg_weights'].clone().detach()
                bg_depth_mid = .5 * (bg_depth[..., 1:] + bg_depth[..., :-1])
                bg_weights = bg_weights[..., 1:-1]  # [..., N_samples-2]
                bg_depth_samples = sample_pdf(bins=bg_depth_mid, weights=bg_weights,
                                              N_samples=N_samples, det=True)  # [..., N_samples]
                bg_depth, _ = torch.sort(torch.cat((bg_depth, bg_depth_samples), dim=-1))

                # delete unused memory
                del fg_weights
                del fg_depth_mid
                del fg_depth_samples
                del bg_weights
                del bg_depth_mid
                del bg_depth_samples
                torch.cuda.empty_cache()

            with torch.no_grad():
                ret = net(ray_o, ray_d, fg_far_depth, fg_depth, bg_depth, iteration,
                          img_name=ray_sampler.img_path if img_name is None else img_name,
                          rot_angle=rot_angle, save_memory4validation=True)

            for key in ret:
                if key not in ['fg_weights', 'bg_weights']:
                    if torch.is_tensor(ret[key]):
                        if key not in ret_merge_chunk[m]:
                            ret_merge_chunk[m][key] = [ret[key].cpu(), ]
                        else:
                            ret_merge_chunk[m][key].append(ret[key].cpu())

                        ret[key] = None

            # clean unused memory
            torch.cuda.empty_cache()

    # merge results from different chunks
    for m in range(len(ret_merge_chunk)):
        for key in ret_merge_chunk[m]:
            ret_merge_chunk[m][key] = torch.cat(ret_merge_chunk[m][key], dim=0)

    # merge results from different processes
    if rank == 0:
        ret_merge_rank = [OrderedDict() for _ in range(len(ret_merge_chunk))]
        for m in range(len(ret_merge_chunk)):
            for key in ret_merge_chunk[m]:
                # generate tensors to store results from other processes
                sh = list(ret_merge_chunk[m][key].shape[1:])
                ret_merge_rank[m][key] = [torch.zeros(*[size, ] + sh, dtype=torch.float32) for size in rank_split_sizes]
                torch.distributed.gather(ret_merge_chunk[m][key], ret_merge_rank[m][key])
                ret_merge_rank[m][key] = torch.cat(ret_merge_rank[m][key], dim=0)
                if fixed > 0:
                    ret_merge_rank[m][key] = ret_merge_rank[m][key][:-fixed]
                ret_merge_rank[m][key] = ret_merge_rank[m][key].reshape(
                    (ray_sampler.H, ray_sampler.W, -1)).squeeze()
                # print(m, key, ret_merge_rank[m][key].shape)
    else:  # send results to main process
        for m in range(len(ret_merge_chunk)):
            for key in ret_merge_chunk[m]:
                torch.distributed.gather(ret_merge_chunk[m][key])

    # only rank 0 program returns
    if rank == 0:
        return ret_merge_rank
    else:
        return None

def log_view_to_tb(output_dir, writer, global_step, log_data, gt_img, mask, prefix=''):
    rgb_im = img_HWC2CHW(torch.from_numpy(gt_img))
    # writer.add_image(prefix + 'rgb_gt', rgb_im, global_step)
    save_image(output_dir + prefix + 'rgb_gt.png', 255*gt_img)

    for m in range(len(log_data)):
        rgb_im = (log_data[m]['rgb'])
        rgb_im = torch.clamp(rgb_im, min=0., max=1.)  # just in case diffuse+specular>1
        # writer.add_image(prefix + 'level_{}/rgb'.format(m), rgb_im, global_step)
        save_image(output_dir + prefix + 'level_{}_rgb.png'.format(m), 255*rgb_im.numpy())

        purergb_im = (log_data[m]['pure_rgb'])
        purergb_im = torch.clamp(purergb_im, min=0., max=1.)  # just in case diffuse+specular>1
        # writer.add_image(prefix + 'level_{}/purergb'.format(m), purergb_im, global_step)
        save_image(output_dir + prefix + 'level_{}_purergb.png'.format(m), 255*purergb_im.numpy())

        shadow_im = (log_data[m]['shadow'])
        shadow_im = torch.clamp(shadow_im, min=0., max=1.)  # just in case diffuse+specular>1
        # writer.add_image(prefix + 'level_{}/shadow'.format(m), shadow_im, global_step)
        save_image(output_dir + prefix + 'level_{}_shadow.png'.format(m), 255*shadow_im.numpy())

        albedo_im = (log_data[m]['fg_albedo'])
        albedo_im = torch.clamp(albedo_im, min=0., max=1.)  # just in case diffuse+specular>1
        # writer.add_image(prefix + 'level_{}/fg_albedo'.format(m), albedo_im, global_step)
        save_image(output_dir + prefix + 'level_{}_fg_albedo.png'.format(m), 255*albedo_im.numpy())

        rgb_im = (log_data[m]['fg_rgb'])
        rgb_im = torch.clamp(rgb_im, min=0., max=1.)  # just in case diffuse+specular>1
        # writer.add_image(prefix + 'level_{}/fg_rgb'.format(m), rgb_im, global_step)
        save_image(output_dir + prefix + 'level_{}_fg_rgb.png'.format(m), 255*rgb_im.numpy())

        depth = log_data[m]['fg_depth']
        depth_im = (colorize(depth, cmap_name='jet', append_cbar=True, mask=mask))
        # writer.add_image(prefix + 'level_{}/fg_depth'.format(m), depth_im, global_step)
        save_image(output_dir + prefix + 'level_{}_fg_depth.png'.format(m), 255*depth_im.numpy())

        normal_im = (log_data[m]['fg_normal'])
        normal_im = (normal_im + 1) / 2
        normal_im = torch.clamp(normal_im, min=0., max=1.)  # just in case diffuse+specular>1
        # writer.add_image(prefix + 'level_{}/fg_normal'.format(m), normal_im, global_step)
        save_image(output_dir + prefix + 'level_{}_fg_normal.png'.format(m), 255*normal_im.numpy())

        # for debugging
        # normal_im = (log_data[m]['mean_normal_grad'])
        # normal_im = (normal_im + 1) / 2
        # normal_im = torch.clamp(normal_im, min=0., max=1.)  # just in case diffuse+specular>1
        # # writer.add_image(prefix + 'level_{}/fg_normal'.format(m), normal_im, global_step)
        # save_image(output_dir + prefix + 'level_{}_fg_normal_smooth.png'.format(m), 255*normal_im.numpy())

        irradiance_im = (log_data[m]['irradiance'])
        irradiance_im = irradiance_im / max(1.0, irradiance_im.max().item())
        irradiance_im = torch.clamp(irradiance_im, min=0., max=1.)  # just in case diffuse+specular>1
        # writer.add_image(prefix + 'level_{}/fg_irradiance'.format(m), irradiance_im, global_step)
        save_image(output_dir + prefix + 'level_{}_fg_irradiance.png'.format(m), 255*irradiance_im.numpy())

        rgb_im = (log_data[m]['bg_rgb'])
        rgb_im = torch.clamp(rgb_im, min=0., max=1.)  # just in case diffuse+specular>1
        # writer.add_image(prefix + 'level_{}/bg_rgb'.format(m), rgb_im, global_step)
        save_image(output_dir + prefix + 'level_{}_bg_rgb.png'.format(m), 255*rgb_im.numpy())

        depth = log_data[m]['bg_depth']
        depth_im = (colorize(depth, cmap_name='jet', append_cbar=True, mask=mask))
        # writer.add_image(prefix + 'level_{}/bg_depth'.format(m), depth_im, global_step)
        save_image(output_dir + prefix + 'level_{}_bg_depth.png'.format(m), 255*depth_im.numpy())

        bg_lambda = log_data[m]['bg_lambda']
        bg_lambda_im = (colorize(bg_lambda, cmap_name='hot', append_cbar=True,  mask=mask))
        # writer.add_image(prefix + 'level_{}/bg_lambda'.format(m), bg_lambda_im, global_step)
        save_image(output_dir + prefix + 'evel_{}_bg_lambda.png'.format(m), 255*bg_lambda_im.numpy())

# eikonal for normal smoothness
def get_eikonal_loss(mean_normal_grad):
    if mean_normal_grad.shape[0] == 0:
        return torch.tensor(0.0).cuda().float()

    eikonal_loss = ((mean_normal_grad.norm(2, dim=1) - 1) ** 2).mean()
    return eikonal_loss


def setup(rank, world_size, master_port):
    # initialize the process group
    slurmjob = os.environ.get('SLURM_JOB_ID', '')
    os.environ['MASTER_ADDR'] = 'localhost'
    if len(slurmjob) > 0:
        os.environ['MASTER_PORT'] = str(12000+int(slurmjob)%10000)
        logger.info('using master port ' + os.environ['MASTER_PORT'] + ' based on slurmjob ' + slurmjob)
        torch.distributed.init_process_group("gloo", rank=rank, world_size=world_size)
    else:
        try:
            # os.environ['MASTER_PORT'] = '12420'
            os.environ['MASTER_PORT'] = str(master_port + rank * 100)
            logger.info('using master port ' + os.environ['MASTER_PORT'] + ' based on first try')
            torch.distributed.init_process_group("gloo", rank=rank, world_size=world_size)
        except RuntimeError:
            try:
                # os.environ['MASTER_PORT'] = '12612'
                os.environ['MASTER_PORT'] = str(master_port + rank * 43)
                logger.info('using master port ' + os.environ['MASTER_PORT'] + ' based on second try')
                torch.distributed.init_process_group("gloo", rank=rank, world_size=world_size)
            except RuntimeError:
                # os.environ['MASTER_PORT'] = '15125'
                os.environ['MASTER_PORT'] = str(master_port + rank * 77)
                logger.info('using master port ' + os.environ['MASTER_PORT'] + ' based on third try')
                torch.distributed.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    torch.distributed.destroy_process_group()


def create_nerf(rank, args):
    ###### create network and wrap in ddp; each process should do this
    # fix random seed just to make sure the network is initialized with same weights at different processes
    torch.manual_seed(777)
    # very important!!! otherwise it might introduce extra memory in rank=0 gpu
    torch.cuda.set_device(rank)

    models = OrderedDict()
    models['cascade_level'] = args.cascade_level
    models['cascade_samples'] = [int(x.strip()) for x in args.cascade_samples.split(',')]
    for m in range(models['cascade_level']):
        img_names = None
        if args.optim_autoexpo or True:  # todo: upstream
            # load training image names for autoexposure
            f = os.path.join(args.basedir, args.expname, 'train_images.json')
            with open(f) as file:
                img_names = json.load(file)
        net = NerfNetWithAutoExpo(args, optim_autoexpo=args.optim_autoexpo, img_names=img_names).to(rank)
        net = DDP(net, device_ids=[rank], output_device=rank, find_unused_parameters=True)
        # net = DDP(net, device_ids=[rank], output_device=rank)
        optim = torch.optim.Adam(net.parameters(), lr=args.lrate)
        models['net_{}'.format(m)] = net
        models['optim_{}'.format(m)] = optim

    start = -1

    ###### load pretrained weights; each process should do this
    if (args.ckpt_path is not None) and (os.path.isfile(args.ckpt_path)):
        ckpts = [args.ckpt_path]
    else:
        ckpts = [os.path.join(args.basedir, args.expname, f)
                 for f in sorted(os.listdir(os.path.join(args.basedir, args.expname))) if f.endswith('.pth')]

    def path2iter(path):
        tmp = os.path.basename(path)[:-4]
        idx = tmp.rfind('_')
        return int(tmp[idx + 1:])

    ckpts = sorted(ckpts, key=path2iter)
    logger.info('Found ckpts: {}'.format(ckpts))

    if len(ckpts) > 0 and not args.no_reload:
        fpath = ckpts[-1]
        logger.info('Reloading from: {}'.format(fpath))
        start = path2iter(fpath)
        # configure map_location properly for different processes
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        to_load = torch.load(fpath, map_location=map_location)
        for m in range(models['cascade_level']):
            for name in ['net_{}'.format(m), 'optim_{}'.format(m)]:
                # if name.startswith('net_') and 'module.nerf_net.env' not in to_load[name]:  # todo: upstream
                #     logger.warning('augmenting module.nerf_net.env')
                #     to_load[name]['module.nerf_net.env'] = models[name].module.nerf_net.env
                if 'module.defaultenv' in to_load[name]:
                    device = to_load[name]['module.defaultenv'].device
                    to_load[name]['module.defaultenv'] = torch.tensor([
                        # extracted with eccv
			  # [-0.01647952,  0.02611163, -0.2271134 ],
			  # [ 0.82858413,  0.53440607,  0.3167584 ],
			  # [ 0.6309354 ,  0.7007196 ,  0.8924455 ],
			  # [-0.10366065, -0.16238835, -0.18382286],
			  # [-0.2793248 , -0.07769008,  0.15571593],
			  # [-0.5931262 , -0.02438579,  0.71688586],
			  # [ 0.0773728 , -0.24582024, -0.6405687 ],
			  # [-0.39268702, -0.19289029,  0.04664679],
			  # [ 0.5131015 ,  0.24060975, -0.12801889]], device=device)

                        # # self-learned
                        # [ 2.9861e+00,  3.4646e+00,  3.9559e+00],
                        # [ 1.0013e-01, -6.7589e-02, -3.1161e-01],
                        # [ 8.2520e-01,  5.2738e-01,  9.7385e-02],
                        # [-2.2311e-03, -4.3553e-03, -4.9501e-03],
                        # [ 6.4355e-03, -9.7476e-03,  2.3863e-02],
                        # [-1.1078e-01,  6.0607e-02,  1.9541e-01],
                        # [ 7.9123e-01,  7.6916e-01,  5.6288e-01],
                        # [ 6.5793e-02,  4.3270e-02, -1.7002e-01],
                        # [-7.2674e-02,  4.5177e-02,  2.2858e-01]], device=device)

                        # self-learned
                        [ 2.9861e+00,  3.4646e+00,  3.9559e+00],
                        [ 1.0013e-01, -6.7589e-02, -3.1161e-01],
                        [-8.2520e-01, -5.2738e-01, -9.7385e-02],
                        [ 2.2311e-03,  4.3553e-03,  4.9501e-03],
                        [-6.4355e-03,  9.7476e-03, -2.3863e-02],
                        [ 1.1078e-01, -6.0607e-02, -1.9541e-01],
                        [ 7.9123e-01,  7.6916e-01,  5.6288e-01],
                        [ 6.5793e-02,  4.3270e-02, -1.7002e-01],
                        [-7.2674e-02,  4.5177e-02,  2.2858e-01]], device=device)
                models[name].load_state_dict(to_load[name])

    return start, models


def ddp_train_nerf(rank, args, one_card=False):
    ###### set up multi-processing
    if one_card == False:
        setup(rank, args.world_size, args.master_port)
    else:
        os.environ['MASTER_PORT'] = '12413'
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
    ###### set up logger
    logger = logging.getLogger(__package__)
    setup_logger()

    ###### decide chunk size according to gpu memory
    logger.info('gpu_mem: {}'.format(torch.cuda.get_device_properties(rank).total_memory))
    if torch.cuda.get_device_properties(rank).total_memory / 1e9 > 25:
        logger.info('setting batch size according to 24G gpu')
        args.N_rand = 512
        args.chunk_size = 4096
    elif torch.cuda.get_device_properties(rank).total_memory / 1e9 > 9:
        logger.info('setting batch size according to 12G gpu')
        args.N_rand = 512
        # args.N_rand = 128
        args.chunk_size = int(4096//1.15)
        # args.chunk_size = 1024
    else:
        logger.info('setting batch size according to 4G gpu')
        args.N_rand = 128
        args.chunk_size = 1024

    ###### Create log dir and copy the config file
    if rank == 0:
        os.makedirs(os.path.join(args.basedir, args.expname), exist_ok=True)
        f = os.path.join(args.basedir, args.expname, 'args.txt')
        with open(f, 'w') as file:
            for arg in sorted(vars(args)):
                attr = getattr(args, arg)
                file.write('{} = {}\n'.format(arg, attr))
        if args.config is not None:
            f = os.path.join(args.basedir, args.expname, 'config.txt')
            with open(f, 'w') as file:
                file.write(open(args.config, 'r').read())
    torch.distributed.barrier()

    ray_samplers = load_data_split(args.datadir, args.scene, split='train',
                                   try_load_min_depth=args.load_min_depth, use_ray_jitter=args.use_ray_jitter,
                                   resolution_level=args.resolution_level)
    val_ray_samplers = load_data_split(args.datadir, args.scene, split='validation',
                                       try_load_min_depth=args.load_min_depth, skip=args.testskip, use_ray_jitter=args.use_ray_jitter,
                                       resolution_level=args.resolution_level)

    # write training image names for autoexposure
    if args.optim_autoexpo or True:  # todo: upstream
        f = os.path.join(args.basedir, args.expname, 'train_images.json')
        with open(f, 'w') as file:
            img_names = [ray_samplers[i].img_path for i in range(len(ray_samplers))]
            json.dump(img_names, file, indent=2)

    ###### create network and wrap in ddp; each process should do this
    start, models = create_nerf(rank, args)

    ##### important!!!
    # make sure different processes sample different rays
    np.random.seed((rank + 1) * 777)
    # make sure different processes have different perturbations in depth samples
    torch.manual_seed((rank + 1) * 777)

    ##### only main process should do the logging
    if rank == 0:
        writer = SummaryWriter(os.path.join(args.basedir, args.expname))
        # overall loss writer
        os.makedirs(os.path.join(args.basedir, args.expname, 'loss_curve'), exist_ok=True)
        writer_loss = SummaryWriter(os.path.join(args.basedir, args.expname, 'loss_curve'))

    # start training
    what_val_to_log = 0  # helper variable for parallel rendering of a image
    what_train_to_log = 0
    for global_step in range(start + 1, start + 1 + args.N_iters):
        time0 = time.time()
        scalars_to_log = OrderedDict()
        ### Start of core optimization loop
        scalars_to_log['resolution'] = ray_samplers[0].resolution_level
        # randomly sample rays and move to device
        i = np.random.randint(low=0, high=len(ray_samplers))
        ray_batch = ray_samplers[i].random_sample(args.N_rand, center_crop=False)
        for key in ray_batch:
            if torch.is_tensor(ray_batch[key]):
                ray_batch[key] = ray_batch[key].to(rank)

        # forward and backward
        dots_sh = list(ray_batch['ray_d'].shape[:-1])  # number of rays
        all_rets = []  # results on different cascade levels
        for m in range(models['cascade_level']):
            optim = models['optim_{}'.format(m)]
            net = models['net_{}'.format(m)]

            ## debug: how to cube marching
            # li = [10000]
            # for threshold in li:
            # vertices, triangles = net.module.nerf_net.extract_mesh(torch.tensor([-1.1, -1.1, -1.1]).cuda(), torch.tensor([1.1, 1.1, 1.1]).cuda(), 128, 5000, global_step)
            # mcubes.export_obj(vertices, triangles, '/home/youmingdeng/lwp.obj')
            # ForkedPdb().set_trace()
            ## debug: how to cube marching

            # sample depths
            N_samples = models['cascade_samples'][m]
            if m == 0:
                # foreground depth
                fg_far_depth = intersect_sphere(ray_batch['ray_o'], ray_batch['ray_d'])  # [...,]
                fg_near_depth = ray_batch['min_depth']  # [..., ]
                step = (fg_far_depth - fg_near_depth) / (N_samples - 1)
                fg_depth = torch.stack([fg_near_depth + i * step for i in range(N_samples)], dim=-1)  # [..., N_samples]
                fg_depth = perturb_samples(fg_depth)  # random perturbation during training

                ## debugging: to find far and close boundary
                '''
                far = ray_batch['ray_o'] + fg_far_depth.unsqueeze(-1) * ray_batch['ray_d']
                close = ray_batch['ray_o'] + fg_near_depth.unsqueeze(-1) * ray_batch['ray_d']
                ForkedPdb().set_trace()
                '''

                # background depth
                bg_depth = torch.linspace(0., 1., N_samples).view(
                    [1, ] * len(dots_sh) + [N_samples, ]).expand(dots_sh + [N_samples, ]).to(rank)
                bg_depth = perturb_samples(bg_depth)  # random perturbation during training
            else:
                # sample pdf and concat with earlier samples
                fg_weights = ret['fg_weights'].clone().detach()
                fg_depth_mid = .5 * (fg_depth[..., 1:] + fg_depth[..., :-1])  # [..., N_samples-1]
                fg_weights = fg_weights[..., 1:-1]  # [..., N_samples-2]
                fg_depth_samples = sample_pdf(bins=fg_depth_mid, weights=fg_weights,
                                              N_samples=N_samples, det=False)  # [..., N_samples]
                fg_depth, _ = torch.sort(torch.cat((fg_depth, fg_depth_samples), dim=-1))

                # sample pdf and concat with earlier samples
                bg_weights = ret['bg_weights'].clone().detach()
                bg_depth_mid = .5 * (bg_depth[..., 1:] + bg_depth[..., :-1])
                bg_weights = bg_weights[..., 1:-1]  # [..., N_samples-2]
                bg_depth_samples = sample_pdf(bins=bg_depth_mid, weights=bg_weights,
                                              N_samples=N_samples, det=False)  # [..., N_samples]
                bg_depth, _ = torch.sort(torch.cat((bg_depth, bg_depth_samples), dim=-1))

            optim.zero_grad()
            ret = net(ray_batch['ray_o'], ray_batch['ray_d'], fg_far_depth, fg_depth,
                      bg_depth, global_step, img_name=ray_batch['img_name'])
            all_rets.append(ret)

            rgb_gt = ray_batch['rgb'].to(rank)

            if ray_batch['mask'] is not None:
                mask = ray_batch['mask'].to(rank)
            else:
                mask = None

            if 'autoexpo' in ret:
                assert (False)
                scale, shift = ret['autoexpo']
                scalars_to_log['level_{}/autoexpo_scale'.format(m)] = scale.item()
                scalars_to_log['level_{}/autoexpo_shift'.format(m)] = shift.item()
                # rgb_gt = scale * rgb_gt + shift
                rgb_pred = (ret['rgb'] - shift) / scale
                rgb_loss = img2mse(rgb_pred, rgb_gt)

                if args.normal_loss_weight != -1:
                    # ret['fg_normal_map_postintegral'] and 'fg_normal' cannot be detached!!!!!
                    # important !!!
                    fg_normal_map_postintegral = ret['fg_normal_map_postintegral']
                    fg_normal = ret['fg_normal'].unsqueeze(-2).expand(ret['fg_normal_map_postintegral'].shape)
                    cosine = torch.sum(fg_normal_map_postintegral * fg_normal, dim=-1)
                    normal_direction_loss = (((1 - cosine)**2) * ret['fg_weights']).mean()
                    # normal_direction_loss = ((1 - cosine)) * ret['fg_weights']
                else:
                    normal_direction_loss = torch.tensor(0.0).cuda().float()

                """just avoid trivial solution at first 10 step, I don't even know why it have such huge effect...."""
                if global_step < 10:
                    loss = rgb_loss + args.lambda_autoexpo * (torch.abs(scale - 1.) + torch.abs(shift)) + \
                           (args.normal_loss_weight / 100000) * normal_direction_loss
                else:
                    loss = rgb_loss + args.lambda_autoexpo * (torch.abs(scale - 1.) + torch.abs(shift)) + \
                           args.normal_loss_weight * normal_direction_loss
            else:
                # zeros = ret['rgb'] * 0
                # rgb_loss = img2mse(ret['pure_rgb'], torch.min(rgb_gt/ret['shadow'], ones))
                # rgb_loss = img2mse(ret['pure_rgb'], torch.max(rgb_gt/ret['shadow'], zeros))
                # rgb_loss = img2mse(ret['rgb'], rgb_gt)
                rgb_loss = img2mse(ret['rgb'], rgb_gt, mask=mask)
                shadow_reg = torch.mean((1 - ret['shadow']) ** 2)
                if not args.use_shadow_reg:
                    shadow_reg = 0 * shadow_reg

                if args.normal_loss_weight != -1:
                    # ret['fg_normal_map_postintegral'] and 'fg_normal' cannot be detached!!!!!
                    # important !!!
                    fg_normal_map_postintegral = ret['fg_normal_map_postintegral']
                    fg_normal = ret['fg_normal'].unsqueeze(-2).expand(ret['fg_normal_map_postintegral'].shape)
                    cosine = torch.sum(fg_normal_map_postintegral * fg_normal, dim=-1)
                    normal_direction_loss = (((1 - cosine)**2) * ret['fg_weights']).mean()
                    # normal_direction_loss = ((1 - cosine)) * ret['fg_weights']
                else:
                    normal_direction_loss = torch.tensor(0.0).cuda().float()

                """just avoid trivial solution at first 10 step, I don't even know why it have such huge effect...."""
                if global_step < 10:
                    loss = rgb_loss + (shadow_reg * args.shadow_reg) * np.clip((global_step - 10000) / 20000, 0, 1) + \
                           (args.normal_loss_weight / 100000) * normal_direction_loss
                else:
                    loss = rgb_loss + (shadow_reg * args.shadow_reg) * np.clip((global_step - 10000) / 20000, 0, 1) + \
                           args.normal_loss_weight * normal_direction_loss
            # record loss curve
            if rank == 0 and (global_step % args.i_print == 0 or global_step < 10):
                writer_loss.add_scalar('level{}'.format(m) + 'rgb_loss', rgb_loss.item(), global_step)
                writer_loss.add_scalar('level{}'.format(m) + 'pnsr', mse2psnr(rgb_loss.item()), global_step)
                writer_loss.add_scalar('level{}'.format(m) + 'shadow_reg', shadow_reg.item(), global_step)
                writer_loss.add_scalar('level{}'.format(m) + 'normal_direction_loss', normal_direction_loss.item(), global_step)
                scalars_to_log['level_{}/loss'.format(m)] = rgb_loss.item()
                scalars_to_log['level_{}/pnsr'.format(m)] = mse2psnr(rgb_loss.item())
                scalars_to_log['level_{}/shadow_reg'.format(m)] = shadow_reg.item()
                scalars_to_log['level_{}/normal_direction_loss'.format(m)] = normal_direction_loss.item()


            loss.backward()
            optim.step()

            # # clean unused memory
            torch.cuda.empty_cache()

        ### end of core optimization loop
        dt = time.time() - time0
        scalars_to_log['iter_time'] = dt

        # ForkedPdb().set_trace()
        ### only main process should do the logging
        if rank == 0 and (global_step % args.i_print == 0 or global_step < 10):
            logstr = '{} step: {} '.format(args.expname, global_step)
            for k in scalars_to_log:
                logstr += ' {}: {:.6f}'.format(k, scalars_to_log[k])
                writer.add_scalar(k, scalars_to_log[k], global_step)
            logger.info(logstr)

        if rank == 0 and (global_step % args.i_weights == 0 and global_step > 0):
            # saving checkpoints and logging
            fpath = os.path.join(args.basedir, args.expname, 'model_{:06d}.pth'.format(global_step))
            to_save = OrderedDict()
            for m in range(models['cascade_level']):
                name = 'net_{}'.format(m)
                to_save[name] = models[name].state_dict()

                name = 'optim_{}'.format(m)
                to_save[name] = models[name].state_dict()
            torch.save(to_save, fpath)

        ### each process should do this; but only main process merges the results
        if (global_step % args.i_img == 0 and global_step != 0) or (global_step == start + 1 and args.start_val):
            #### critical: make sure each process is working on the same random image
            time0 = time.time()
            idx = what_val_to_log % len(val_ray_samplers)

            # create dir of validation visualization
            output_dir = os.path.join(args.basedir, args.expname, 'step' + str(global_step))
            os.makedirs(output_dir, exist_ok=True)



            # change chunk_size for validation on 1080Ti
            ############################################
            if torch.cuda.get_device_properties(rank).total_memory / 1e9 > 9 and \
                    torch.cuda.get_device_properties(rank).total_memory / 1e9 < 30:
                    # torch.cuda.get_device_properties(rank).total_memory / 1e9 < 20: # two training processes on 3090
                logger.info('change chunk_size for validation part according to 12G gpu')
                args.chunk_size = int(args.chunk_size / 4)
            else:
                logger.info('original chunk_size for validation part according to 24G gpu')
                args.chunk_size = int(args.chunk_size / 2)

            ## debug: get SH and rot SH
            # SH = models['net_1'].module.env_params['train/rgb/26-04_19_00_DSC_2474-jpg'].cpu().detach().numpy()
            # imageio.imwrite('/home/youmingdeng/test.exr', SH)
            # rotate_SH(SH, 0., -np.pi/3, 0.)
            for i in range(36):
                time0 = time.time()
                angle = i * np.pi / 36
                reli_data = render_single_image(rank, args.world_size, models, val_ray_samplers[idx],
                                               args.chunk_size, global_step, angle)
                reli_dir = os.path.join(args.basedir, args.expname, 'step' + str(global_step) + '_rot')
                os.makedirs(reli_dir, exist_ok=True)
                if rank == 0:
                    for m in range(len(reli_data)):
                        rgb_im = (reli_data[m]['rgb'])
                        rgb_im = torch.clamp(rgb_im, min=0., max=1.)  # just in case diffuse+specular>1
                        # writer.add_image(prefix + 'level_{}/rgb'.format(m), rgb_im, global_step)
                        save_image(reli_dir + '/level_{}_rgb.png'.format(m), 255 * rgb_im.numpy())
                dt = time.time() - time0
                print(angle)
                print(dt)
            ForkedPdb().set_trace()
            ## debug: get SH and rot SH

            log_data = render_single_image(rank, args.world_size, models, val_ray_samplers[idx],
                                           args.chunk_size, global_step)

            what_val_to_log += 1
            dt = time.time() - time0

            if rank == 0:  # only main process should do this
                logger.info('Logged a random validation view in {} seconds'.format(dt))
                if 'home' not in output_dir:
                    log_view_to_tb(os.getcwd() + '/' + output_dir, writer, global_step,
                                   log_data, gt_img=val_ray_samplers[idx].get_img(),
                                   mask=None, prefix='/val_')
                else:
                    log_view_to_tb(output_dir, writer, global_step,
                                   log_data, gt_img=val_ray_samplers[idx].get_img(),
                                   mask=None, prefix='/val_')

            time0 = time.time()
            idx = what_train_to_log % len(ray_samplers)

            log_data = render_single_image(rank, args.world_size, models, ray_samplers[idx], args.chunk_size,
                                           global_step)

            if torch.cuda.get_device_properties(rank).total_memory / 1e9 > 9 and \
                    torch.cuda.get_device_properties(rank).total_memory / 1e9 < 30:
                    # torch.cuda.get_device_properties(rank).total_memory / 1e9 < 20:
                logger.info('change back!')
                args.chunk_size = int(args.chunk_size * 4)
            else:
                logger.info('original chunk_size for validation part according to 24G gpu')
                args.chunk_size = int(args.chunk_size * 2)
            # change back chunk_size for validation on 1080Ti
            ###############################################

            what_train_to_log += 1
            dt = time.time() - time0
            if rank == 0:  # only main process should do this
                logger.info('Logged a random training view in {} seconds'.format(dt))
                if 'home' not in output_dir:
                    log_view_to_tb(os.getcwd() + '/' + output_dir, writer, global_step,
                                   log_data, gt_img=ray_samplers[idx].get_img(),
                                   mask=None, prefix='/train_')
                else:
                    log_view_to_tb(output_dir, writer, global_step,
                                   log_data, gt_img=ray_samplers[idx].get_img(),
                                   mask=None, prefix='/train_')

            del log_data
            torch.cuda.empty_cache()


    # clean up for multi-processing
    if one_card == False:
        cleanup()


def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()

    def str2bool(v):
        if isinstance(v, bool):
            return v
        elif v == 'True':
            return True
        elif v == 'False':
            return False
        else:
            raise configargparse.ArgumentTypeError('Boolean value expected.')

    parser.add_argument('--config', is_config_file=True, help='config file path')
    parser.add_argument("--expname", type=str, help='experiment name')
    parser.add_argument("--slurmjob", type=str, default='', help='slurm job id')
    parser.add_argument("--basedir", type=str, default='./logs/', help='where to store ckpts and logs')

    # ablation options
    parser.add_argument("--use_shadows", type=str2bool, default=True)
    parser.add_argument("--use_shadow_reg", type=str2bool, default=True)
    parser.add_argument("--shadow_reg", type=float, default=0.01, help='shadow regulariser strength')
    parser.add_argument("--use_shadow_jitter", type=str2bool, default=True)
    parser.add_argument("--use_annealing", type=str2bool, default=True)
    parser.add_argument("--use_ray_jitter", type=str2bool, default=True)

    # test options
    parser.add_argument("--rotate_test_env", action='store_true', help='rotate test env (timelapse)')
    parser.add_argument("--test_env", type=str, default=None, help='which environment to use for test render')
    # dataset options
    parser.add_argument("--datadir", type=str, default=None, help='input data directory')
    parser.add_argument("--scene", type=str, default=None, help='scene name')
    parser.add_argument("--testskip", type=int, default=8,
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    # model size
    parser.add_argument("--netdepth", type=int, default=8, help='layers in coarse network')
    parser.add_argument("--netwidth", type=int, default=256, help='channels per layer in coarse network')
    parser.add_argument("--use_viewdirs", action='store_true', help='use full 5D input instead of 3D')
    parser.add_argument("--activation", type=str, default=None, help='activation function (relu, elu, sine)')

    parser.add_argument("--with_bg", action='store_true', help='use background network')

    # checkpoints
    parser.add_argument("--no_reload", action='store_true', help='do not reload weights from saved ckpt')
    parser.add_argument("--ckpt_path", type=str, default=None,
                        help='specific weights npy file to reload for coarse network')

    # batch size
    parser.add_argument("--N_rand", type=int, default=32 * 32 * 2,
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--chunk_size", type=int, default=1024 * 8,
                        help='number of rays processed in parallel, decrease if running out of memory')

    # iterations
    parser.add_argument("--N_iters", type=int, default=250001,
                        help='number of iterations')

    # render only
    parser.add_argument("--render_splits", type=str, default='test',
                        help='splits to render')
    parser.add_argument("--resolution_level", type=float, default=1,
                        help='resolution multiplier')

    # cascade training
    parser.add_argument("--cascade_level", type=int, default=2,
                        help='number of cascade levels')
    parser.add_argument("--cascade_samples", type=str, default='64,64',
                        help='samples at each level')

    # multiprocess learning
    parser.add_argument("--world_size", type=int, default='-1',
                        help='number of processes')
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='node rank for distributed training')

    # optimize autoexposure
    parser.add_argument("--optim_autoexpo", action='store_true',
                        help='optimize autoexposure parameters')
    parser.add_argument("--lambda_autoexpo", type=float, default=1., help='regularization weight for autoexposure')

    # learning rate options
    parser.add_argument("--lrate", type=float, default=5e-4, help='learning rate')
    parser.add_argument("--lrate_decay_factor", type=float, default=0.1,
                        help='decay learning rate by a factor every specified number of steps')
    parser.add_argument("--lrate_decay_steps", type=int, default=5000,
                        help='decay learning rate by a factor every specified number of steps')

    # rendering options
    parser.add_argument("--det", action='store_true', help='deterministic sampling for coarse and fine samples')
    parser.add_argument("--max_freq_log2", type=int, default=10,
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--max_freq_log2_viewdirs", type=int, default=4,
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--N_anneal", type=int, default=100000,
                        help='number of embedder anneal iterations')
    parser.add_argument("--N_anneal_min_freq", type=int, default=0,
                        help='number of embedder frequencies to start annealing from')
    parser.add_argument("--N_anneal_min_freq_viewdirs", type=int, default=0,
                        help='number of viewdir embedder frequencies to start annealing from')
    parser.add_argument("--load_min_depth", action='store_true', help='whether to load min depth')

    # logging/saving options
    parser.add_argument("--i_print", type=int, default=100, help='frequency of console printout and metric logging')
    parser.add_argument("--i_img", type=int, default=500, help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=10000, help='frequency of weight ckpt saving')

    # youming options
    parser.add_argument("--normal_loss_weight", type=float, default=-1, help='normal direction loss weight')
    parser.add_argument("--master_port", type=int, default=12222, help='master_port of the program')
    parser.add_argument("--start_val", default = False, action = "store_true", help = 'if reload, start validation at start+1 step')

    return parser


def train():
    parser = config_parser()
    args = parser.parse_args()
    if 'SLURM_JOB_ID' in os.environ:
        args.slurmjob = os.environ['SLURM_JOB_ID']
    logger.info(parser.format_values())

    if args.world_size == -1:
        args.world_size = torch.cuda.device_count()
        logger.info('Using # gpus: {}'.format(args.world_size))

    torch.multiprocessing.spawn(ddp_train_nerf,
                                args=(args,),
                                nprocs=args.world_size,
                                join=True)

def train_one_card():
    parser = config_parser()
    args = parser.parse_args()
    if 'SLURM_JOB_ID' in os.environ:
        args.slurmjob = os.environ['SLURM_JOB_ID']
    logger.info(parser.format_values())

    if args.world_size == -1:
        args.world_size = torch.cuda.device_count()
        logger.info('Using # gpus: {}'.format(1))

    args.world_size = 1
    ddp_train_nerf(rank=args.local_rank, args=args, one_card=True)

if __name__ == '__main__':
    # u = np.ones((2, 2, 2), dtype=np.float32)
    # u[u==1] = 100
    # u[0][0][0] = 10
    # vertices, triangles = mcubes.marching_cubes(u, 50)
    # mcubes.export_obj(vertices, triangles, '/home/youmingdeng/tri.obj')
    # ForkedPdb().set_trace()
    setup_logger()
    train()
    # train_one_card()
