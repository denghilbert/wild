import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import sys
import pdb

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
from go_model.decoder import NeRFDecoder
# from multi_grid import MultiGrid
from go_model.utils import compute_world_dims, coordinates

from grid_function.hashencoder.hashgrid import _hash_encode, HashEncoder
from grid_function.ray_sampler import ErrorBoundSampler

class SDFGridModel(torch.nn.Module):
    def __init__(self,
                 config,
                 ):
        super(SDFGridModel, self).__init__()

        base_size = 16  # 64,
        end_size = 2048  # 8192,
        logmap = 19  # 2,
        num_levels = 16
        level_dim = 2  # 8,
        print(f"using hash encoder with {num_levels} levels, each level with feature dim {level_dim}")
        print(f"resolution:{base_size} -> {end_size} with hash map size {logmap}")
        self.grid = HashEncoder(input_dim=3, num_levels=num_levels, level_dim=level_dim,
                                    per_level_scale=2, base_resolution=base_size,
                                    log2_hashmap_size=logmap, desired_resolution=end_size)
        """
        (Pdb) self.grid
        MultiGrid(
          (volumes): ParameterList(
              (0): Parameter containing: [torch.cuda.FloatTensor of size 1x4x129x97x161 (GPU 0)]
              (1): Parameter containing: [torch.cuda.FloatTensor of size 1x10x65x49x81 (GPU 0)]
              (2): Parameter containing: [torch.cuda.FloatTensor of size 1x4x17x13x21 (GPU 0)]
              (3): Parameter containing: [torch.cuda.FloatTensor of size 1x4x5x4x6 (GPU 0)]
          )
        """
        sdf_args = {'W': 32, 'D': 2, 'skips': [], 'n_freq': -1, 'weight_norm': False, 'concat_qp': False}
        rgb_args = {'W': 32, 'D': 2, 'skips': [], 'use_view_dirs': True, 'use_normals': False, 'use_dot_prod': False, 'n_freq': -1, 'weight_norm': False, 'concat_qp': False}
        self.decoder = NeRFDecoder(sdf_args,
                                   rgb_args,
                                   sdf_feat_dim=32,
                                   rgb_feat_dim=42-3)
        """
        (Pdb) config["decoder"]["geometry"], config["decoder"]["radiance"], sum(config["sdf_feature_dim"]), sum(config["rgb_feature_dim"])
        ({'W': 32, 'D': 2, 'skips': [], 'n_freq': -1, 'weight_norm': False, 'concat_qp': False}, 
        {'W': 32, 'D': 2, 'skips': [], 'use_view_dirs': True, 'use_normals': False, 'use_dot_prod': False, 'n_freq': -1, 'weight_norm': False, 'concat_qp': False}, 
        16, 
        6)
        """
        self.sdf_decoder = batchify(self.decoder.geometry_net, max_chunk=None)
        self.rgb_decoder = batchify(self.decoder.radiance_net, max_chunk=None)

        # Inverse sigma from NeuS paper
        self.inv_s = nn.parameter.Parameter(torch.tensor(0.3))

        ray_sampler_near = 0.0
        ray_sampler_N_samples = 64
        ray_sampler_N_samples_eval = 128
        ray_sampler_N_samples_extra = 32
        ray_sampler_eps = 0.1
        ray_sampler_beta_iters = 10
        ray_sampler_max_total_iters = 5
        self.scene_bounding_sphere = 1.  # conf.get_float('scene_bounding_sphere', default=1.0)
        self.ray_sampler = ErrorBoundSampler(
            self.scene_bounding_sphere, ray_sampler_near, ray_sampler_N_samples, ray_sampler_N_samples_eval,
            ray_sampler_N_samples_extra, ray_sampler_eps, ray_sampler_beta_iters, ray_sampler_max_total_iters)

    def forward(self, ray_o, ray_d, fg_z_max, fg_z_vals, bg_z_vals, env, iteration, c2w, intrinsic, ray_matrix, validation=False):
        ray_dirs, cam_loc = ray_d.unsqueeze(0), ray_o[:1, :]
        batch_size, num_pixels, _ = ray_dirs.shape

        cam_loc = cam_loc.unsqueeze(1).repeat(1, num_pixels, 1).reshape(-1, 3)
        ray_dirs = ray_dirs.reshape(-1, 3)

        z_vals = fg_z_vals
        N_samples = z_vals.shape[1]
        dists = z_vals[:, 1:] - z_vals[:, :-1]
        dists = torch.cat([dists, torch.ones(z_vals.shape[0], 1).cuda()], dim=1)
        z_vals_mid = z_vals + dists * 0.5
        """
        (Pdb) dirs.shape
        torch.Size([1024, 4096, 3])
        (Pdb) points_flat.shape
        torch.Size([4194304, 3])
        """
        points = cam_loc.unsqueeze(1) + z_vals.unsqueeze(2) * ray_dirs.unsqueeze(1)
        # points_flat = points.reshape(-1, 3)
        dirs = ray_dirs.unsqueeze(1).repeat(1, N_samples, 1)
        # dirs_flat = dirs.reshape(-1, 3)

        rend_dict = render_rays(self.sdf_decoder,
                                self.rgb_decoder,
                                self.grid,
                                # points_flat,
                                points,
                                # dirs_flat,
                                dirs,
                                dists,
                                z_vals_mid,
                                z_vals,
                                iter=iteration,
                                inv_s=torch.exp(10. * self.inv_s),
                                validation=validation)

        return rend_dict


def batchify(fn, max_chunk=1024 * 128):
    if max_chunk is None:
        return fn

    def ret(feats):
        chunk = max_chunk // (feats.shape[1] * feats.shape[2])
        return torch.cat([fn(feats[i:i + chunk]) for i in range(0, feats.shape[0], chunk)], dim=0)

    return ret


def render_rays(sdf_decoder,
                rgb_decoder,
                feat_volume,  # regualized feature volume [1, feat_dim, Nx, Ny, Nz]
                # points_flat,
                points,
                view_dirs,
                dists,
                z_vals_mid,
                z_vals,
                near=0.01,
                far=3.0,
                n_samples=128,
                n_importance=16,
                depth_gt=None,
                inv_s=20.,
                normals_gt=None,
                smoothness_std=0.0,
                randomize_samples=True,
                use_view_dirs=False,
                use_normals=False,
                concat_qp_to_rgb=False,
                concat_qp_to_sdf=False,
                concat_dot_prod_to_rgb=False,
                iter=0,
                rgb_feature_dim=[],
                validation=False
                ):
    query_points = points
    # important !!! we should enable_grad within no_grad wrap
    if validation == True:
        with torch.enable_grad():
            query_points = query_points.requires_grad_(True)
            sdf, rgb_feat, grads = qp_to_sdf_rgb_feat(query_points, feat_volume, sdf_decoder)
    else:
        query_points = query_points.requires_grad_(True)
        sdf, rgb_feat, grads = qp_to_sdf_rgb_feat(query_points, feat_volume, sdf_decoder)

    rgb_feat = [rgb_feat]

    rgb_feat.append(view_dirs)
    rgb_feat.append(grads)
    rgb_feat.append(query_points)
    rgb_feat.append((view_dirs * grads).sum(dim=-1, keepdim=True))
    rgb = torch.sigmoid(rgb_decoder(torch.cat(rgb_feat, dim=-1)))

    cos_val = (view_dirs * grads).sum(-1)
    # cos_val = -F.relu(-cos_val)
    cos_anneal_ratio = min(iter / 5000., 1.)
    cos_val = -(F.relu(-cos_val * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) +
                F.relu(-cos_val) * cos_anneal_ratio)

    weights = neus_weights(sdf[:, :, 0], dists, inv_s, cos_val)
    rendered_rgb = torch.sum(weights[..., None] * rgb, dim=-2)
    # rendered_depth = torch.sum(weights * z_vals_mid, dim=-1)
    rendered_depth = torch.sum(weights * z_vals, dim=-1)
    # depth_var = torch.sum(weights * torch.square(z_vals_mid - rendered_depth.unsqueeze(-1)), dim=-1)
    normalized_grads = F.normalize(grads, p=2, dim=-1)
    rendered_normal = torch.sum(weights[..., None] * normalized_grads, dim=-2)
    ForkedPdb().set_trace()

    ret = {"rgb": rendered_rgb,
           "depth": rendered_depth,
           "sdf": sdf,
           "normal": rendered_normal,
           # "sdf_loss": sdf_loss,
           # "fs_loss": fs_loss,
           # "sdfs": sdf,
           "weights": weights,
           # "normal_regularisation_loss": normal_regularisation_loss,
           # "eikonal_loss": eikonal_loss,
           # "normal_supervision_loss": normal_supervision_loss,
           }
    return ret


mse2psnr = lambda x: -10. * torch.log(x) / torch.log(torch.Tensor([10.])).to(x)


def qp_to_sdf_rgb_feat(pts, feat_volume, sdf_decoder, divide_factor=1.5):
    feature = feat_volume(pts / divide_factor)
    sdf = sdf_decoder(feature)

    d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
    gradients = torch.autograd.grad(
        outputs=sdf,
        inputs=pts,
        grad_outputs=d_output,
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0]

    return sdf, feature, gradients


def qp_to_sdf(pts, volume_origin, volume_dim, feat_volume, sdf_decoder, sdf_act=nn.Identity(), concat_qp=False,
              rgb_feature_dim=[]):
    # Normalize point cooridnates and mask out out-of-bounds points
    pts_norm = 2. * (pts - volume_origin[None, None, :]) / volume_dim[None, None, :] - 1.
    mask = (pts_norm.abs() <= 1.).all(dim=-1)
    pts_norm = pts_norm[mask].unsqueeze(0).unsqueeze(0).unsqueeze(0)
    """
    4, 10, 4, 4
    """
    mlvl_feats = feat_volume(pts_norm[..., [2, 1, 0]], concat=False)
    sdf_feats = list(map(lambda feat_pts, rgb_dim: feat_pts[:, :-rgb_dim, ...] if rgb_dim > 0 else feat_pts,
                         mlvl_feats, rgb_feature_dim))
    sdf_feats = torch.cat(sdf_feats, dim=1).squeeze(0).squeeze(1).squeeze(1).t()

    rgb_feats = map(lambda feat_pts, rgb_dim: feat_pts[:, -rgb_dim:, ...] if rgb_dim > 0 else None,
                    mlvl_feats, rgb_feature_dim)
    rgb_feats = list(filter(lambda x: x is not None, rgb_feats))
    rgb_feats = torch.cat(rgb_feats, dim=1).squeeze(0).squeeze(1).squeeze(1).t()

    rgb_feats_unmasked = torch.zeros(list(mask.shape) + [sum(rgb_feature_dim)], device=pts_norm.device)
    rgb_feats_unmasked[mask] = rgb_feats

    if concat_qp:
        sdf_feats.append(pts_norm.permute(0, 4, 1, 2, 3))
    """
    4 res, with 4 4 4 4 = 16
    """
    raw = sdf_decoder(sdf_feats)
    sdf = torch.zeros_like(mask, dtype=pts_norm.dtype)
    sdf[mask] = sdf_act(raw.squeeze(-1))
    """
    with only 6
    """
    return sdf, rgb_feats_unmasked, mask


def neus_weights(sdf, dists, inv_s, cos_val, z_vals=None):
    estimated_next_sdf = sdf + cos_val * dists * 0.5
    estimated_prev_sdf = sdf - cos_val * dists * 0.5

    prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
    next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

    p = prev_cdf - next_cdf
    c = prev_cdf

    alpha = ((p + 1e-5) / (c + 1e-5)).clip(0.0, 1.0)
    weights = alpha * torch.cumprod(
        torch.cat([torch.ones([sdf.shape[0], 1], device=alpha.device), 1. - alpha + 1e-7], -1), -1)[:, :-1]

    if z_vals is not None:
        signs = sdf[:, 1:] * sdf[:, :-1]
        mask = torch.where(signs < 0., torch.ones_like(signs), torch.zeros_like(signs))
        # This will only return the first zero-crossing
        inds = torch.argmax(mask, dim=1, keepdim=True)
        z_surf = torch.gather(z_vals, 1, inds)
        return weights, z_surf

    return weights


def apply_log_transform(tsdf):
    sgn = torch.sign(tsdf)
    out = torch.log(torch.abs(tsdf) + 1)
    out = sgn * out
    return out


def compute_loss(prediction, target, loss_type="l2"):
    if loss_type == "l2":
        return F.mse_loss(prediction, target)
    elif loss_type == "l1":
        return F.l1_loss(prediction, target)
    elif loss_type == "log":
        return F.l1_loss(apply_log_transform(prediction), apply_log_transform(target))
    raise Exception("Unknown loss type")


def compute_grads(predicted_sdf, query_points):
    sdf_grad, = torch.autograd.grad([predicted_sdf], [query_points], [torch.ones_like(predicted_sdf)],
                                    create_graph=True)
    return sdf_grad


def get_sdf_loss(z_vals, target_d, predicted_sdf, truncation):
    depth_mask = target_d > 0.
    front_mask = (z_vals < (target_d - truncation))
    # bask_mask = (z_vals > (target_d + truncation)) & depth_mask
    front_mask = (front_mask | ((target_d < 0.) & (z_vals < 3.5)))
    bound = (target_d - z_vals)
    bound[target_d[:, 0] < 0., :] = 10.  # TODO: maybe use noisy depth for bound?
    sdf_mask = (bound.abs() <= truncation) & depth_mask

    sum_of_samples = front_mask.sum(dim=-1) + sdf_mask.sum(dim=-1) + 1e-8
    rays_w_depth = torch.count_nonzero(target_d)

    fs_loss = (torch.max(torch.exp(-5. * predicted_sdf) - 1., predicted_sdf - bound).clamp(min=0.) * front_mask)
    fs_loss = (fs_loss.sum(dim=-1) / sum_of_samples).sum() / rays_w_depth
    sdf_loss = ((torch.abs(predicted_sdf - bound) * sdf_mask).sum(dim=-1) / sum_of_samples).sum() / rays_w_depth

    return fs_loss, sdf_loss


def sample_pdf(bins, weights, N_importance, det=False, eps=1e-5):
    # device = weights.get_device()
    device = weights.device
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1], device=device), cdf], -1)
    # Take uniform samples
    if det:
        u = torch.linspace(0. + 0.5 / N_importance, 1. - 0.5 / N_importance, steps=N_importance, device=device)
        u = u.expand(list(cdf.shape[:-1]) + [N_importance])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_importance], device=device)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom, device=device), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples