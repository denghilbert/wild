import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import glob
import os, pdb, sys
from utils import TINY_NUMBER, HUGE_NUMBER
from collections import OrderedDict
from nerf_network import Embedder, MLPNet
from sph_util import illuminate_vec, rotate_env
import logging
import mcubes
#
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


from grid_function.hashencoder.hashgrid import _hash_encode, HashEncoder
from grid_function.density import LaplaceDensity
from grid_function.embedder import *
from grid_function.ray_sampler import ErrorBoundSampler

class ImplicitNetworkGrid(nn.Module):
    def __init__(
            self,
            feature_vector_size,
            d_in,
            d_out,
            dims,
            geometric_init=True,
            bias=1.0,
            skip_in=(),
            weight_norm=True,
            multires=0,
            sphere_scale=1.0,
            inside_outside=False,
            base_size=16,
            end_size=2048,
            logmap=19,
            num_levels=16,
            level_dim=2,
            divide_factor=1.5,  # used to normalize the points range for multi-res grid
            use_grid_feature=True
    ):
        super().__init__()

        self.sphere_scale = sphere_scale
        dims = [d_in] + dims + [d_out + feature_vector_size]
        self.embed_fn = None
        self.divide_factor = divide_factor
        self.grid_feature_dim = num_levels * level_dim
        self.use_grid_feature = use_grid_feature
        dims[0] += self.grid_feature_dim

        print(f"using hash encoder with {num_levels} levels, each level with feature dim {level_dim}")
        print(f"resolution:{base_size} -> {end_size} with hash map size {logmap}")
        self.encoding = HashEncoder(input_dim=3, num_levels=num_levels, level_dim=level_dim,
                                    per_level_scale=2, base_resolution=base_size,
                                    log2_hashmap_size=logmap, desired_resolution=end_size)

        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn = embed_fn
            dims[0] += input_ch - 3
        print("network architecture")
        print(dims)

        self.num_layers = len(dims)
        self.skip_in = skip_in

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    if not inside_outside:
                        torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, -bias)
                    else:
                        torch.nn.init.normal_(lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, bias)

                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.softplus = nn.Softplus(beta=100)
        self.cache_sdf = None

    def forward(self, input1):
        if self.use_grid_feature:
            # normalize point range as encoding assume points are in [-1, 1]
            feature = self.encoding(input1 / self.divide_factor)
        else:
            feature = torch.zeros_like(input1[:, :1].repeat(1, self.grid_feature_dim))

        if self.embed_fn is not None:
            embed = self.embed_fn(input1)
            input1 = torch.cat((embed, feature), dim=-1)
        else:
            input1 = torch.cat((input1, feature), dim=-1)

        x = input1

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, input1], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.softplus(x)

        return x

    def gradient(self, x):
        x.requires_grad_(True)
        y = self.forward(x)[:, :1]
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients

    def get_outputs(self, x):
        x.requires_grad_(True)
        output = self.forward(x)
        sdf = output[:, :1]

        feature_vectors = output[:, 1:]
        d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
        gradients = torch.autograd.grad(
            outputs=sdf,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]

        return sdf, feature_vectors, gradients

    def get_sdf_vals(self, x):
        sdf = self.forward(x)[:, :1]
        return sdf

    def mlp_parameters(self):
        parameters = []
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
            parameters += list(lin.parameters())
        return parameters

    def grid_parameters(self):
        print("grid parameters", len(list(self.encoding.parameters())))
        for p in self.encoding.parameters():
            print(p.shape)
        return self.encoding.parameters()


class RenderingNetwork(nn.Module):
    def __init__(
            self,
            feature_vector_size,
            mode,
            d_in,
            d_out,
            dims,
            weight_norm=True,
            multires_view=0,
            per_image_code=False
    ):
        super().__init__()

        self.mode = mode
        dims = [d_in + feature_vector_size] + dims + [d_out]

        self.embedview_fn = None
        if multires_view > 0:
            embedview_fn, input_ch = get_embedder(multires_view)
            self.embedview_fn = embedview_fn
            dims[0] += (input_ch - 3)

        self.per_image_code = per_image_code
        if self.per_image_code:
            # nerf in the wild parameter
            # parameters
            # maximum 1024 images
            self.embeddings = nn.Parameter(torch.empty(1024, 32))
            std = 1e-4
            self.embeddings.data.uniform_(-std, std)
            dims[0] += 32

        print("rendering network architecture:")
        print(dims)

        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.relu = nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, points, normals, view_dirs, feature_vectors, env):
        if self.embedview_fn is not None:
            view_dirs = self.embedview_fn(view_dirs)

        if self.mode == 'idr':
            rendering_input = torch.cat([points, view_dirs, normals, feature_vectors], dim=-1)
        elif self.mode == 'nerf':
            rendering_input = torch.cat([view_dirs, feature_vectors], dim=-1)
        else:
            raise NotImplementedError

        if self.per_image_code:
            # TODO: use env for each image
            print("we haven't implemented env code for each image")
            raise NotImplementedError
            # image_code = self.embeddings[indices].expand(rendering_input.shape[0], -1)
            # rendering_input = torch.cat([rendering_input, image_code], dim=-1)

        x = rendering_input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.relu(x)

        x = self.sigmoid(x)
        return x


class GridNerfNet(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.feature_vector_size = 256 #conf.get_int('feature_vector_size')
        self.scene_bounding_sphere = 1. #conf.get_float('scene_bounding_sphere', default=1.0)
        self.white_bkgd = False #conf.get_bool('white_bkgd', default=False)
        self.bg_color = torch.tensor([1.0, 1.0, 1.0]).float().cuda()#torch.tensor(conf.get_list("bg_color", default=[1.0, 1.0, 1.0])).float().cuda()

        grid_net_d_in = 3
        grid_net_d_out = 1
        grid_net_dims = [256, 256]
        grid_net_geometric_init = True
        grid_net_bias = 0.6
        grid_net_skip_in = [4]
        grid_net_weight_norm = True
        grid_net_multires = 6
        grid_net_inside_outside = True
        grid_net_use_grid_feature = True
        grid_net_divide_factor = 1.0
        self.implicit_network = ImplicitNetworkGrid(
            self.feature_vector_size, grid_net_d_in, grid_net_d_out, grid_net_dims,
            grid_net_geometric_init, grid_net_bias, grid_net_skip_in, grid_net_weight_norm,
            grid_net_multires, grid_net_inside_outside, use_grid_feature=grid_net_use_grid_feature,
            divide_factor=grid_net_divide_factor
        )
        render_mode = 'idr'
        render_d_in = 9
        render_d_out = 3
        render_dims = [256, 256]
        render_weight_norm = True
        render_multires_view = 4
        render_per_image_code = False #True
        self.rendering_network = RenderingNetwork(
            self.feature_vector_size, render_mode, render_d_in, render_d_out, render_dims,
            render_weight_norm, render_multires_view, render_per_image_code
        )
        grid_beta = {'beta': 0.1}
        grid_beta_min = 0.0001
        self.density = LaplaceDensity(grid_beta, grid_beta_min)
        ray_sampler_near = 0.0
        ray_sampler_N_samples = 64
        ray_sampler_N_samples_eval = 128
        ray_sampler_N_samples_extra = 32
        ray_sampler_eps = 0.1
        ray_sampler_beta_iters = 10
        ray_sampler_max_total_iters = 5
        self.ray_sampler = ErrorBoundSampler(
            self.scene_bounding_sphere, ray_sampler_near, ray_sampler_N_samples, ray_sampler_N_samples_eval,
            ray_sampler_N_samples_extra, ray_sampler_eps, ray_sampler_beta_iters, ray_sampler_max_total_iters)

    def forward(self, ray_o, ray_d, fg_z_max, fg_z_vals, bg_z_vals, env, iteration, c2w, intrinsic, ray_matrix):
        ray_dirs, cam_loc = ray_d.unsqueeze(0), ray_o[:1, :]

        # TODO: use normalized ray direction for depth
        """how to use normalized ray direction for depth"""
        # we should use unnormalized ray direction for depth
        # ray_dirs_tmp, _ = rend_util.get_camera_params(uv, torch.eye(4).to(pose.device)[None], intrinsics)
        # depth_scale = ray_dirs_tmp[0, :, 2:]


        batch_size, num_pixels, _ = ray_dirs.shape

        cam_loc = cam_loc.unsqueeze(1).repeat(1, num_pixels, 1).reshape(-1, 3)
        ray_dirs = ray_dirs.reshape(-1, 3)

        """no sure which one to use"""
        z_vals, z_samples_eik = self.ray_sampler.get_z_vals(ray_dirs, cam_loc, self)
        # z_vals = fg_z_vals
        N_samples = z_vals.shape[1]

        points = cam_loc.unsqueeze(1) + z_vals.unsqueeze(2) * ray_dirs.unsqueeze(1)
        points_flat = points.reshape(-1, 3)

        dirs = ray_dirs.unsqueeze(1).repeat(1, N_samples, 1)
        dirs_flat = dirs.reshape(-1, 3)

        sdf, feature_vectors, gradients = self.implicit_network.get_outputs(points_flat)

        rgb_flat = self.rendering_network(points_flat, gradients, dirs_flat, feature_vectors, env)
        rgb = rgb_flat.reshape(-1, N_samples, 3)

        weights = self.volume_rendering(z_vals, sdf)

        rgb_values = torch.sum(weights.unsqueeze(-1) * rgb, 1)

        depth_values = torch.sum(weights * z_vals, 1, keepdims=True) / (weights.sum(dim=1, keepdims=True) + 1e-8)
        # we should scale rendered distance to depth along z direction
        # TODO
        # depth_values = depth_scale * depth_values

        # white background assumption
        if self.white_bkgd:
            acc_map = torch.sum(weights, -1)
            rgb_values = rgb_values + (1. - acc_map[..., None]) * self.bg_color.unsqueeze(0)

        # TODO: depth_values and depth_scale do not have depth scale
        output = {
            'rgb': rgb,
            'rgb_values': rgb_values,
            'depth_values': depth_values,
            'z_vals': z_vals,
            'depth_vals': z_vals,# * depth_scale,
            'sdf': sdf.reshape(z_vals.shape),
            'weights': weights,
        }

        if self.training:
            # Sample points for the eikonal loss
            n_eik_points = batch_size * num_pixels

            eikonal_points = torch.empty(n_eik_points, 3).uniform_(-self.scene_bounding_sphere,
                                                                   self.scene_bounding_sphere).cuda()

            # add some of the near surface points
            eik_near_points = (cam_loc.unsqueeze(1) + z_samples_eik.unsqueeze(2) * ray_dirs.unsqueeze(1)).reshape(-1, 3)
            eikonal_points = torch.cat([eikonal_points, eik_near_points], 0)
            # add some neighbour points as unisurf
            neighbour_points = eikonal_points + (torch.rand_like(eikonal_points) - 0.5) * 0.01
            eikonal_points = torch.cat([eikonal_points, neighbour_points], 0)

            grad_theta = self.implicit_network.gradient(eikonal_points)

            # split gradient to eikonal points and heighbour ponits
            output['grad_theta'] = grad_theta[:grad_theta.shape[0] // 2]
            output['grad_theta_nei'] = grad_theta[grad_theta.shape[0] // 2:]

        # compute normal map
        normals = gradients / (gradients.norm(2, -1, keepdim=True) + 1e-6)
        normals = normals.reshape(-1, N_samples, 3)
        normal_map = torch.sum(weights.unsqueeze(-1) * normals, 1)

        # transform to local coordinate system
        # rot = pose[0, :3, :3].permute(1, 0).contiguous()
        rot = c2w[:3, :3].permute(1, 0).contiguous()
        normal_map = rot @ normal_map.permute(1, 0)
        normal_map = normal_map.permute(1, 0).contiguous()

        output['normal_map'] = normal_map

        return output

    def volume_rendering(self, z_vals, sdf):
        density_flat = self.density(sdf)
        density = density_flat.reshape(-1, z_vals.shape[1])  # (batch_size * num_pixels) x N_samples

        dists = z_vals[:, 1:] - z_vals[:, :-1]
        dists = torch.cat([dists, torch.tensor([1e10]).cuda().unsqueeze(0).repeat(dists.shape[0], 1)], -1)

        # LOG SPACE
        free_energy = dists * density
        shifted_free_energy = torch.cat([torch.zeros(dists.shape[0], 1).cuda(), free_energy[:, :-1]],
                                        dim=-1)  # shift one step
        alpha = 1 - torch.exp(-free_energy)  # probability of it is not empty here
        transmittance = torch.exp(
            -torch.cumsum(shifted_free_energy, dim=-1))  # probability of everything is empty up to now
        weights = alpha * transmittance  # probability of the ray hits something here

        return weights

def remap_name(name):
    name = name.replace('.', '-')  # dot is not allowed by pytorch
    if name[-1] == '/':
        name = name[:-1]
    idx = name.rfind('/')
    for i in range(2):
        if idx >= 0:
            idx = name[:idx].rfind('/')
    return name[idx + 1:]

class NerfNetWithAutoExpo(nn.Module):
    def __init__(self, args, optim_autoexpo=False, img_names=None):
        super().__init__()
        self.nerf_net = GridNerfNet(args)

        self.test_env = args.test_env

        self.optim_autoexpo = optim_autoexpo
        if self.optim_autoexpo:
            assert (img_names is not None)
            logger.info('Optimizing autoexposure!')

            self.img_names = [remap_name(x) for x in img_names]
            logger.info('\n'.join(self.img_names))
            self.autoexpo_params = nn.ParameterDict(
                OrderedDict([(x, nn.Parameter(torch.Tensor([0.5, 0.]))) for x in self.img_names]))

        assert (img_names is not None)
        logger.info('Optimizing envmap!')

        self.img_names = [remap_name(x) for x in img_names]
        logger.info('\n'.join(self.img_names))
        self.env_params = nn.ParameterDict(OrderedDict(
            [(x, nn.Parameter(torch.tensor([
                [2.9861e+00, 3.4646e+00, 3.9559e+00],
                [1.0013e-01, -6.7589e-02, -3.1161e-01],
                [-8.2520e-01, -5.2738e-01, -9.7385e-02],
                [2.2311e-03, 4.3553e-03, 4.9501e-03],
                [-6.4355e-03, 9.7476e-03, -2.3863e-02],
                [1.1078e-01, -6.0607e-02, -1.9541e-01],
                [7.9123e-01, 7.6916e-01, 5.6288e-01],
                [6.5793e-02, 4.3270e-02, -1.7002e-01],
                [-7.2674e-02, 4.5177e-02, 2.2858e-01]
            ], dtype=torch.float32))) for x in self.img_names]))  # todo: limit to max 1

        self.register_buffer('defaultenv', torch.tensor([
                [2.9861e+00, 3.4646e+00, 3.9559e+00],
                [1.0013e-01, -6.7589e-02, -3.1161e-01],
                [-8.2520e-01, -5.2738e-01, -9.7385e-02],
                [2.2311e-03, 4.3553e-03, 4.9501e-03],
                [-6.4355e-03, 9.7476e-03, -2.3863e-02],
                [ 1.1078e-01, -6.0607e-02, -1.9541e-01],
                [7.9123e-01, 7.6916e-01, 5.6288e-01],
                [ 6.5793e-02,  4.3270e-02, -1.7002e-01],
                [-7.2674e-02, 4.5177e-02, 2.2858e-01]
        ], dtype=torch.float32))

    def forward(self, ray_o, ray_d, fg_z_max, fg_z_vals, bg_z_vals, iteration, img_name=None, rot_angle=None, save_memory4validation=False, c2w=None, intrinsic=None, ray_matrix=None):
        '''
        :param ray_o, ray_d: [..., 3]
        :param fg_z_max: [...,]
        :param fg_z_vals, bg_z_vals: [..., N_samples]
        :return
        '''
        if img_name is not None:
            img_name = remap_name(img_name)
        env = None
        if self.test_env is not None:
            if not os.path.isdir(self.test_env):
                if 'test_env_val' not in dir(self):
                    env_data = np.loadtxt(self.test_env)
                    self.test_env_val = torch.tensor(env_data, dtype=torch.float32).to(ray_o.device)
                env = self.test_env_val
                logger.warning('using env ' + self.test_env)
            else:
                if 'test_env_val' not in dir(self):
                    self.test_env_val = dict()
                    for env_fn in sorted(glob.glob(os.path.join(self.test_env, '*'))):
                        env_data = np.loadtxt(env_fn)
                        env_name = os.path.splitext(os.path.basename(env_fn))[0]
                        self.test_env_val[env_name] = torch.tensor(env_data, dtype=torch.float32).to(ray_o.device)
                # env_name = img_name.split('/')[-1][:-4]
                env_name = img_name.split('_IMG')[0]
                env = self.test_env_val[env_name]
                logger.warning('using env ' + env_name)
        elif img_name in self.env_params:
            env = self.env_params[img_name]
        else:
            logger.warning('no envmap found for ' + str(img_name))
            env = self.defaultenv

        if rot_angle is not None:
            old_shape = env.shape
            env = rotate_env(env, rot_angle)
            if env.shape != old_shape:
                print(env.shape, old_shape)
            env = env.reshape(old_shape)


        ret = self.nerf_net(ray_o, ray_d, fg_z_max, fg_z_vals, bg_z_vals, env, iteration, c2w, intrinsic, ray_matrix)
        if self.optim_autoexpo and (img_name in self.autoexpo_params):
            autoexpo = self.autoexpo_params[img_name]
            scale = torch.abs(autoexpo[0]) + 0.5  # make sure scale is always positive
            shift = autoexpo[1]
            ret['autoexpo'] = (scale, shift)

        return ret

