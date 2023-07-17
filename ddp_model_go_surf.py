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


from go_model.sdf_grid_model import SDFGridModel
import torch.nn.functional as F


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
        self.nerf_net = SDFGridModel(args)

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

    def forward(self, ray_o, ray_d, fg_z_max, fg_z_vals, bg_z_vals, iteration, img_name=None, rot_angle=None, save_memory4validation=False, c2w=None, intrinsic=None, ray_matrix=None, validation=False):
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

        ret = self.nerf_net(ray_o, ray_d, fg_z_max, fg_z_vals, bg_z_vals, env, iteration, c2w, intrinsic, ray_matrix, validation=validation)
        if self.optim_autoexpo and (img_name in self.autoexpo_params):
            autoexpo = self.autoexpo_params[img_name]
            scale = torch.abs(autoexpo[0]) + 0.5  # make sure scale is always positive
            shift = autoexpo[1]
            ret['autoexpo'] = (scale, shift)

        return ret

