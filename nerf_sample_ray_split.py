import numpy as np
from collections import OrderedDict
import torch
import cv2
import imageio
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
########################################################################################################################
# ray batch sampling
########################################################################################################################
def get_rays_single_image(H, W, intrinsics, c2w):
    '''
    :param H: image height
    :param W: image width
    :param intrinsics: 4 by 4 intrinsic matrix
    :param c2w: 4 by 4 camera to world extrinsic matrix
    :return:
    '''
    u, v = np.meshgrid(np.arange(W), np.arange(H))

    # noise_u = np.random.rand(H*W).astype(np.float32)-0.5
    # noise_v = np.random.rand(H*W).astype(np.float32)-0.5
    # u = u.reshape(-1).astype(dtype=np.float32) + 0.5 + noise_u    # add half pixel
    # v = v.reshape(-1).astype(dtype=np.float32) + 0.5 + noise_v
    u = u.reshape(-1).astype(dtype=np.float32) + 0.5    # add half pixel
    v = v.reshape(-1).astype(dtype=np.float32) + 0.5
    pixels = np.stack((u, v, np.ones_like(u)), axis=0)  # (3, H*W)

    ray_matrix = np.dot(c2w[:3, :3], np.linalg.inv(intrinsics[:3, :3]))
    # rays_d = np.dot(np.linalg.inv(intrinsics[:3, :3]), pixels)
    # rays_d = np.dot(c2w[:3, :3], rays_d)  # (3, H*W)
    rays_d = np.dot(ray_matrix, pixels)  # (3, H*W)
    rays_d = rays_d.transpose((1, 0))  # (H*W, 3)

    rays_o = c2w[:3, 3].reshape((1, 3))
    rays_o = np.tile(rays_o, (rays_d.shape[0], 1))  # (H*W, 3)

    depth = np.linalg.inv(c2w)[2, 3]
    depth = depth * np.ones((rays_o.shape[0],), dtype=np.float32)  # (H*W,)

    return rays_o, rays_d, depth, ray_matrix


class RaySamplerSingleImage(object):
    def __init__(self, H, W, intrinsics, c2w,
                       img_path=None,
                       resolution_level=1,
                       mask_path=None,
                       min_depth_path=None,
                       max_depth=None,
                       use_ray_jitter=True):
        super().__init__()
        self.W_orig = W
        self.H_orig = H
        self.intrinsics_orig = intrinsics
        self.c2w_mat = c2w

        self.img_path = img_path
        self.mask_path = mask_path
        self.min_depth_path = min_depth_path
        self.max_depth = max_depth

        self.resolution_level = -1
        self.set_resolution_level(resolution_level)

        self.use_ray_jitter = use_ray_jitter

    def set_resolution_level(self, resolution_level):
        if resolution_level != self.resolution_level:
            self.resolution_level = resolution_level
            self.W = int(self.W_orig // resolution_level)
            self.H = int(self.H_orig // resolution_level)
            self.intrinsics = np.copy(self.intrinsics_orig)
            self.intrinsics[:2, :3] /= resolution_level
            # only load image at this time
            if self.img_path is not None:
                self.img = imageio.imread(self.img_path).astype(np.float32) / 255.
                self.img = cv2.resize(self.img, (self.W, self.H), interpolation=cv2.INTER_AREA)
                self.img = self.img.reshape((-1, 3))
            else:
                self.img = None

            if self.mask_path is not None:
                self.mask = imageio.imread(self.mask_path).astype(np.float32) / 255.
                self.mask = cv2.resize(self.mask, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
                if len(self.mask.shape) == 3:  # if RGB mask, take R
                    print('mask shape', self.mask.shape, 'taking first channel only')
                    self.mask = self.mask[..., 0]
                self.mask = self.mask.reshape((-1,))
            else:
                self.mask = None

            if self.min_depth_path is not None:
                self.min_depth = imageio.imread(self.min_depth_path).astype(np.float32) / 255. * self.max_depth + 1e-4
                self.min_depth = cv2.resize(self.min_depth, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
                self.min_depth = self.min_depth.reshape((-1))
            else:
                self.min_depth = None

            self.rays_o, self.rays_d, self.depth, self.ray_matrix = get_rays_single_image(self.H, self.W,
                                                                                          self.intrinsics, self.c2w_mat)

    def get_img(self):
        if self.img is not None:
            return self.img.reshape((self.H, self.W, 3))
        else:
            # print('bad get_img', self.img_path)
            return None

    def get_all(self, with_pose_intrinsic=False):
        if self.min_depth is not None:
            min_depth = self.min_depth
        else:
            min_depth = 1e-4 * np.ones_like(self.rays_d[..., 0])

        if with_pose_intrinsic == False:
            ret = OrderedDict([
                ('ray_o', self.rays_o),
                ('ray_d', self.rays_d),
                ('depth', self.depth),
                ('rgb', self.img),
                ('mask', self.mask),
                ('min_depth', min_depth),
            ])
        else:
            ret = OrderedDict([
                ('ray_o', self.rays_o),
                ('ray_d', self.rays_d),
                ('depth', self.depth),
                ('rgb', self.img),
                ('mask', self.mask),
                ('min_depth', min_depth),

                ('c2w', self.c2w_mat),
                ('intrinsic', self.intrinsics),
            ])
        # return torch tensors
        for k in ret:
            if ret[k] is not None:
                ret[k] = torch.from_numpy(ret[k])
        return ret

    def random_sample(self, N_rand, center_crop=False, with_pose_intrinsic=False):
        '''
        :param N_rand: number of rays to be casted
        :return:
        '''
        if center_crop:
            half_H = self.H // 2
            half_W = self.W // 2
            quad_H = half_H // 2
            quad_W = half_W // 2

            # pixel coordinates
            u, v = np.meshgrid(np.arange(half_W-quad_W, half_W+quad_W),
                               np.arange(half_H-quad_H, half_H+quad_H))
            u = u.reshape(-1)
            v = v.reshape(-1)

            select_inds = np.random.choice(u.shape[0], size=(N_rand,), replace=False)

            # Convert back to original image
            select_inds = v[select_inds] * self.W + u[select_inds]
        else:
            # Random from one image
            select_inds = np.random.choice(self.H*self.W, size=(N_rand,), replace=False)

        rays_o = self.rays_o[select_inds, :]    # [N_rand, 3]
        rays_d = self.rays_d[select_inds, :]    # [N_rand, 3]
        depth = self.depth[select_inds]         # [N_rand, ]
        ray_matrix = self.ray_matrix

        noise = np.random.rand(2, len(select_inds)).astype(np.float32)-0.5  # [2, N_rand]
        noise = np.stack((noise[0], noise[1], np.zeros(len(select_inds), dtype=np.float32)), axis=0)  # [3, N_rand]
        noise = np.dot(ray_matrix, noise)
        noise = noise.transpose((1, 0))  # [N_rand, 3]
        assert(noise.shape == rays_d.shape)

        if self.use_ray_jitter:
            rays_d = rays_d + noise

        if self.img is not None:
            rgb = self.img[select_inds, :]          # [N_rand, 3]
        else:
            rgb = None

        if self.mask is not None:
            mask = self.mask[select_inds]
        else:
            mask = None

        if self.min_depth is not None:
            min_depth = self.min_depth[select_inds]
        else:
            min_depth = 1e-4 * np.ones_like(rays_d[..., 0])

        if with_pose_intrinsic == False:
            ret = OrderedDict([
                ('ray_o', rays_o),
                ('ray_d', rays_d),
                ('depth', depth),
                ('rgb', rgb),
                ('mask', mask),
                ('min_depth', min_depth),
                ('img_name', self.img_path)
            ])
        else:
            ret = OrderedDict([
                ('ray_o', rays_o),
                ('ray_d', rays_d),
                ('depth', depth),
                ('rgb', rgb),
                ('mask', mask),
                ('min_depth', min_depth),
                ('img_name', self.img_path),

                ('c2w', self.c2w_mat),
                ('intrinsic', self.intrinsics),
                ('ray_matrix', ray_matrix),
            ])
        # return torch tensors
        for k in ret:
            if isinstance(ret[k], np.ndarray):
                ret[k] = torch.from_numpy(ret[k])

        return ret
