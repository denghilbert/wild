import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
# import mcubes
import marching_cubes as mcubes

import logging
from tqdm import tqdm, trange
from ddp_train_nerf import config_parser, setup_logger, setup, cleanup, create_nerf

logger = logging.getLogger(__package__)

import pdb, sys
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

def ddp_mesh_nerf(rank, args):
    ###### set up multi-processing
    assert(args.world_size==1)
    setup(rank, args.world_size, args.master_port)
    ###### set up logger
    logger = logging.getLogger(__package__)
    setup_logger()

    ###### decide chunk size according to gpu memory
    if torch.cuda.get_device_properties(rank).total_memory / 1e9 > 14:
        logger.info('setting batch size according to 24G gpu')
        args.N_rand = 1024
        args.chunk_size = 8192
    else:
        logger.info('setting batch size according to 12G gpu')
        args.N_rand = 512
        args.chunk_size = 4096

    ###### create network and wrap in ddp; each process should do this
    start, models = create_nerf(rank, args)

    # center on lk
    # ax = np.linspace(-1, 1, num=256, endpoint=True, dtype=np.float16)/5
    # X, Y, Z = np.meshgrid(ax, ax+0.1, ax-0.3)

    # ax = np.linspace(-0.4, 0.4, num=256, endpoint=True, dtype=np.float16)
    # ay = np.linspace(-0.3, 0.3, num=256, endpoint=True, dtype=np.float16)
    # az = np.linspace(-0.6, -0.3, num=256, endpoint=True, dtype=np.float16)
    # max, min

    # lwp use depth filter
    ax = np.linspace(-0.5520, 0.1663, num=256, endpoint=True, dtype=np.float16)
    ay = np.linspace(-0.1259, 0.1416, num=256, endpoint=True, dtype=np.float16)
    az = np.linspace(-0.5084, -0.0390, num=256, endpoint=True, dtype=np.float16)


    # ax = np.linspace(-0.5,  0.3633, num=256, endpoint=True, dtype=np.float16)
    # ay = np.linspace(-0.5,  0.3633, num=256, endpoint=True, dtype=np.float16)
    # az = np.linspace(-0.5,  0.2486, num=256, endpoint=True, dtype=np.float16)
    X, Y, Z = np.meshgrid(ax, ay, az)
    # ForkedPdb().set_trace()
    # flip yz
    pts = np.stack((X, Y, Z), -1)
    pts = pts.reshape((-1, 3))

    pts = torch.tensor(pts).float().to(rank)

    u = models['net_1']
    nerf_net = u.module.nerf_net
    fg_net = nerf_net.fg_net

    allres = []
    allcolor = []
    with torch.no_grad():
        posemb = nerf_net.fg_embedder_position
        vdemb = nerf_net.fg_embedder_viewdir
        # direction = torch.tensor([0, 0, -1], dtype=torch.float32).to(rank)
        for bid in trange((pts.shape[0]+args.chunk_size-1)//args.chunk_size):
            bstart = bid * args.chunk_size
            bend = bstart + args.chunk_size
            cpts = pts[bstart:bend].float()
            cem = (cpts[..., 0:1]*0).expand((cpts.shape[0], 9))
            cvd = cpts*0#+direction
            inp = torch.cat((posemb(cpts, start), cem, vdemb(cvd, start)), -1)

            out = fg_net(inp)

            res = out['sigma'].detach().cpu().half().numpy()
            allres.append(res)
            color = out['rgb'].detach().cpu().half().numpy()
            allcolor.append(color)

    allres = np.concatenate(allres, 0)
    allres = allres.reshape(X.shape)

    allcolor = np.concatenate(allcolor, 0)
    allcolor = allcolor.reshape(list(X.shape)+[3,])

    print(allres.min(), allres.max(), allres.mean(), np.median(allres), allres.shape)

    logger.info('Doing MC')
    # vtx, tri = mcubes.marching_cubes(allres.astype(np.float32), 100)
    # tri: gives ref to vtx, 3 refs from a triangle plane
    # vtx: connection between two points
    vtx, tri = mcubes.marching_cubes_color(allres.astype(np.float32), allcolor.astype(np.float32), 200)
    vtx1, tri1 = mcubes.marching_cubes(allres.astype(np.float32), 200)
    x_min = -0.5520
    x_max = 0.1663
    y_min = -0.1259
    y_max = 0.1416
    z_min = -0.5084
    z_max = -0.0390
    res = 256 - 1
    delta_x, delta_y, delta_z = (x_max - x_min) / res, (y_max - y_min) / res, (z_max - z_min) / res
    delta = np.array([[delta_x, delta_y, delta_z, delta_x, delta_y, delta_z]]).repeat(vtx.shape[0], axis=0).reshape(-1, 6)
    min_xyz = np.array([[x_min, y_min, z_min, x_min, y_min, z_min]]).repeat(vtx.shape[0], axis=0).reshape(-1, 6)
    transform_xyz = min_xyz + delta * vtx
    ForkedPdb().set_trace()
    logger.info('Exporting mesh')
    # mcubes.export_mesh(vtx, tri, "mesh5.dae", "Mesh")
    mcubes.export_obj(vtx, tri, "/home/youmingdeng/lwp.obj")


def mesh():
    parser = config_parser()
    args = parser.parse_args()
    logger.info(parser.format_values())

    if args.world_size == -1:
        args.world_size = torch.cuda.device_count()
        logger.info('Using # gpus: {}'.format(args.world_size))
    torch.multiprocessing.spawn(ddp_mesh_nerf,
                                args=(args,),
                                nprocs=args.world_size,
                                join=True)


if __name__ == '__main__':
    setup_logger()
    mesh()

