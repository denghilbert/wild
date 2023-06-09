from hashencoder.hashgrid import _hash_encode, HashEncoder

geometric_init = True
bias = 1.0
skip_in = ()
weight_norm = True
multires = 0
sphere_scale = 1.0
inside_outside = False
base_size = 16
end_size = 2048
logmap = 19
num_levels = 16
level_dim = 2
divide_factor = 1.5  # used to normalize the points range for multi-res grid
use_grid_feature = True

print(f"using hash encoder with {num_levels} levels, each level with feature dim {level_dim}")
print(f"resolution:{base_size} -> {end_size} with hash map size {logmap}")
encoding = HashEncoder(input_dim=3, num_levels=num_levels, level_dim=level_dim,
                            per_level_scale=2, base_resolution=base_size,
                            log2_hashmap_size=logmap, desired_resolution=end_size).cuda()

import torch

tensor = torch.tensor([[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]).cuda()
print(tensor)
print(encoding(tensor))