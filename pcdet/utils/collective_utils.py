import torch
import numpy as np
from torchist import ravel_multi_index
from torch_scatter import scatter
from pcdet.utils.spconv_utils import spconv
from typing import Optional

def scatter_duplicates(x: spconv.SparseConvTensor, scatter_function='sum'):
    idx = ravel_multi_index(x.indices, shape=(x.batch_size, *x.spatial_shape))
    _, idx = torch.unique(idx, return_inverse=True)
    scattered_features = scatter(x.features, idx, dim=0, reduce=scatter_function)

    res_shape = [x.batch_size, *x.spatial_shape, x.features.shape[1]]

    x_th = torch.sparse_coo_tensor(x.indices.T, x.features, res_shape, requires_grad=True)

    x_th = x_th.coalesce()
    unique_indices = x_th.indices().T.contiguous().int()

    assert scattered_features.is_contiguous()
    return spconv.SparseConvTensor(scattered_features, unique_indices, x.spatial_shape, x.batch_size)

def sparse_concat(x: spconv.SparseConvTensor, y: spconv.SparseConvTensor, z: Optional[spconv.SparseConvTensor]=None, scatter_function='sum'):
    assert x.spatial_shape == y.spatial_shape
    assert x.features.shape[1] == y.features.shape[1]
    if z is not None:
        assert x.spatial_shape == z.spatial_shape
        assert x.features.shape[1] == z.features.shape[1]
        features = torch.cat([x.features, y.features, z.features], dim=0)
        indices = torch.cat([x.indices, y.indices, z.indices], dim=0)
    else:
        features = torch.cat([x.features, y.features], dim=0)
        indices = torch.cat([x.indices, y.indices], dim=0)
    xyz = spconv.SparseConvTensor(features=features, indices=indices, spatial_shape=x.spatial_shape, batch_size=x.batch_size)
    return scatter_duplicates(xyz, scatter_function=scatter_function)
    

def get_common_voxel_features_diff_dim(high_coords, low_coords, high_features, low_features, scale_xyz=(2, 2, 2), window_size_xyz=(1, 1, 1), spatial_shape_zyx=(41, 1600, 5600)):
    scale_x, scale_y, scale_z = [s * w for s, w in zip(scale_xyz, window_size_xyz)]
    window_size_x, window_size_y, window_size_z = window_size_xyz
    spatial_shape_z, spatial_shape_y, spatial_shape_x = spatial_shape_zyx

    max_num_win_x = int(np.ceil((spatial_shape_x / scale_x)))
    max_num_win_y = int(np.ceil((spatial_shape_y / scale_y)))
    max_num_win_z = int(np.ceil((spatial_shape_z / scale_z)))
    max_num_win_per_sample = max_num_win_x * max_num_win_y * max_num_win_z

    high_coords_x, low_coords_x = high_coords[:, 3], low_coords[:, 3]
    high_coords_y, low_coords_y = high_coords[:, 2], low_coords[:, 2]
    high_coords_z, low_coords_z = high_coords[:, 1], low_coords[:, 1]

    # calculate common voxel coords in the window resolution
    # if windowsize = (1, 1, 1), window coords equal low coords
    high_in_common_coords_x = high_coords_x // scale_x
    high_in_common_coords_y = high_coords_y // scale_y
    high_in_common_coords_z = high_coords_z // scale_z

    low_in_common_coords_x = low_coords_x // window_size_x
    low_in_common_coords_y = low_coords_y // window_size_y
    low_in_common_coords_z = low_coords_z // window_size_z

    # flatten coords to calculate the intersection of the low and high resolution voxel grid
    flat_voxel_inds_high_in_common = high_coords[:, 0] * max_num_win_per_sample + \
                                     high_in_common_coords_x * max_num_win_y * max_num_win_z + \
                                     high_in_common_coords_y * max_num_win_z + \
                                     high_in_common_coords_z

    flat_voxel_inds_low_in_common = low_coords[:, 0] * max_num_win_per_sample + \
                                    low_in_common_coords_x * max_num_win_y * max_num_win_z + \
                                    low_in_common_coords_y * max_num_win_z + \
                                    low_in_common_coords_z

    # calculate the intersection of both voxel grids
    shared_indices_low = torch.where(torch.isin(flat_voxel_inds_low_in_common, flat_voxel_inds_high_in_common))[0]
    if shared_indices_low.shape[0] == 0:
        return
    shared_indices_high = torch.where(torch.isin(flat_voxel_inds_high_in_common, flat_voxel_inds_low_in_common))[0]

    # calculate mapping to output tensor
    sorted_shared_inds = flat_voxel_inds_low_in_common[shared_indices_low].unique()
    shared_low_in_common_index = torch.searchsorted(sorted_shared_inds, flat_voxel_inds_low_in_common[shared_indices_low])
    shared_high_in_common_index = torch.searchsorted(sorted_shared_inds, flat_voxel_inds_high_in_common[shared_indices_high])

    shared_voxel_coords_high = high_coords[shared_indices_high]
    shared_voxel_coords_low = low_coords[shared_indices_low]

    high_coors_shared_x = shared_voxel_coords_high[:, 3]
    high_coors_shared_y = shared_voxel_coords_high[:, 2]
    high_coors_shared_z = shared_voxel_coords_high[:, 1]

    low_coors_shared_x = shared_voxel_coords_low[:, 3]
    low_coors_shared_y = shared_voxel_coords_low[:, 2]
    low_coors_shared_z = shared_voxel_coords_low[:, 1]

    # calculate inner window indices
    high_coords_in_window_x = high_coors_shared_x % scale_x
    high_coords_in_window_y = high_coors_shared_y % scale_y
    high_coords_in_window_z = high_coors_shared_z % scale_z

    low_coords_in_window_x = low_coors_shared_x % window_size_x
    low_coords_in_window_y = low_coors_shared_y % window_size_y
    low_coords_in_window_z = low_coors_shared_z % window_size_z

    high_index_in_window = high_coords_in_window_x * scale_y * scale_z + \
                           high_coords_in_window_y * scale_z + \
                           high_coords_in_window_z

    low_index_in_window = low_coords_in_window_x * window_size_y * window_size_z + \
                          low_coords_in_window_y * window_size_z + \
                          low_coords_in_window_z

    # prepare output tensors
    num_voxels_in_window_high = scale_x * scale_y * scale_z
    num_voxels_in_window_low = window_size_x * window_size_y * window_size_z
    num_windows = shared_low_in_common_index.max() + 1

    common_high_voxel_features = torch.zeros((num_windows, num_voxels_in_window_high, high_features.shape[-1]), device=high_features.device, requires_grad=True)
    common_low_voxel_features = torch.zeros((num_windows, num_voxels_in_window_low, low_features.shape[-1]), device=high_features.device, requires_grad=True)
    # # todo: check if requires grad is needed and remove this
    common_high_voxel_features = common_high_voxel_features + 0
    common_low_voxel_features = common_low_voxel_features + 0
    # write voxel features in output tensors
    common_high_voxel_features[shared_high_in_common_index, high_index_in_window] = high_features[shared_indices_high]
    common_low_voxel_features[shared_low_in_common_index, low_index_in_window] = low_features[shared_indices_low]

    return common_low_voxel_features, common_high_voxel_features, shared_indices_low, shared_indices_high, (shared_low_in_common_index, low_index_in_window), (shared_high_in_common_index, high_index_in_window)