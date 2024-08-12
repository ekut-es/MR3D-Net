from functools import partial

import torch.nn as nn
import torch
from ...utils.spconv_utils import spconv
from pcdet.utils.collective_utils import scatter_duplicates, sparse_concat
import numpy as np

def conv_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,conv_type='subm', norm_fn=None):
    if conv_type == 'subm':
        conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
    elif conv_type == 'spconv':
        conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   bias=False, indice_key=indice_key)
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
    else:
        raise NotImplementedError

    m = spconv.SparseSequential(
        conv,
        norm_fn(out_channels),
        nn.ReLU(),
    )

    return m

def build_resolution_stream(input_channels, conv_channels, strides, subm_per_block=2, key='local'):
    block = conv_block
    norm_fn = partial(nn.BatchNorm1d, eps=6.2e-5, momentum=0.01)
    conv_input = spconv.SparseSequential(
        spconv.SubMConv3d(input_channels, conv_channels[0], 3, padding=1, bias=False, indice_key='subm1'+key),
        norm_fn(conv_channels[0]),
        nn.ReLU(),
    )
    conv1 = spconv.SparseSequential(block(conv_channels[0], conv_channels[0], 3, norm_fn=norm_fn, stride=strides[0], padding=1, indice_key='subm1'+key))
    conv2 = spconv.SparseSequential(block(conv_channels[0], conv_channels[1], 3, norm_fn=norm_fn, stride=strides[1], padding=1, indice_key='spconv2'+key, conv_type='spconv'))
    conv3 = spconv.SparseSequential(block(conv_channels[1], conv_channels[2], 3, norm_fn=norm_fn, stride=strides[2], padding=1, indice_key='spconv3'+key, conv_type='spconv'))
    conv4 = spconv.SparseSequential(block(conv_channels[2], conv_channels[2], 3, norm_fn=norm_fn, stride=strides[3], padding=(0, 1, 1), indice_key='spconv4'+key, conv_type='spconv'))

    for _ in range(subm_per_block):
        conv2.add(block(conv_channels[1], conv_channels[1], 3, norm_fn=norm_fn, padding=1, indice_key='subm2'+key))
        conv3.add(block(conv_channels[2], conv_channels[2], 3, norm_fn=norm_fn, padding=1, indice_key='subm3'+key))
        conv4.add(block(conv_channels[2], conv_channels[2], 3, norm_fn=norm_fn, padding=1, indice_key='subm4'+key))


    return conv_input, conv1, conv2, conv3, conv4

class MR3DNet(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=6.2e-5, momentum=0.01)
        conv_channels = self.model_cfg.CONV_CHANNELS

        ## local input stream
        self.local_input, self.local_conv1, self.local_conv2, self.local_conv3, self.local_conv4 = build_resolution_stream(input_channels, conv_channels, self.model_cfg.HIGH_STRIDES, self.model_cfg.LOCAL_SUBM_PER_BLOCK,  key='local')

        # local  out conv
        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out_loc = spconv.SparseSequential(
            #[200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(conv_channels[2], conv_channels[3], (3, 1, 1), stride=(2, 1, 1), padding=last_pad, bias=False, indice_key='spconv_down2'),
            norm_fn(conv_channels[3]),
            nn.ReLU(),
        )

        ## collective input streams
        collective_input_channels = self.model_cfg.COLLECTIVE_INPUT_CHANNELS
        self.conv_input_low, self.conv1_low, self.conv2_low, self.conv3_low, self.conv4_low = build_resolution_stream(collective_input_channels, conv_channels, self.model_cfg.LOW_STRIDES, self.model_cfg.COLLECTIVE_SUBM_PER_BLOCK, key='low')
        self.conv_input_mid, self.conv1_mid, self.conv2_mid, self.conv3_mid, self.conv4_mid = build_resolution_stream(collective_input_channels, conv_channels, self.model_cfg.MID_STRIDES, self.model_cfg.COLLECTIVE_SUBM_PER_BLOCK, key='mid')
        self.conv_input_high, self.conv1_high, self.conv2_high, self.conv3_high, self.conv4_high = build_resolution_stream(collective_input_channels, conv_channels, self.model_cfg.HIGH_STRIDES, self.model_cfg.COLLECTIVE_SUBM_PER_BLOCK, key='high')


        # collective out conv
        self.conv_out_coll = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(conv_channels[2], conv_channels[3], (3, 1, 1), stride=(2, 1, 1), padding=last_pad, bias=False, indice_key='spconv_down2'),
            norm_fn(conv_channels[3]),
            nn.ReLU(),
        )
        
        self.num_point_features = 256
        self.backbone_channels = {
            'x_conv1': 16,
            'x_conv2': 32,
            'x_conv3': 64,
            'x_conv4': 64
        }



    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """

        input_sp_tensor_loc = spconv.SparseConvTensor(
            features=batch_dict['voxel_features'],
            indices=batch_dict['voxel_coords'].int(),
            spatial_shape=self.model_cfg.HIGH_GRID_SIZE,
            batch_size=batch_dict['batch_size']
        )

        input_sp_tensor_high = spconv.SparseConvTensor(
            features=batch_dict['coop_voxel_features_high'],
            indices=batch_dict['coop_voxel_coords_high'].int(),
            spatial_shape=self.model_cfg.HIGH_GRID_SIZE,
            batch_size=batch_dict['batch_size']
        )

        input_sp_tensor_mid = spconv.SparseConvTensor(
            features=batch_dict['coop_voxel_features_mid'],
            indices=batch_dict['coop_voxel_coords_mid'].int(),
            spatial_shape=np.array([21, 800, 2800]),
            batch_size=batch_dict['batch_size']
        )

        input_sp_tensor_low = spconv.SparseConvTensor(
            features=batch_dict['coop_voxel_features_low'],
            indices=batch_dict['coop_voxel_coords_low'].int(),
            spatial_shape=np.array([11, 400, 1400]),
            batch_size=batch_dict['batch_size']
        )

        scatter_func = 'max'
        
        #Input (Stride = 1)
        x_low = self.conv_input_low(input_sp_tensor_low)
        x_mid = self.conv_input_mid(input_sp_tensor_mid)
        x_high = self.conv_input_high(input_sp_tensor_high)
        x_loc = self.local_input(input_sp_tensor_loc)

        # conv1
        x_conv1_low = self.conv1_low(x_low)
        x_conv1_mid  = self.conv1_mid(x_mid)
        x_conv1_high = self.conv1_high(x_high)
        x_conv1_loc = self.local_conv1(x_loc)

        # conv 1 scatter
        x_conv1_high_loc = sparse_concat(x_conv1_high, x_conv1_loc, scatter_function=scatter_func)

        # conv2
        x_conv2_low = self.conv2_low(x_conv1_low)
        x_conv2_mid  = self.conv2_mid(x_conv1_mid)
        x_conv2_high = self.conv2_high(x_conv1_high)
        x_conv2_loc = self.local_conv2(x_conv1_high_loc)

        x_conv2_mid_high = sparse_concat(x_conv2_mid, x_conv2_high, scatter_function=scatter_func)
        x_conv2_mid_high_loc = sparse_concat(x_conv2_mid_high, x_conv2_loc, scatter_function=scatter_func)


        # conv3
        x_conv3_low = self.conv3_low(x_conv2_low)
        x_conv3_mid  = self.conv3_mid(x_conv2_mid)
        x_conv3_high = self.conv3_high(x_conv2_mid_high)
        x_conv3_loc = self.local_conv3(x_conv2_mid_high_loc)

        # conv 3 scatter
        x_conv3_low_mid = sparse_concat(x_conv3_low, x_conv3_mid, scatter_function=scatter_func)
        x_conv3_low_mid_high = sparse_concat(x_conv3_low_mid, x_conv3_high, scatter_function=scatter_func)
        x_conv3_low_mid_high_loc = sparse_concat(x_conv3_low_mid_high, x_conv3_loc, scatter_function=scatter_func)

        # conv4
        x_conv4_low = self.conv4_low(x_conv3_low)
        x_conv4_mid  = self.conv4_mid(x_conv3_low_mid)
        x_conv4_high = self.conv4_high(x_conv3_low_mid_high)
        x_conv4_loc = self.local_conv4(x_conv3_low_mid_high_loc)

        # conv4 scatter
        x_conv4_low_mid_high = sparse_concat(x_conv4_low, x_conv4_mid, x_conv4_high, scatter_function=scatter_func)
        x_conv4_low_mid_high_loc = sparse_concat(x_conv4_low_mid_high, x_conv4_loc, scatter_function=scatter_func)

        #  out conv
        out_coll = scatter_duplicates(self.conv_out_coll(x_conv4_low_mid_high_loc))
        out_loc = scatter_duplicates(self.conv_out_loc(x_conv4_loc))


        # feature channel concatenation
        out = out_loc.replace_feature(torch.cat((out_loc.features, out_coll.features), dim=1))


        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1_loc,
                'x_conv2': x_conv2_loc,
                'x_conv3': x_conv3_loc,
                'x_conv4': x_conv4_loc,
            }
        })
        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })

        return batch_dict