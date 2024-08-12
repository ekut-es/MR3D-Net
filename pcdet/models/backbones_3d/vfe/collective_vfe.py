import torch

from .vfe_template import VFETemplate


class CollectiveMeanVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, **kwargs):
        super().__init__(model_cfg=model_cfg)
        self.num_point_features = num_point_features

    def get_output_feature_dim(self):
        return self.num_point_features

    def forward(self, batch_dict, **kwargs):
        """
        Args:
            batch_dict:
                voxels: (num_voxels, max_points_per_voxel, C)
                voxel_num_points: optional (num_voxels)
            **kwargs:

        Returns:
            vfe_features: (num_voxels, C)
        """


        if batch_dict['voxel_method'][0] == 'all':
            voxel_features, voxel_num_points = batch_dict['coop_voxels'], batch_dict['coop_voxel_num_points']
            points_mean = voxel_features[:, :, :].sum(dim=1, keepdim=False)
            normalizer = torch.clamp_min(voxel_num_points.view(-1, 1), min=1.0).type_as(voxel_features)
            points_mean = points_mean / normalizer
            batch_dict['coop_voxel_features'] = points_mean.contiguous()
            del batch_dict['resolution']
            del batch_dict['point_cloud_range']
        elif batch_dict['voxel_method'][0] == 'multi_resolution':
            resolution_names = ['high', 'mid', 'low']
            for r_name in resolution_names:
                voxel_features, voxel_num_points = batch_dict['coop_voxels_'+r_name], batch_dict['coop_voxel_num_points_'+r_name]
                points_mean = voxel_features[:, :, :].sum(dim=1, keepdim=False)
                normalizer = torch.clamp_min(voxel_num_points.view(-1, 1), min=1.0).type_as(voxel_features)
                points_mean = points_mean / normalizer
                batch_dict['coop_voxel_features_'+r_name] = points_mean.contiguous()
                del batch_dict['resolution_'+r_name]
            del batch_dict['point_cloud_range']
        return batch_dict


class CollectiveCenterVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, **kwargs):
        super().__init__(model_cfg=model_cfg)
        self.num_point_features = 3 # only use voxel centers as features

    def get_output_feature_dim(self):
        return self.num_point_features

    def forward(self, batch_dict, **kwargs):
        """
        Args:
            batch_dict:
                voxels: (num_voxels, max_points_per_voxel, C)
                voxel_num_points: optional (num_voxels)
            **kwargs:

        Returns:
            vfe_features: (num_voxels, C)
        """
        if batch_dict['voxel_method'][0] == 'all':
            voxel_coords = batch_dict['coop_voxel_coords']
            resolution = batch_dict['resolution']
            point_cloud_range =  batch_dict['point_cloud_range']
            voxel_centers = self.get_voxel_centers(voxel_coords, resolution, point_cloud_range)
            batch_dict['coop_voxel_features'] = voxel_centers
            del batch_dict['resolution']
            del batch_dict['point_cloud_range']
        elif batch_dict['voxel_method'][0] == 'multi_resolution':
            resolution_names = ['high', 'mid', 'low']
            for r_name in resolution_names:
                voxel_coords = batch_dict['coop_voxel_coords_'+r_name]
                resolution = batch_dict['resolution_'+r_name]
                point_cloud_range =  batch_dict['point_cloud_range']
                voxel_centers = self.get_voxel_centers(voxel_coords, resolution, point_cloud_range)
                batch_dict['coop_voxel_features_'+r_name] = voxel_centers
                del batch_dict['resolution_'+r_name]
            del batch_dict['point_cloud_range']
        return batch_dict
    
    @staticmethod
    def get_voxel_centers(voxel_coords, resolution, point_cloud_range):
        zyx_resolution = torch.flip(resolution[0], [0])
        zyx_pc_range =  torch.flip(point_cloud_range[0, :3], [0])
        voxel_centers = torch.mul(voxel_coords[:, 1:], zyx_resolution)
        voxel_centers = torch.add(voxel_centers, 0.5 *  zyx_resolution)
        voxel_centers = torch.add(voxel_centers, zyx_pc_range)
        voxel_centers = torch.fliplr(voxel_centers)
        return voxel_centers.contiguous()