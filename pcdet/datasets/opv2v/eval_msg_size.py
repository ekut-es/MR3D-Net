import os
import glob
import numpy as np
from pyntcloud import PyntCloud
from pcdet.datasets.processor.data_processor import VoxelGeneratorWrapper
import tqdm

root_path = '../../../data/OPV2V/'
split = 'test'

# COLLECTIVE_DATA_PROCESSOR:
#     use_coop_info: True
#     early_fusion: False # use all point clouds as ego
#     transform_points_to_voxels:
#       VOXEL_SIZE: [ 0.05, 0.05, 0.1 ]
#       MAX_POINTS_PER_VOXEL: 5
#       MAX_NUMBER_OF_VOXELS: {
#         'train': 150000,
#         'test': 150000
#       }
#     voxel_method: 'multi_resolution' # use 'all', 'individual' or 'multi_resolution' to voxelize the coop points together, induvidually or individually with multiple resolutions
#     resolutions: [[ 0.05, 0.05, 0.1 ], [ 0.1, 0.1, 0.2 ], [ 0.2, 0.2, 0.4 ]]

pc_range = [-140, -40, -3, 140, 40, 1]

voxel_size_low = [ 0.2, 0.2, 0.4 ]
voxel_generator_low = VoxelGeneratorWrapper(
                    vsize_xyz=voxel_size_low,
                    coors_range_xyz=pc_range,
                    num_point_features=4,
                    max_num_points_per_voxel=5,
                    max_num_voxels=150000,
                )
coors_size_low = np.ceil(np.log2(abs(pc_range[0]-pc_range[3])/voxel_size_low[0])) + np.ceil(np.log2(abs(pc_range[1]-pc_range[4])/voxel_size_low[1])) + np.ceil(np.log2(abs(pc_range[2]-pc_range[5])/voxel_size_low[2]))

voxel_size_mid = [ 0.1, 0.1, 0.2 ]
voxel_generator_mid = VoxelGeneratorWrapper(
                    vsize_xyz=voxel_size_mid,
                    coors_range_xyz=pc_range,
                    num_point_features=4,
                    max_num_points_per_voxel=5,
                    max_num_voxels=150000,
                )
coors_size_mid = np.ceil(np.log2(abs(pc_range[0]-pc_range[3])/voxel_size_mid[0])) + np.ceil(np.log2(abs(pc_range[1]-pc_range[4])/voxel_size_mid[1])) + np.ceil(np.log2(abs(pc_range[2]-pc_range[5])/voxel_size_mid[2]))

voxel_size_high = [ 0.05, 0.05, 0.1 ]
voxel_generator_high = VoxelGeneratorWrapper(
                    vsize_xyz=voxel_size_high,
                    coors_range_xyz=pc_range,
                    num_point_features=4,
                    max_num_points_per_voxel=5,
                    max_num_voxels=150000,
                )
coors_size_high = np.ceil(np.log2(abs(pc_range[0]-pc_range[3])/voxel_size_high[0])) + np.ceil(np.log2(abs(pc_range[1]-pc_range[4])/voxel_size_high[1])) + np.ceil(np.log2(abs(pc_range[2]-pc_range[5])/voxel_size_high[2]))

raw_sum = 0 # sum of all raw point clouds in bytes
all_feature_sum_high = 0 # sum of all grid features in bytes
all_feature_sum_mid = 0 # sum of all grid features in bytes
all_feature_sum_low = 0 # sum of all grid features in bytes
low_res_sum = 0 # sum of all low res voxel grids in bytes
mid_res_sum = 0 # sum of all low res voxel grids in bytes
high_res_sum = 0 # sum of all low res voxel grids in bytes
points_sum = 0
count = 0
scenarios = sorted([os.path.basename(x) for x in glob.glob(str(root_path + (split + '/*')), recursive=True)])
for scenario in tqdm.tqdm(scenarios):
    vehicles = sorted([os.path.basename(x) for x in glob.glob(str(root_path + (split + '/' + scenario + '/[!data_protocol]*')), recursive=True)])
    for vehicle in vehicles:
        frames = sorted(glob.glob(str(root_path + (split + '/' + scenario + '/' + vehicle + '/*.pcd')), recursive=True))
        for frame in frames:
            pcd = PyntCloud.from_file(str(frame))
            points = pcd.points.to_numpy()[:, :4]

            points_sum += len(points)

            raw_sum += len(points)*16 # number of points times 4 coords times 4 byte/coord

            coop_voxels, coop_coordinates, coop_num_points = voxel_generator_low.generate(points)
            low_res_sum += (len(coop_voxels)*coors_size_low)/8 # number of voxels times bits/voxel divided by 8 to get bytes
            all_feature_sum_low += len(coop_voxels)*16 # number of voxels times 4 features times 4 byte/feature

            coop_voxels, coop_coordinates, coop_num_points = voxel_generator_mid.generate(points)
            mid_res_sum += (len(coop_voxels)*coors_size_mid)/8 # number of voxels times bits/voxel divided by 8 to get bytes
            all_feature_sum_mid += len(coop_voxels)*16 # number of voxels times 4 features times 4 byte/feature

            coop_voxels, coop_coordinates, coop_num_points = voxel_generator_high.generate(points)
            high_res_sum += (len(coop_voxels)*coors_size_high)/8 # number of voxels times bits/voxel divided by 8 to get bytes
            all_feature_sum_high += len(coop_voxels)*16 # number of voxels times 4 features times 4 byte/feature
            count += 1

print('Average Number of Points: ', (points_sum/count))
print('Average Raw Point Cloud Size: ', (raw_sum/count)/1000, ' kB' )
print('Average Low Resolution Voxel Grid Size: ', (low_res_sum/count)/1000, ' kB' )
print('Average Mid Resolution Voxel Grid Size: ', (mid_res_sum/count)/1000, ' kB' )
print('Average High Resolution Voxel Grid Size: ', (high_res_sum/count)/1000, ' kB' )

print('---------------------- Bandwidth@10Hz ---------------------------')
print('Average Number of Points: ', (points_sum/count))
print('Average Raw Point Cloud Bandwidth: ', ((raw_sum/count)*80)/1e6, ' Mb/s' )
print('Average Low Resolution Voxel Grid Bandwidth: ', ((low_res_sum/count)*80)/1e6, ' Mb/s' )
print('Average Mid Resolution Voxel Grid Bandwidth: ', ((mid_res_sum/count)*80)/1e6, ' Mb/s' )
print('Average High Resolution Voxel Grid Bandwidth: ', ((high_res_sum/count)*80)/1e6, ' Mb/s' )

print('Average All Feature High Voxel Grid Bandwidth: ', ((all_feature_sum_high/count)*80)/1e6, ' Mb/s' )
print('Average All Feature Mid Voxel Grid Bandwidth: ', ((all_feature_sum_mid/count)*80)/1e6, ' Mb/s' )
print('Average All Feature Low Voxel Grid Bandwidth: ', ((all_feature_sum_low/count)*80)/1e6, ' Mb/s' )