import glob
import os
import numpy as np
import torch
import tqdm
import re
import copy
import pickle
from pyntcloud import PyntCloud
import yaml
import random

from pcdet.datasets import DatasetTemplate
from pcdet.utils import box_utils, calibration_kitti, common_utils
from pcdet.datasets.processor.data_processor import VoxelGeneratorWrapper
#from pcdet.visualization import PCViewer
from pcdet.ops.opv2v_tools import opv2v_transformations as transformations



class OPV2VDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        super().__init__(dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger)
        self.augmentor = None

        self.infos = []
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        self.sample_id_list = []

        self.scenarios = sorted([os.path.basename(x) for x in glob.glob(str(self.root_path / (self.split + '/*/')), recursive=True)])
        self.frames = {}
        for scenario in self.scenarios:
            self.frames[scenario] = {}
            vehicles = sorted([os.path.basename(x) for x in glob.glob(str(self.root_path / (self.split + '/' + scenario + '/[!data_protocol]*/')), recursive=True)])
            for vehicle in vehicles:
                self.frames[scenario][vehicle] = sorted([os.path.basename(x).split('.')[0] for x in glob.glob(str(self.root_path / (self.split + '/' + scenario + '/' + vehicle + '/*.pcd')), recursive=True)])
        for key, value in self.frames.items():
            self.sample_id_list += [(key, x) for x in next(iter(value.values()))]
        self.infos = self.load_info_data(self.mode)
        self.coop_voxel_generator = None
        self.num_voxels_at_resolution = {}

    def load_info_data(self, mode):
        info_path = self.root_path / self.dataset_cfg.INFO_PATH[mode]
        if not info_path.exists():
            return []
        with open(info_path, 'rb') as f:
            infos = pickle.load(f)
        return infos

    def __len__(self):
        return len(self.sample_id_list)

    def get_lidar(self, idx):
        lidar_file = self.root_path / (self.split+'/'+idx+'.pcd')
        assert lidar_file.exists()
        pcd = PyntCloud.from_file(str(lidar_file))
        points = pcd.points.to_numpy()[:, :4]
        return points

    def get_label(self, idx):
        label_file = self.root_path / (self.split+'/'+idx+'.yaml')
        assert label_file.exists()
        with open(str(label_file), 'r') as stream:
            loader = yaml.Loader
            pattern = re.compile(u'''^(?:[-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?|[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)|\\.[0-9_]+(?:[eE][-+][0-9]+)?|[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*|[-+]?\\.(?:inf|Inf|INF)|\\.(?:nan|NaN|NAN))$''', re.X)
            loader.add_implicit_resolver(u'tag:yaml.org,2002:float', pattern, list(u'-+0123456789.')) 
            label = yaml.load(stream, Loader=loader)
            if "yaml_parser" in label:
                label = eval(label["yaml_parser"])(label)

        return label

    def get_calib(self, idx):
        calib = {'P2': np.eye(3, 4),
                 'R0': np.eye(3),
                 'Tr_velo2cam': np.eye(3, 4)}
        return calibration_kitti.Calibration(calib)

    def evaluation(self, det_annos, class_names, **kwargs):
        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = []
        for det_anno in det_annos:
            scenario, vehicle, frame = det_anno['frame_id']
            info = copy.deepcopy(self.infos[scenario][vehicle][frame])
            # collect all boxes from all vehicles and transform them to ego coords
            # this is necesarry since only visible objects are labeled for the ego
            gt_coop_boxes = {}
            for v in self.infos[scenario].keys():
                label = copy.deepcopy(self.infos[scenario][v][frame]['label'])
                label['lidar_pose'] = info['label']['lidar_pose']  # set lidar pose of ego vehicle
                gt_coop_boxes.update(self.get_objects_from_label(label))
            gt_coop_boxes.pop(int(vehicle), None)  # remove box of ego vehicle
            gt_coop_boxes = list(gt_coop_boxes.values())
            num_gt_coop_boxes = len(gt_coop_boxes)
            gt_lidar_boxes = np.squeeze(np.stack(gt_coop_boxes), axis=1) if num_gt_coop_boxes > 0 else np.array([])
            # mask ground truth that is out of range
            gt_mask = box_utils.mask_boxes_outside_range_numpy(
                gt_lidar_boxes, 
                [-140, -40, -3, 140, 40, 1], # use official opv2v eval range (z not provided)
                min_num_corners=1,
                use_center_to_filter=True
            )
            eval_gt_annos.append(gt_lidar_boxes[gt_mask])

        from pcdet.ops.opv2v_tools import opv2v_evaluation as evaluation

        result_stat_template = {3: {'tp': [], 'fp': [], 'gt': [0], 'score': []},                
                                5: {'tp': [], 'fp': [], 'gt': [0], 'score': []},                
                                7: {'tp': [], 'fp': [], 'gt': [0], 'score': []}}
        
        result_stat = evaluation.make_result_dict(result_stat_template)
        for gt_anno, det_anno in zip(eval_gt_annos, eval_det_annos):
            det_boxes = torch.from_numpy(det_anno['boxes_lidar']).float()
            gt_boxes = torch.from_numpy(gt_anno).float()
            scores = torch.from_numpy(det_anno['score']).float()
            evaluation.calculate_false_and_true_positive(det_boxes, scores , gt_boxes, float(0.3), result_stat)
            evaluation.calculate_false_and_true_positive(det_boxes, scores , gt_boxes, float(0.5), result_stat)
            evaluation.calculate_false_and_true_positive(det_boxes, scores , gt_boxes, float(0.7), result_stat)
        aps = evaluation.evaluate(result_stat, False)

        for i, ap in enumerate(aps):
            aps[i] = round(ap*100, 2)

        ap_dict = {'ap_30': aps[0], 'ap_50': aps[1], 'ap_70': aps[2]}
        ap_result_str=''' 
        AP @ IoU 0.3: {0}
        AP @ IoU 0.5: {1}
        AP @ IoU 0.7: {2}'''.format(aps[0], aps[1], aps[2])
        return ap_result_str, ap_dict

    def get_infos(self, has_label=True, count_inside_pts=True):
        infos = {}
        for scenario, vehicles in tqdm.tqdm(self.frames.items(), desc='collecting infos', position=0):
            infos[scenario] = {}
            for vehicle, samples in vehicles.items():
                infos[scenario][vehicle] = {}
                for sample in samples:
                    sample_path = scenario+'/'+vehicle+'/'+sample
                    infos[scenario][vehicle][sample] = self.get_info(sample_path, has_label, count_inside_pts)
        return infos

    def get_info(self, sample_idx, has_label=True, count_inside_pts=True):
        info = {}
        if has_label:
            label = self.get_label(sample_idx)

            objects = list(self.get_objects_from_label(label).values())
            num_gt = len(objects)
            gt_boxes = np.squeeze(np.stack(objects), axis=1) if num_gt > 0 else np.array([])
            info['annos'] = {'gt_boxes_lidar': gt_boxes, 'name': np.full(num_gt, 'Car')}
            info['label'] = label

        return info

    def __getitem__(self, index):
        scenario, frame = self.sample_id_list[index]
        keys = list(self.frames[scenario].keys())

        # select random CAV for training
        if self.mode == 'train': 
            np.random.shuffle(keys)

        vehicle = keys.pop(0)
        while len(keys) > 0 and len(self.infos[scenario][vehicle][frame]['annos']['gt_boxes_lidar']) == 0:
            vehicle = keys.pop(0)
        info = copy.deepcopy(self.infos[scenario][vehicle][frame])

        calib = self.get_calib(index)

        input_dict = {
            'frame_id': (scenario, vehicle, frame),
            'calib': calib,
        }


        # collect all boxes from all vehicles and transform them to ego coords
        # this is necesarry since only visible objects are labeled for the ego
        gt_coop_boxes = {}
        for v in self.infos[scenario].keys():
            label = copy.deepcopy(self.infos[scenario][v][frame]['label'])
            label['lidar_pose'] = info['label']['lidar_pose']  # set lidar pose of ego vehicle
            gt_coop_boxes.update(self.get_objects_from_label(label))
        gt_coop_boxes.pop(int(vehicle), None)  # remove box of ego vehicle
        gt_coop_boxes = list(gt_coop_boxes.values())
        num_gt_coop_boxes = len(gt_coop_boxes)
        coop_lidar_boxes = np.squeeze(np.stack(gt_coop_boxes), axis=1) if num_gt_coop_boxes > 0 else np.array([])
        gt_coop_names = np.full(num_gt_coop_boxes, 'Car')

        assert len(coop_lidar_boxes > 0)

        input_dict.update({
            'gt_names': gt_coop_names,
            'gt_boxes': coop_lidar_boxes
        })

        points = self.get_lidar('/'.join([scenario, vehicle, frame]))
        assert np.shape(points)[0] > 0
        input_dict['points'] = points

        # filter empty boxes for training
        if self.mode == 'train' and not self.dataset_cfg.COLLECTIVE_DATA_PROCESSOR.use_coop_info:
            corners_lidar = box_utils.boxes_to_corners_3d(coop_lidar_boxes)
            num_points_in_gt = -np.ones(len(coop_lidar_boxes), dtype=np.int32)

            for k in range(len(coop_lidar_boxes)):
                flag = box_utils.in_hull(points[:, 0:3], corners_lidar[k])
                num_points_in_gt[k] = flag.sum()
            input_dict['gt_boxes'] = input_dict['gt_boxes'][num_points_in_gt > 0]
            input_dict['gt_names'] = input_dict['gt_names'][num_points_in_gt > 0]

        # print(scenario, vehicle, frame)
        # pcv = PCViewer(np.concatenate(coop_points, axis=0), input_dict['gt_boxes'])
        # pcv.draw_point_cloud()

        collective_cfg = self.dataset_cfg.COLLECTIVE_DATA_PROCESSOR

        if collective_cfg.use_coop_info or collective_cfg.early_fusion:
            if collective_cfg.early_fusion:
                coop_points = [points]
            else:
                coop_points = []
            ego_pose = np.array(info['label']['lidar_pose'])
            for v in self.infos[scenario].keys():
                if v != vehicle:
                    # load coop points and transform to ego coords
                    coop_pose = np.array(self.infos[scenario][v][frame]['label']['lidar_pose'])
                    ego_coop_dist = np.linalg.norm(coop_pose[:2]-ego_pose[:2])
                    if collective_cfg.limit_com_range and ego_coop_dist > collective_cfg.range_limit: # skip vehicles out of range
                        continue
                    pts = self.get_lidar('/'.join([scenario, v, frame]))
                    intensity = pts[:, 3].copy()  # save intensity for later
                    pts[:, 3] = 1  # convert to homogeneus coords
                    transform = np.array(transformations.local_to_local_transform(torch.as_tensor(coop_pose), torch.as_tensor(ego_pose)))
                    pts_transformed = pts.dot(transform.T)  # apply transformation
                    pts_transformed[:, 3] = intensity
                    coop_points.append(pts_transformed)
        else:
            coop_points = []

        # if early fusion is used, use all point clouds as ego
        if self.dataset_cfg.COLLECTIVE_DATA_PROCESSOR.early_fusion:
            # overwrite ego points by coop points
            input_dict['points'] = np.concatenate(coop_points, axis=0)

        data_dict = self.prepare_data(data_dict=input_dict)

        if not self.dataset_cfg.COLLECTIVE_DATA_PROCESSOR.early_fusion:
            data_dict['coop_points'] = coop_points
            data_dict = self.prepare_coop_data(data_dict=data_dict)

        # pcv = PCViewer(data_dict['points'], data_dict['gt_boxes'])
        # pcv.draw_point_cloud()

        if self.augmentor is not None:
            data_dict = self.augmentor.forward(data_dict)

        return data_dict
    
    def prepare_coop_data(self, data_dict):
        coop_data_config = self.dataset_cfg.COLLECTIVE_DATA_PROCESSOR
        coop_voxel_config = coop_data_config.transform_points_to_voxels
        # mask points out of range
        masked_points = []
        for i, cp in enumerate(data_dict['coop_points']):
            mask = common_utils.mask_points_by_range_with_z(cp, self.point_cloud_range)
            if np.any(mask):
                masked_points.append(data_dict['coop_points'][i][mask])
        data_dict['coop_points'] = masked_points
        

        if self.coop_voxel_generator is None:
            if coop_data_config.voxel_method == 'single_resolution':
                self.coop_voxel_generator = VoxelGeneratorWrapper(
                    vsize_xyz=coop_voxel_config.VOXEL_SIZE,
                    coors_range_xyz=self.point_cloud_range,
                    num_point_features=self.point_feature_encoder.num_point_features,
                    max_num_points_per_voxel=coop_voxel_config.MAX_POINTS_PER_VOXEL,
                    max_num_voxels=coop_voxel_config.MAX_NUMBER_OF_VOXELS[self.mode],
                )
            elif coop_data_config.voxel_method == 'multi_resolution':
                self.coop_voxel_generator = []
                for resolution in coop_data_config.resolutions:
                    self.coop_voxel_generator.append(
                        VoxelGeneratorWrapper(
                            vsize_xyz=resolution,
                            coors_range_xyz=self.point_cloud_range,
                            num_point_features=self.point_feature_encoder.num_point_features,
                            max_num_points_per_voxel=coop_voxel_config.MAX_POINTS_PER_VOXEL,
                            max_num_voxels=coop_voxel_config.MAX_NUMBER_OF_VOXELS[self.mode],
                        ))
        
        # sparse voxelization of coop points
        if coop_data_config.voxel_method == 'single_resolution':
            # use local point cloud as fallback if no other vehicles are in range
            if len(data_dict['coop_points']) == 0:
                coop_points = data_dict['points']
            else: 
                 coop_points = np.concatenate(data_dict['coop_points'], axis=0)

            coop_voxels, coop_coordinates, coop_num_points = self.coop_voxel_generator.generate(coop_points)
            data_dict['coop_voxels'] = coop_voxels
            data_dict['coop_voxel_coords'] = coop_coordinates
            data_dict['coop_voxel_num_points'] = coop_num_points
            # add resolution and pc range for voxel center calculation in vfe
            data_dict['resolution'] = coop_voxel_config.VOXEL_SIZE
            data_dict['point_cloud_range'] = self.point_cloud_range
        elif coop_data_config.voxel_method == 'multi_resolution':
            resolution_names = ['high', 'mid', 'low']
            points_at_resolution = self.map_coop_points_to_resolutions(data_dict['coop_points'], data_dict['points'], coop_data_config.assign_method)
            for resolution, r_name, voxel_generator, points in zip(coop_data_config.resolutions, resolution_names, self.coop_voxel_generator, points_at_resolution):
                points = np.concatenate(points, axis=0)
                coop_voxels, coop_coordinates, coop_num_points = voxel_generator.generate(points)
                data_dict['coop_voxels_'+r_name] = coop_voxels
                data_dict['coop_voxel_coords_'+r_name] = coop_coordinates
                data_dict['coop_voxel_num_points_'+r_name] = coop_num_points
                # add resolution and pc range for voxel center calculation in vfe
                data_dict['resolution_'+r_name] = resolution
            data_dict['point_cloud_range'] = self.point_cloud_range
        
        data_dict['voxel_method'] = coop_data_config.voxel_method

        #remove coop points from dict to save gpu mem
        del data_dict['coop_points'] 

        return data_dict

    def set_split(self, split):
        super().__init__(
            dataset_cfg=self.dataset_cfg, class_names=self.class_names, training=self.training, root_path=self.root_path, logger=None
        )

        self.split = split
        self.scenarios = sorted([os.path.basename(x) for x in glob.glob(str(self.root_path / (self.split + '/*/')), recursive=True)])
        self.frames = {}
        for scenario in self.scenarios:
            self.frames[scenario] = {}
            vehicles = sorted([os.path.basename(x) for x in glob.glob(
                str(self.root_path / (self.split + '/' + scenario + '/[!data_protocol]*/')), recursive=True)])
            for vehicle in vehicles:
                self.frames[scenario][vehicle] = sorted([os.path.basename(x).split('.')[0] for x in glob.glob(
                    str(self.root_path / (self.split + '/' + scenario + '/' + vehicle + '/*.pcd')),
                    recursive=True)])
        for key, value in self.frames.items():
            self.sample_id_list += [(key, x) for x in next(iter(value.values()))]
        self.infos = self.load_info_data(self.mode)

    def get_objects_from_label(self, label):
        output_dict = {}
        for id, obj in label['vehicles'].items():
            object_pose = np.hstack((np.array(obj['location'])+np.array(obj['center']), np.array(obj['angle'])))
            transform = np.array(transformations.local_to_local_transform(torch.as_tensor(object_pose), torch.as_tensor(label['lidar_pose'])))
            box_lidar_pos_hom = np.array([0, 0, 0, 1])
            box_lidar_pos = np.dot(transform, box_lidar_pos_hom).T[:3]
            yaw_rot = np.radians(object_pose[4] - label['lidar_pose'][4])
            yaw_rot = np.arctan2(np.sin(yaw_rot), np.cos(yaw_rot)) # norm to [-pi, pi]
            box_lidar = np.hstack((box_lidar_pos, np.array(obj['extent']) * 2, yaw_rot))
            box_lidar = np.expand_dims(box_lidar, 0)

            assert box_lidar.shape[0] > 0
            output_dict.update({id: box_lidar})

        return output_dict
    
    
    @staticmethod
    def map_coop_points_to_resolutions(coop_points, points, method='random'):
        # use local points as fallback if no coop_points are available
        if len(coop_points) == 0:
            points_at_resolution = [[points], [points], [points]]
            return points_at_resolution
        
        res_to_idx = {'high': 0, 'mid': 1, 'low': 2}
        points_at_resolution = [[], [], []]
        if method == 'random':
            # randomly assign coop points to resolutions
            for cp in coop_points:
                idx = random.randint(0, 2)
                points_at_resolution[idx].append(cp)
        elif method == 'none':
            pass
        else:
            for cp in coop_points:
                points_at_resolution[res_to_idx[method]].append(cp)

        # for empty resolutions assign ego points
        for pr in points_at_resolution:
            if not pr:
                pr.append(points)

        return points_at_resolution


def create_pkl_file(dataset_cfg, class_names, data_path, save_path):
    import pickle
    print('--------------- Start to collect infos ---------------')
    dataset = OPV2VDataset(dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path, training=True)
    train_split, val_split, test_split = 'train', 'validate', 'test'

    train_filename = save_path / ('opv2v_infos_%s.pkl' % train_split)
    val_filename = save_path / ('opv2v_infos_%s.pkl' % val_split)
    test_filename = save_path / ('opv2v_infos_%s.pkl' % test_split)

    print('--------------- train infos ---------------')
    dataset.set_split(train_split)
    opv2v_infos_train = dataset.get_infos(has_label=True, count_inside_pts=True)
    print('train samples:', len(opv2v_infos_train))
    with open(train_filename, 'wb') as f:
        pickle.dump(opv2v_infos_train, f)
    print('OPV2V info train file is saved to %s' % train_filename)

    print('--------------- val infos ---------------')
    dataset.set_split(val_split)
    opv2v_infos_val = dataset.get_infos(has_label=True, count_inside_pts=True)
    print('val samples:', len(opv2v_infos_val))
    with open(val_filename, 'wb') as f:
        pickle.dump(opv2v_infos_val, f)
    print('OPV2V info val file is saved to %s' % val_filename)

    print('--------------- test infos ---------------')
    dataset.set_split(test_split)
    opv2v_infos_test = dataset.get_infos(has_label=True, count_inside_pts=True)
    print('test samples:', len(opv2v_infos_test))
    with open(test_filename, 'wb') as f:
        pickle.dump(opv2v_infos_test, f)
    print('OPV2V info test file is saved to %s' % test_filename)
    print('---------------Data preparation Done---------------')


if __name__ == '__main__':
    import yaml
    from pathlib import Path
    from easydict import EasyDict
    dataset_path = Path('data/OPV2V')

    dataset_config = EasyDict(yaml.safe_load(open('tools/cfgs/dataset_configs/opv2v_dataset.yaml')))
    create_pkl_file(
            dataset_cfg=dataset_config,
            class_names=['Car'],
            data_path=dataset_path,
            save_path=dataset_path)
