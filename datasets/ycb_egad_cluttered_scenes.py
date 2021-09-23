import glob

import numpy as np
import pickle

from datasets.dataset import DatasetBase
from utils import util
import trimesh
import os

# TODO: Check the randomize indices function


class Dataset(DatasetBase):
    def __init__(self, opt):
        super(Dataset, self).__init__(opt)
        self.name = 'Dataset_ycb_egad_cluttered_scenes'
        self.data_dir = opt.data_dir

        self.prepare_dataset()

    def __getitem__(self, index):
        assert (index < self.dataset_size)
        file_index = self.indices[index]
        input_scene = self.scene_files[file_index]
        rgb_img = util.load_image(input_scene)
        segmentation_img = np.load(input_scene.replace("rgb_scene.png", "segmentation_scene.npy"))
        object_poses = np.load(input_scene.replace("rgb_scene.png", "object_poses_scene.npy"))
        objects_in_scene = pickle.load(open(input_scene.replace("rgb_scene.png", "objects_placed.pkl"), "rb"))
        # NOTE: ALSO DIVIDES BY 256:
        assert (rgb_img[:, :, 0].shape == (256, 256))
        rgb_img = rgb_img/255
        img = rgb_img - self.means_rgb
        img = img / self.std_rgb

        all_obj_verts = []
        all_obj_verts_resampled800 = []
        all_obj_faces = []
        for i, path in enumerate(objects_in_scene):
            key = os.path.splitext(os.path.basename(path))[0]
            p = np.matmul(object_poses[i, :3, :3],
                          self.obj_verts[key].T) + object_poses[i, :3, 3].reshape(-1, 1)
            all_obj_verts.append(p.T)
            all_obj_faces.append(self.obj_faces[key])
            points800 = np.matmul(
                object_poses[i, :3, :3], self.resampled_objects_800verts[key].T) + object_poses[i, :3, 3].reshape(-1, 1)
            all_obj_verts_resampled800.append(points800.T)

        # LOADING PRECOMPUTED AVAILABLE GRASPS!!!
        grasp_index = input_scene.split("/")[-1].split("_")[0]
        grasp_folder = input_scene.split("/")[-2]
        available_repr = np.load(os.path.join(self.data_dir, 'grasps', grasp_folder, 'barrett_representation_%s.npy' %
                                              (grasp_index)), allow_pickle=True)
        available_trans = np.load(os.path.join(self.data_dir, 'grasps', grasp_folder, 'barrett_translation_%s.npy' %
                                               (grasp_index)), allow_pickle=True)
        if available_repr.shape[-1] == 0:
            return self.__getitem__(np.random.randint(0, self.dataset_size))

        while True:
            obj_id = np.random.randint(0, len(objects_in_scene))
            if len(available_repr[obj_id]) > 0:
                grasp_ind = np.random.randint(0, len(available_repr[obj_id]))
                break
        grasp_pose = util.convert_qt_to_T_matrix(
            available_trans[obj_id][grasp_ind])
        grasp_repr = util.joints_to_grasp_representation(
            available_repr[obj_id][grasp_ind])
        rt_curr_frame = object_poses[obj_id]
        grasp_pose = rt_curr_frame.dot(grasp_pose)
        img = np.float32(img)

        mask_img = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)
        key = list(objects_in_scene.keys())[obj_id]
        ind_pybullet = objects_in_scene[key]
        mask_img[segmentation_img == ind_pybullet, 0] = 1
        sample = {'rgb_img': img,
                  'mask_img': mask_img,
                  'plane_eq': self.plane_eq,
                  'hand_gt_representation': grasp_repr,
                  'hand_gt_pose': grasp_pose,
                  '3d_points_object': all_obj_verts,
                  '3d_faces_object': all_obj_faces,
                  'object_points_resampled': all_obj_verts_resampled800,
                  }

        return sample

    def __len__(self):
        return self.dataset_size

    # TODO: Rename the prepare_curriculum method
    def read_all_scene_files_with_varying_amount_of_clutter(self):
        self.scene_files = glob.glob(os.path.join(self.data_dir, 'scenes',
                                                  "*_objects_placed", "*rgb_scene.png"))
        self.scene_files.sort()
        self.current_total_number_of_scenes = len(self.scene_files)
        self.dataset_size = int(self.current_total_number_of_scenes/4)

    def randomize_indices(self):
        self.indices = np.random.choice(self.current_total_number_of_scenes, self.dataset_size, replace=False)

    def prepare_dataset(self):
        self.read_all_scene_files_with_varying_amount_of_clutter()

        self.randomize_indices()

        self.models = glob.glob(os.path.abspath(self.data_dir) + '/models/**/*.obj')

        obj_verts = {}
        obj_faces = {}
        self.resampled_objects_800verts = {}
        for i in range(len(self.models)):
            obj = trimesh.load(self.models[i])
            key = os.path.splitext(os.path.basename(self.models[i]))[0]
            obj_verts[key] = obj.vertices
            obj_faces[key] = obj.faces
            obj_orig = trimesh.load(self.models[i])
            resampled = trimesh.sample.sample_surface_even(obj_orig, 800)[0]
            if resampled.shape[0] < 800:
                resampled = trimesh.sample.sample_surface(obj_orig, 800)[0]
            self.resampled_objects_800verts[key] = resampled

        self.obj_verts = obj_verts
        self.obj_faces = obj_faces

        self.plane_eq = np.array([0, 0, 1, 0])
