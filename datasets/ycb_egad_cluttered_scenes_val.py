import glob
import os
import os.path
from zodbpickle import pickle
import numpy as np
import torch
import trimesh
from PIL import Image
from skimage.transform import resize
import itertools
from data.dataset import DatasetBase
from utils import util
from utils.data_utils import fast_load_obj


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        if img.size[-1] > 3:
            rgb_img = Image.new("RGB", img.size, (255, 255, 255))
            rgb_img.paste(img, mask=img.split()[3])
            img = rgb_img
        return np.array(img)


class Dataset(DatasetBase):
    def __init__(self, opt, mode):
        super(Dataset, self).__init__(opt, mode)
        self._name = 'pybullet_barrett_cluttered'
        self.data_dir = opt.data_dir
        self.use_7_taxonomies = opt.use_7_taxonomies
        self.use_curriculum = opt.use_curriculum
        self.objects_to_place = opt.objects_in_scene
        self.mode = mode
        # read dataset
        self.prepare_dataset()
        self.noise_units = 12
        self.loader = pil_loader
        # Resnet norms
        self.means_rgb = [0.485, 0.456, 0.406]
        self.std_rgb = [0.229, 0.224, 0.225]
        self.depth_normalization = 3.0

    def __getitem__(self, index):
        assert (index < self.dataset_size)
        # FORCE IMAGE FROM VIDEO 5
        file_index = self.indices[index]
        input_scene = self.current_curriculum_files[file_index]  # self.data_dir + '/pybullet_output/'+str(index)+'_rgb_scene.png'
        rgb_img = self.loader(input_scene)
        segmentation_img = np.load(input_scene.replace("rgb_scene.png", "segmentation_scene.npy"))
        depth_img = np.load(input_scene.replace("rgb_scene.png", "depth_scene.npy"))
        # object_poses = np.load(input_scene.replace("rgb_scene.png", "object_poses_scene.npy"))
        object_poses = pickle.load(open(input_scene.replace("rgb_scene.png", "object_poses_scene.npy"), "rb"))
        objects_in_scene = pickle.load(open(input_scene.replace("rgb_scene.png", "objects_placed.pkl"), "rb"))
        scene_number = os.path.basename(input_scene).split("_")[0]
        objects_in_scene_keys = list(objects_in_scene.keys())
        # NOTE: ALSO DIVIDES BY 256:
        assert (rgb_img[:, :, 0].shape == (256, 256))
        depth_img = depth_img/self.depth_normalization
        rgb_img = rgb_img/255
        img = rgb_img - self.means_rgb
        img = img / self.std_rgb

        all_obj_verts = []
        all_obj_verts_resampled800 = []
        all_obj_faces = []
        for i in range(len(object_poses)):
            # key = path.split("/")[-2]
            key_object = objects_in_scene_keys[i].split("/")[-1]
            key_pose = objects_in_scene_keys[i]
            p = np.matmul(object_poses[key_pose][:3, :3],
                          self.obj_verts[key_object].T) + object_poses[key_pose][:3, 3].reshape(-1, 1)
            all_obj_verts.append(p.T)
            all_obj_faces.append(self.obj_faces[key_object])
            points800 = np.matmul(
                object_poses[key_pose][:3, :3], self.resampled_objects_800verts[key_object].T) + object_poses[key_pose][:3, 3].reshape(-1, 1)
            all_obj_verts_resampled800.append(points800.T)

        # Get list of all obj poses (Vertices and faces!)
        # for i, path in enumerate(objects_in_scene):
        #    key = os.path.splitext(os.path.basename(path))[0]  # mesh_files[i].split("/")[-2]
        #    p = np.matmul(object_poses[i, :3, :3],
        #                  self.obj_verts[key].T) + object_poses[i, :3, 3].reshape(-1, 1)
        #    all_obj_verts.append(p.T)
        #    all_obj_faces.append(self.obj_faces[key])
        #    points800 = np.matmul(
        #        object_poses[i, :3, :3], self.resampled_objects_800verts[key].T) + object_poses[i, :3, 3].reshape(-1, 1)
        #    all_obj_verts_resampled800.append(points800.T)

        ind = np.random.randint(0, len(objects_in_scene))
        img = np.float32(img)

        mask_img = np.zeros((img.shape[0], img.shape[1], len(self.objects_to_place)), np.uint8)
        for i, object_in_scene in enumerate(objects_in_scene.keys()):
            ind_pybullet = objects_in_scene[object_in_scene]
            mask_img[segmentation_img == ind_pybullet, i] = 1
        # pack data
        sample = {'rgb_img': img,
                  'mask_img': mask_img,
                  'depth_img': depth_img,
                  'object_id': ind,
                  'plane_eq': self.plane_eq,
                  'taxonomy': None,
                  'hand_gt_representation': np.empty(0),
                  'hand_gt_pose': np.empty(0),
                  '3d_points_object': all_obj_verts,
                  '3d_faces_object': all_obj_faces,
                  'object_points_resampled': all_obj_verts_resampled800,
                  'fullsize_imgs': rgb_img,
                  'scene_number': scene_number,
                  }

        return sample

    def __len__(self):
        return self.dataset_size

    def increse_difficulty(self):
        if self.current_curriculum < len(self.objects_to_place)-1:
            self.current_curriculum += 1
            self.current_total_number_of_scenes = self.number_of_scenes[self.current_curriculum]
            self.current_curriculum_files = self.scene_files[self.current_curriculum]
            self.current_scene = self.scenes[self.current_curriculum]
            self.randomize_indices()

    def prepare_curriculum(self):
        # self.number_of_objects_per_scene = os.listdir(self.data_dir + '/pybullet_output/')
        # Sort in ascending difficulty, i.e. the more objects in a scene the more difficult it is
        # self.number_of_objects_per_scene.sort()
        self.number_of_scenes = []
        self.scene_files = []
        # for scenes in self.number_of_objects_per_scene:
        self.scenes = []
        for num_objects_to_place in self.objects_to_place:
            # One scene consist of 6 files: rgb, depth, segmentation, object poses, camera pose and object names
            # scene_files = glob.glob(os.path.join(self.data_dir,
            self.scenes.append(num_objects_to_place+"_objects_placed")
            scene_files = glob.glob(os.path.join(self.data_dir,
                                                 num_objects_to_place+"_objects_placed", "*rgb_scene.png"))
            self.number_of_scenes.append(len(scene_files))
            self.scene_files.append(scene_files)

        self.current_curriculum = 0
        if self.use_curriculum:
            self.current_curriculum_files = self.scene_files[0]
            self.current_total_number_of_scenes = self.number_of_scenes[0]
            self.current_scene = self.scenes[0]
        else:
            self.current_curriculum_files = list(itertools.chain.from_iterable(self.scene_files))
            self.current_total_number_of_scenes = sum(self.number_of_scenes)

    def randomize_indices(self):
        self.indices = np.random.choice(self.current_total_number_of_scenes, self.dataset_size, replace=False)
        self.indices.sort()

    def prepare_dataset(self):
        self.prepare_curriculum()
        # We always keep the same number of mini-batches per epoch. Altough the total dataset size grows with harder curriculum, we do not grow how many mini-batches we sample per epoch
        self.dataset_size = self.number_of_scenes[0]
        self.randomize_indices()
        self.models = glob.glob(os.path.abspath(self.data_dir) + '/models/*.obj')

        obj_verts = {}
        obj_faces = {}
        self.resampled_objects_800verts = {}
        for i in range(len(self.models)):
            obj = fast_load_obj(open(self.models[i], 'rb'))[0]
            # key = os.path.splitext(os.path.basename(self.models[i]))[0]
            key = os.path.basename(self.models[i])
            obj_verts[key] = obj['vertices']  # - offset_ycbs[i])
            obj_faces[key] = obj['faces']
            obj_orig = trimesh.load(self.models[i])
            resampled = trimesh.sample.sample_surface_even(obj_orig, 800)[0]
            if resampled.shape[0] < 800:
                resampled = trimesh.sample.sample_surface(obj_orig, 800)[0]
            self.resampled_objects_800verts[key] = resampled

        self.obj_verts = obj_verts
        self.obj_faces = obj_faces
        self.plane_eq = np.array([0, 0, 1, 0])

    def _read_ids(self, file_path, extension):
        files = os.listdir(file_path)
        files = [f.replace(extension, '') for f in files]
        return files

    def collate_fn(self, args):
        length = len(args)
        keys = list(args[0].keys())
        data = {}

        for i, key in enumerate(keys):
            data_type = []

            if key == 'rgb_img' or key == 'depth_img' or key == 'mask_img' or key == 'noise_img' or key == 'plane_eq' or key == 'hand_gt_representation' or key == 'hand_gt_pose':
                for j in range(length):
                    data_type.append(torch.FloatTensor(args[j][key]))
                data_type = torch.stack(data_type)
            elif key == 'label':
                labels = []
                for j in range(length):
                    labels.append(args[j][key])
                data_type = torch.LongTensor(labels)
            else:
                for j in range(length):
                    data_type.append(args[j][key])
            data[key] = data_type

        return data
