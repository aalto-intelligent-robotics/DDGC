import torch.utils.data as data
import torch


class DatasetFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_by_name(dataset_name, opt):
        if dataset_name == 'ycb_egad_cluttered_scenes_train':
            from datasets.ycb_egad_cluttered_scenes import Dataset
            dataset = Dataset(opt)
        elif dataset_name == 'ycb_egad_cluttered_scenes_test':
            from datasets.ycb_egad_cluttered_scenes_val import Dataset
            dataset = Dataset(opt)
        else:
            raise ValueError("Dataset [%s] not recognized." % dataset_name)

        print('Dataset {} was created'.format(dataset.name))
        return dataset


class DatasetBase(data.Dataset):
    def __init__(self, opt):
        super(DatasetBase, self).__init__()
        self.name = 'BaseDataset'
        self.opt = opt

        self.all_object_vertices = []
        self.all_object_faces = []
        self.all_object_textures = []
        self.all_object_vertices_simplified = []
        self.resampled_objects_800verts = []
        self.all_object_faces_simplified = []
        self.all_grasp_translations = []
        self.all_grasp_rotations = []
        self.all_grasp_hand_configurations = []

        # Resnet normalization values
        self.means_rgb = [0.485, 0.456, 0.406]
        self.std_rgb = [0.229, 0.224, 0.225]

        self.IMG_EXTENSIONS = [
            '.jpg', '.JPG', '.jpeg', '.JPEG',
            '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
        ]

    def get_name(self):
        return self.name

    def collate_fn(self, args):
        length = len(args)
        keys = list(args[0].keys())
        data = {}

        for _, key in enumerate(keys):
            data_type = []

            if key == 'rgb_img' or key == 'mask_img' or key == 'noise_img' or key == 'plane_eq' or key == 'hand_gt_representation' or key == 'hand_gt_pose':
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
