import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, set_image_backend
from PIL import Image
import os
import math
import random
import numpy as np
# import functools
# import accimage
# set_image_backend('accimage')


class Normaliztion(object):
    """
        same as mxnet, normalize into [-1, 1]
        image = (image - 127.5)/128
    """

    def __call__(self, Image):
        new_video_x = (Image - 127.5) / 128
        return new_video_x


class Videodatasets_RGBD(Dataset):
    def __init__(self, dataset_root, ground_truth1, typ1, ground_truth2, typ2, sample_duration=16, phase='train'):
    # def __init__(self, dataset_root, ground_truth1, typ1, ground_truth2, typ2, sample_duration=16, phase='train'):
    
        def get_data_list_and_label(data_df, typ):
            result = []
            textlines = []
            with open(data_df) as f:
                textlines = f.readlines()
            for line in textlines:
                line = line.strip()
                if not line:
                    continue
                c1,c2,c3 = line.split(" ")
                #print(c1,c2)
                #i train/003/M_00419.avi
                if typ == 'rgb':
                    data_path = "/".join(c1.split('/')[1:])
                elif typ == 'depth':
                    data_path = "/".join(c2.split('/')[1:])
                else:
                    continue
                label = int(c3)
                result.append((data_path, label))
                #o ('003/M_00401.avi', 233)
            return result

        self.dataset_root = dataset_root
        self.sample_duration = sample_duration
        self.phase = phase
        self.typ1, self.typ2 = typ1, typ2
        self.transform = transforms.Compose([Normaliztion(), transforms.ToTensor()])

        lines = filter(lambda x: x[1] > 7, get_data_list_and_label(ground_truth1, typ1))
        lines2 = filter(lambda x: x[1] > 7, get_data_list_and_label(ground_truth2, typ2))
        self.inputs = list(lines)
        self.inputs2 = list(lines2)

    def transform_params(self, resize=(320, 240), crop_size=224, flip=0.5):
        if self.phase == 'train':
            left, top = random.randint(0, resize[0] - crop_size), random.randint(0, resize[1] - crop_size)
            is_flip = True if random.uniform(0, 1) > flip else False
        else:
            left, top = (resize[0] - crop_size) // 2, (resize[1] - crop_size) // 2
            is_flip = False
        return (left, top, left + crop_size, top + crop_size), is_flip

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        resize = (320, 240)  # default | (256, 256) may be helpful
        crop_rect, is_flip = self.transform_params(resize=resize, flip=1.0)  # no flip

        def image_to_np(image):
            """
            Returns:
                np.ndarray: Image converted to array with shape (width, height, channels)
            """
            image_np = np.empty([image.channels, image.height, image.width], dtype=np.uint8)
            image.copyto(image_np)
            image_np = np.transpose(image_np, (1, 2, 0))
            return image_np

        def transform(img):
            img = img.resize(resize)
            img = img.crop(crop_rect)
            if is_flip:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            return np.array(img.resize((112, 112)))  # Image.open
            # return image_to_np(img.resize((112, 112)))    # accimage.Image

        def Sample_Image(imgs_path, sl):
            frams = []
            for a in sl:
                # img = transform(accimage.Image(os.path.join(imgs_path, "%06d.jpg" % a))) #if use Accimage
                img = transform(Image.open(os.path.join(imgs_path, "%06d.jpg" % a)))
                frams.append(self.transform(img).view(3, 112, 112, 1))
            return torch.cat(frams, dim=3).type(torch.FloatTensor)

        sn = self.sample_duration
        if self.phase == 'train':
            f = lambda n: [(lambda n, arr: n if arr == [] else random.choice(arr))(n * i / sn,
                                                                                   range(int(n * i / sn),
                                                                                         max(int(n * i / sn) + 1,
                                                                                             int(n * (
                                                                                                     i + 1) / sn))))
                           for i in range(sn)]
        else:
            f = lambda n: [(lambda n, arr: n if arr == [] else int(np.mean(arr)))(n * i / sn, range(int(n * i / sn),
                                                                                                    max(int(
                                                                                                        n * i / sn) + 1,
                                                                                                        int(n * (
                                                                                                                i + 1) / sn))))
                           for i in range(sn)]
        
        #print(index)
        #print(self.inputs[index])
        videoPath = self.inputs[index][0]
        videoPath = videoPath.split(".")[0] # remove .avi from the video path 
        data_path = os.path.join(self.dataset_root, videoPath) 
        currentImageFramesMaximum = len(os.listdir(data_path))
        sl = f(currentImageFramesMaximum)

        #Iso
        #data_path = os.path.join('/'.join(self.dataset_root.split('/')[:-1]), self.typ1, self.phase,
        #                         '/'.join(self.inputs[index][0].split('/')[-3:]))
        clip = Sample_Image(data_path, sl)

        #data_path2 = os.path.join('/'.join(self.dataset_root.split('/')[:-1]), self.typ2,self.phase,
        #                          '/'.join(self.inputs2[index][0].split('/')[-3:]))
        videoPath2 = self.inputs2[index][0]
        videoPath2 = videoPath2.split(".")[0] # remove .avi from the video path 
        data_path2 = os.path.join(self.dataset_root, videoPath) 
        clip2 = Sample_Image(data_path2, sl)

        # check if label is the same (for both input lists) depth&rgb
        assert self.inputs[index][1] == self.inputs2[index][1]
        return clip.permute(0, 3, 1, 2), self.inputs[index][1], clip2.permute(0, 3, 1, 2)

    def __len__(self):
        return len(self.inputs2)