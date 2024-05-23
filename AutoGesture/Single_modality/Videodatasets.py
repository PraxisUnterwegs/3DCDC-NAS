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

class Videodatasets(Dataset):
    def __init__(self, dataset_root, ground_truth,  typ, sample_duration=16, phase='train'):

        def get_data_list_and_label(data_df, typ):
            #T = 0 if typ == 'M' else 1
            result = []
            for line in open(data_df).readlines():
                line = line.strip()
                if not line:
                    continue
                c1,c2,c3 = line.split(" ")
                #i train/003/M_00419.avi
                data_path = "/".join(c1.split('/')[:])
                label = int(c3)
                result.append((data_path, label))
                #o ('003/M_00401.avi', 233)
            return result
        
        self.dataset_root = dataset_root
        self.sample_duration = sample_duration
        self.phase = phase

        self.transform = transforms.Compose([Normaliztion(), transforms.ToTensor()])
        lines= filter(lambda x: x[1] > 8, get_data_list_and_label(ground_truth, typ))
        #self.inputs = list(lines)
        self.train_miss_path_num = 0
        self.valid_miss_path_num = 0
        self.test_miss_path_num = 0
        self.inputs,output1 = self.fixInputsfiles(list(lines))
 
        
    def fixInputsfiles(self, input):
        train_path = "/home/jzf/Tasks__Gestures_Classification/3DCDC-NAS/Dataset/"  # TODO
        valid_path = "/home/jzf/Tasks__Gestures_Classification/3DCDC-NAS/Dataset/"  # TODO
        test_path = "/home/jzf/Tasks__Gestures_Classification/3DCDC-NAS/Dataset/"  # TODO
        if self.phase == "train":
            errorInputs = []
            inputs = []
            inputss = []
            with open("failure_path_train.txt", "w") as file1:
                file1.truncate(0)
            for videopath, label in input: # videopath = "train/178/K_35598.avi"
                full_path = train_path + videopath
                if os.path.exists(full_path[:-4]) & os.path.exists(os.path.join(full_path[:-4], "000000.jpg")):
                    inputs.append((videopath[6:], label))  # only "178/K_35598.avi", without "train/"
                    inputss.append((videopath, label)) # with "train/178/K_35598.avi"
                    #print(f"this path exists: {full_path[:-4]}")
                else:
                    errorInputs.append((videopath, label))
                    #print(f"this path does not exist: {path[-6:] + videopath}")
                    #print(f"this path does not exist: {full_path[:-4]}")
                    self.train_miss_path_num = self.train_miss_path_num + 1
                    with open('failure_path_train.txt', 'a', encoding='utf-8') as ff1:
                        ff1.write(f"{full_path[:-4]}\n")
            ## check if path video exists
        elif self.phase == "valid":
            errorInputs = []
            inputs = []
            inputss = []
            with open("failure_path_valid.txt", "w") as file2:
                file2.truncate(0)
            for videopath, label in input: 
                full_path = valid_path + videopath
                if os.path.exists(full_path[:-4]) & os.path.exists(os.path.join(full_path[:-4], "000000.jpg")):
                    inputs.append((videopath[6:], label)) # only "002/M_00398.avi", without "valid/"
                    inputss.append((videopath, label))
                else:
                    errorInputs.append((videopath, label))
                    #print(f"this path does not exist: {full_path[:-4]}")
                    self.valid_miss_path_num = self.valid_miss_path_num + 1
                    with open('failure_path_valid.txt', 'a', encoding='utf-8') as ff2:
                        ff2.write(f"{full_path[:-4]}\n")
        elif self.phase == "test":
            errorInputs = []
            inputs = []
            inputss = []
            with open("failure_path_test.txt", "w") as file3:
                file3.truncate(0)
            for videopath, label in input: # videopath = "/178/K_35598.avi"
                full_path = test_path + videopath
                if os.path.exists(full_path[:-4]) & os.path.exists(os.path.join(full_path[:-4], "000000.jpg")):
                    inputs.append((videopath[5:], label)) # only "003/M_00401.avi", without "test/"
                    inputss.append((videopath, label))
                else:
                    errorInputs.append((videopath, label))
                    #print(f"this path does not exist: {full_path[:-4]}")
                    self.test_miss_path_num = self.test_miss_path_num + 1
                    with open('failure_path_test.txt', 'a', encoding='utf-8') as ff3:
                        ff3.write(f"{full_path[:-4]}\n")
        print(f"the number of missing path of train:{self.train_miss_path_num}")
        print(f"the number of missing path of valid:{self.valid_miss_path_num}")
        print(f"the number of missing path of test:{self.test_miss_path_num}")
        inputing = list(inputs)
        inputted = list(inputss)
        return inputing, inputted
        
        
        
        
        #print(self.inputs)
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
            return np.array(img.resize((112, 112)))
            # return image_to_np(img.resize((112, 112)))
        def Sample_Image(imgs_path, frame_count):
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
            sl = f(frame_count)
            frams = []
            for a in sl:
                # img = transform(accimage.Image(os.path.join(imgs_path, "%06d.jpg" % a))) #if use Accimage
                img = transform(Image.open(os.path.join(imgs_path, "%06d.jpg" % a)))
                frams.append(self.transform(img).view(3, 112, 112, 1))
            return torch.cat(frams, dim=3).type(torch.FloatTensor)
        
        videoPath = self.inputs[index][0]
        videoPath = videoPath.split(".")[0] # remove .avi from the video path 
        data_path = os.path.join(self.dataset_root, videoPath )
        currentImageFramesMaximum = len(os.listdir(data_path))
        #clip = Sample_Image(data_path, self.inputs[index][1])
        clip = Sample_Image(data_path, currentImageFramesMaximum)
        return clip.permute(0, 3, 1, 2), self.inputs[index][1]

    def __len__(self):
        return len(self.inputs)