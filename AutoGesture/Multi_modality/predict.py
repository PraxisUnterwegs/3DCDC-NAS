import cv2 #
import matplotlib.pyplot as plt 
import numpy as np   
import torch
import os
import os.path as osp
import shutil
import torch.nn as nn
from PIL import Image
from torchvision import datasets, models, transforms
import skimage.exposure
import depthai
import blobconverter
from models.AutoGesture_RGBD_searched_12layers_DiffChannels import AutoGesture_RGBD_12layers
from utils.config import Config
from train_AutoGesture_CDC_RGBD_sgd_12layers import parse, Module
import yaml
import time
import oak_d_driver as oakd
from RingBuffer import RingBuffer

# config_file_path = 'config.yml'
# # 使用safe_load来防止加载任何非标准的YAML标签，提高安全性
# with open(config_file_path, 'r') as file:
#     config = yaml.safe_load(file)


# class SELFMODEL(nn.Module):
#     def __init__(self, model_name= params['model'], out_features=params['num_classes'],
#                  pretrained=True):
#         super().__init__()
#         # timm.create_model根据指定的模型名称自动下载并加载预训练的权重，然后会根据指定的模型名称返回一个 PyTorch 模型对象，该对象已经加载了预训练的权重。
#         # 这样，用户可以立即开始使用该模型进行图像分类任务。
#         self.model = timm.create_model(model_name, pretrained=pretrained)  # 从预训练的库中加载模型
#         # self.model = timm.create_model(model_name, pretrained=pretrained, checkpoint_path="pretrained/resnet50d_ra2-464e36ba.pth")  # 从预训练的库中加载模型
#         # classifier
#         if model_name[:3] == "res":
#             n_features = self.model.fc.in_features  # Modify the number of fully connected layers
#             self.model.fc = nn.Linear(n_features, out_features)  # Modify to the number of categories corresponding to this task
#         elif model_name[:3] == "vit":
#             n_features = self.model.head.in_features  # Modify the number of fully connected layers
#             self.model.head = nn.Linear(n_features, out_features)  # Modify to the number of categories corresponding to this task
#         else:
#             n_features = self.model.classifier.in_features
#             self.model.classifier = nn.Linear(n_features, out_features)
#         print(self.model)  

#     def forward(self, x):  
#         x = self.model(x)
#         return x
    
    
# def get_torch_transforms(img_size=224):
#     data_transforms = {
#         'train': transforms.Compose([
#             # transforms.RandomResizedCrop(img_size),
#             transforms.Resize((img_size, img_size)),
#             transforms.RandomHorizontalFlip(p=0.2),
#             transforms.RandomRotation((-5, 5)),
#             transforms.RandomAutocontrast(p=0.2),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#         ]),
#         'val': transforms.Compose([
#             # transforms.Resize((img_size, img_size)),
#             # transforms.Resize(256),
#             # transforms.CenterCrop(img_size),
#             transforms.Resize((img_size, img_size)),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#         ]),
#         'test':transforms.Compose([
#             # transforms.Resize((img_size, img_size)),
#             # transforms.Resize(256),
#             # transforms.CenterCrop(img_size),
#             transforms.Resize((img_size, img_size)),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#         ]),
#     }
#     return data_transforms



def cv_show(name,img):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
# vc = cv2.VideoCapture('02_Video/00_Scenery.mp4')
# if vc.isOpened():   # if the file is open properly
#     open, frame = vc.read() # read the first frame, open return the read status,frame is the frame read
#     print(open) # When opened properly, open will return True
#     cv_show('image_scenery',frame)  # display the frame
# else:
#     open = False


preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
    


# model = torch.load('AutoGesture/checkpoints/epoch26-MK-valid_0.6004-test_0.6224.pth')  # TODO
# model.eval()  # eval mode(prepare for the inference)

# args = Config(parse())
# config_file_path = 'AutoGesture/Multi_modality/config_hpc.yml'

# with open(config_file_path, 'r') as file:
#     config = yaml.safe_load(file)


# init_channels8 = config['common']['init_channels8']
# init_channels16 = config['common']['init_channels16']
# init_channels32 = config['common']['init_channels32']
# num_classes = config['common']['num_classes']
# layers = config['common']['layers']

args = Config(parse())

device = torch.device('cpu')
model = Module(args)
#model = AutoGesture_RGBD_12layers(init_channels8, init_channels16, init_channels32, num_classes, layers)  # 实例化模型架构
checkpoint = torch.load('/home/jzf/Tasks__Gestures_Classification/3DCDC-NAS/AutoGesture/checkpoints/epoch26-MK-valid_0.6004-test_0.6224.pth',
                        map_location=device)
model.load_state_dict(checkpoint)  # 加载模型参数
model.eval() 



pipeline = oakd.init()

# Pipeline is now finished, and we need to find an available device to run our pipeline
# we are using context manager here that will dispose the device after we stop using it
with depthai.Device(pipeline) as cam:
    # From this point, the Device will be in "running" mode and will start sending data via XLink
    # To consume the device results, we get two output queues from the device, with stream names we assigned earlier
    q_rgb = cam.getOutputQueue("rgb")
    q_nn = cam.getOutputQueue("nn")
    q_left = cam.getOutputQueue("left")
    q_right = cam.getOutputQueue("right")
    q_depth = cam.getOutputQueue("depth")
        
    # Here, some of the default values are defined. Frame will be an image from "rgb" stream, detections will contain nn results
    rgbframe = None
    monoLeftFrame = None
    monoRightFrame = None
    depthFrame = None
    detections = []
    input_tensor_rgb = None
    input_tensor_depth = None
    preparedRgb = None
    buffer_rgb = RingBuffer(32)
    buffer_depth = RingBuffer(32)
    class Normaliztion(object):
        """
            same as mxnet, normalize into [-1, 1]
            image = (image - 127.5)/128
        """
        def __call__(self, Image):
            new_video_x = (Image - 127.5) / 128
            return new_video_x
    tensorTransformer = transforms.Compose([Normaliztion(), transforms.ToTensor()])
    
    def frameNorm(frame, bbox):
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

    while(True):
        # read frame
        ##############################################################################
        # we try to fetch the data from nn/rgb queues. tryGet will return either the data packet or None if there isn't any
        in_rgb = q_rgb.tryGet()
        in_nn = q_nn.tryGet()
        in_left = q_left.tryGet()
        in_right = q_right.tryGet()
        in_depth = q_depth.tryGet()
        
        
        def transform_params( resize=(320, 240), crop_size=224, flip=0.5):
            left, top = (resize[0] - crop_size) // 2, (resize[1] - crop_size) // 2
            is_flip = False
            return (left, top, left + crop_size, top + crop_size), is_flip
        
        resize = (320, 240)  # default | (256, 256) may be helpful
        crop_rect, is_flip = transform_params(resize=resize, flip=1.0)
        left, top, right, bottom = crop_rect
        def transform(img):
            img = cv2.resize(img,resize)
            #img = cv2.crop(crop_rect)
            img = img[top:bottom, left:right]
            # #if is_flip:
            # #    img = img.transpose(Image.FLIP_LEFT_RIGHT)
            return cv2.resize(img,(112, 112)).astype(np.float32)
            #return img
        
        frame_list = []
        sample_duration = 32
        
        
        
        
        if in_rgb is not None:
            # If the packet from RGB camera is present, we're retrieving the frame in OpenCV format using getCvFrame
            rgbframe = in_rgb.getCvFrame()
            preparedRgb = transform(rgbframe)
            preparedRgbTensor = tensorTransformer(preparedRgb).view(3, 112, 112, 1)
            buffer_rgb.push(preparedRgbTensor)            
        if in_nn is not None:
            # when data from nn is received, we take the detections array that contains mobilenet-ssd results
            detections = in_nn.detections
        if in_left is not None:
            monoLeftFrame = in_left.getCvFrame()
        if in_right is not None:
            monoRightFrame = in_right.getCvFrame()
        if in_depth is not None:
            depthFrame = in_depth.getCvFrame()
            preparedDepth = transform(depthFrame)
            bgr_image = cv2.cvtColor(preparedDepth,cv2.COLOR_GRAY2BGR)
            preparedDepthTensor = tensorTransformer(bgr_image).view(3, 112, 112, 1)
            buffer_depth.push(preparedDepthTensor)
            
        ##############################################################################
        
        # convert OpenCV image into PIL image
        # pil_img = Image.fromarray(cv2.cvtColor(rgbframe, cv2.COLOR_BGR2RGB))
        
        # preprocess - data augmentaion
        # input_tensor = preprocess(Image)
        # input_tensor = preprocess(pil_img)
        #input_batch = input_tensor.unsqueeze(0)  # Add a dimension because the model requires a batch as input
        
        #input_tensor_rgb = rgbframe
        #input_tensor_depth = depthFrame
        #input_tensor_rgb = torch.from_numpy(rgbframe).copy()
        #input_tensor_depth = torch.from_numpy(depthFrame).copy()
        # torch.from_numpy(frames_array).permute(0, 3, 1, 2)
        if buffer_rgb.is_full() and buffer_depth.is_full():
            
            start = time.time()
            numpy_rgb = np.stack(buffer_rgb.get())
            #print(f"输出{numpy_rgb.shape}")
            input_tensor_rgb = torch.from_numpy(numpy_rgb).permute(4,1,0,2,3)
            #frams.append(self.transform(img).view(3, 112, 112, 1))
            #input_tensor_rgb = torch.cat(buffer_rgb.get(), dim=3).type(torch.FloatTensor)
            #print(f"输出：{input_tensor_rgb.shape}")
            
            numpy_depth = np.stack(buffer_depth.get())
            input_tensor_depth = torch.from_numpy(numpy_depth).permute(4,1,0,2,3)
            #input_tensor_depth = torch.cat(buffer_depth.get(), dim=3).type(torch.FloatTensor)
            #print(f"输出：{input_tensor_depth.shape}")
        
            # if input_tensor_rgb is not None and input_tensor_depth is not None:
            #     # inference
            with torch.no_grad():  # no bp & gradients compute during inference
                #print(f"input_tensor_rgb: {input_tensor_rgb.shape}")
                output = model(input_tensor_rgb, input_tensor_depth)  # input data during inference
            
                # get the predict result
                _, predicted = torch.max(output, 1)  # return the largest probability classifcation result index
                # predicted catch the classification discrimination tag's index ($$id0)
                prediction = predicted.item()  # convert the index into int type
                
                cv2.putText(rgbframe, f"Prediction: {prediction}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                # open a window to show the frame
                cv2.imshow('frame', rgbframe)
            
            dt = time.time() - start
            print("dt: %s" % (dt))

        #     # real-time rendering display the classification result (244:shake hands) on the screen
            
        
        # # press 'q' quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # if preparedRgb is not None:
        #     cv2.imshow('frame', preparedRgb)
    # release the cam object
    cv2.destroyAllWindows()