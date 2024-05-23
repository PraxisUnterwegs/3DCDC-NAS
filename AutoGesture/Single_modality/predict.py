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
from network.AutoGesture_searched import AutoGesture
from utils.config import Config
from train_AutoGesture_3DCDC import parse, Module
import yaml
import time
import oak_d_driver as oakd




preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])



def Frame_Extraction(n, sn):
    result = []
    for i in range(sn):
        start = int(n * i / sn)
        end = max(int(n * (i + 1) / sn), start + 1)
        arr = range(start, end)
        if arr:  # 如果end 比 start 大，则arr不为空
            mean = int(np.mean(arr)) 
        else:
            mean = n
        result.append(mean)
    return result


#extract_frames_id_list = Frame_Extraction(10, sample_duration)
sample_duration = 32 
    

args = Config(parse())

device = torch.device('cpu')  # TODO
#device = torch.device('cuda:0')  # TODO
model = Module(args)
#model = AutoGesture_RGBD_12layers(init_channels8, init_channels16, init_channels32, num_classes, layers)  # 实例化模型架构
checkpoint = torch.load('/home/jzf/Tasks__Gestures_Classification/3DCDC-NAS/AutoGesture/checkpoints/epoch22-M-valid_0.5888-test_0.6289.pth',
                        map_location=device)
model.load_state_dict(checkpoint)  # 加载模型参数
model.eval() 



pipeline = oakd.init()


with depthai.Device(pipeline) as cam:

    q_rgb = cam.getOutputQueue("rgb")
    #q_left = cam.getOutputQueue("left")
    #q_right = cam.getOutputQueue("right")
    #q_depth = cam.getOutputQueue("depth")
        
    
    rgbframe = None
    # monoLeftFrame = None
    # monoRightFrame = None
    # depthFrame = None
    detections = []
    input_tensor_rgb = None
    # input_tensor_depth = None
    preparedRgb = None

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

    frams_rgb = []
    # frams_depth = []
    buffer_rgb_ready = False
    # buffer_depth_ready = False
    previous_time = time.time()

    
    while(True):
        current_time = time.time()
        time_difference = current_time - previous_time
        previous_time = current_time
       
        in_rgb = q_rgb.tryGet()

        # in_left = q_left.tryGet()
        # in_right = q_right.tryGet()
        # in_depth = q_depth.tryGet()
        
    
        
        
        def transform_params( resize=(320, 240), crop_size=224, flip=0.5):
            left, top = (resize[0] - crop_size) // 2, (resize[1] - crop_size) // 2
            is_flip = False
            return (left, top, left + crop_size, top + crop_size), is_flip
        
        resize = (320, 240)  # default | (256, 256) may be helpful
        crop_rect, is_flip = transform_params(resize=resize, flip=1.0)
        left, top, right, bottom = crop_rect
        def transform(img):
            img = cv2.resize(img, resize)
       
            img = img[top:bottom, left:right]

            return cv2.resize(img,(112, 112)).astype(np.float32)

        start = time.time()
        # asynchronous
        if in_rgb is not None:
        
            rgbframe = in_rgb.getCvFrame()
            if not buffer_rgb_ready:
                preparedRgb = transform(rgbframe)
                
                preparedRgbTensor = tensorTransformer(preparedRgb).view(3, 112, 112, 1)
                frams_rgb.append(preparedRgbTensor)
                if len(frams_rgb) == sample_duration:
                    input_tensor_rgb_no_perm = torch.cat(frams_rgb, dim=3).type(torch.FloatTensor)
                    input_tensor_rgb_perm = input_tensor_rgb_no_perm.permute(0, 3, 1, 2)
                    input_tensor_rgb = input_tensor_rgb_perm.unsqueeze(0)
          
                    buffer_rgb_ready = True
              
         

        # if in_left is not None:
        #     monoLeftFrame = in_left.getCvFrame()

        # if in_right is not None:
        #     monoRightFrame = in_right.getCvFrame()

        # if in_depth is not None:
        #     depthFrame = in_depth.getCvFrame()
        #     if not buffer_depth_ready:
        #         preparedDepth = transform(depthFrame)
        #         bgr_image = cv2.cvtColor(preparedDepth,cv2.COLOR_GRAY2BGR)
                
        #         preparedDepthTensor = tensorTransformer(bgr_image).view(3, 112, 112, 1)
        #         frams_depth.append(preparedDepthTensor)  
        #         if len(frams_depth) == sample_duration:
                    
        #             input_tensor_depth_no_perm = torch.cat(frams_depth, dim=3).type(torch.FloatTensor)
                    
        #             input_tensor_depth_perm = input_tensor_depth_no_perm.permute(0, 3, 1, 2)
        
        #             input_tensor_depth = input_tensor_depth_perm.unsqueeze(0)
    
        #             buffer_depth_ready = True           
                    

        if  buffer_rgb_ready:
            print("---------------------------------")
            
           
            with torch.no_grad():  # no bp & gradients compute during inference
               
                output = model(input_tensor_rgb)  # output是logits tensor
                #print(f"output张量的尺寸：{output.shape}")  # torch.Size([1, 249]),1 means batch_size = 1, i.e. process one inupt(32 frames stack), 249 means thec nums of classes
               

                
           
                
                predict_result = torch.argmax(output,dim=1)
                
                predict_tag = torch.argmax(output,dim=1)
                probs = [float(output[i,p]) for i,p in enumerate(predict_tag)]
                #print(f"probs scores是：{probs}")
                
                #
                
                print(f"录像的预测结果tag:{predict_tag.item()}")
              
                cv2.putText(rgbframe, f"Prediction: {predict_tag.item()}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                # open a window to show the frame
                cv2.imshow('frame', rgbframe)
                #depthFrame_resized = cv2.resize(depthFrame, (320, 240))
                #colored = oakd.color_depth_image(depthFrame_resized)
                #cv2.imshow("depth preview", colored)
               
            
            dt = time.time() - start
            print("dt: %s" % (dt))
            
            buffer_depth_ready = False
            buffer_rgb_ready = False
            frams_rgb.clear()
            #frams_depth.clear()

            
        
        # # press 'q' quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        

    cv2.destroyAllWindows()