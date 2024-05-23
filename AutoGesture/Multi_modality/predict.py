import cv2 #
import matplotlib.pyplot as plt 
import numpy as np   
import torch
import os
import os.path as osp
import torch.nn as nn
from PIL import Image
from torchvision import datasets, models, transforms
import time

#################################################################################################
### Tensor image normalization stuff
#################################################################################################
class TensorImageNormalization(object):
    """
        same as mxnet, normalize into [-1, 1]
        image = (image - 127.5)/128
    """
    def __call__(self, Image):
        new_video_x = (Image - 127.5) / 128
        return new_video_x
tensorTransformer = transforms.Compose([TensorImageNormalization(), transforms.ToTensor()])
def normalizeImageTensor(img):
    return tensorTransformer(img)

#################################################################################################
### Model input transformation (cropping, resizing)
#################################################################################################
resize = (320, 240)  # default | (256, 256) may be helpful
def image_crop_rect( resize=(320, 240), crop_size=224, flip=0.5):
    left, top = (resize[0] - crop_size) // 2, (resize[1] - crop_size) // 2
    is_flip = False
    return (left, top, left + crop_size, top + crop_size), is_flip
crop_rect, is_flip = image_crop_rect(resize=resize, flip=1.0)
left, top, right, bottom = crop_rect
def transform_image(img):
    img = cv2.resize(img, resize)
    img = img[top:bottom, left:right]
    return cv2.resize(img,(112, 112)).astype(np.float32)

#################################################################################################
### Main
#################################################################################################
def main():
    from utils.config import Config
    from train_AutoGesture_CDC_RGBD_sgd_12layers import parse, Module
    import depthai
    import oak_d_driver as oakd

    args = Config(parse())
    device = torch.device('cpu')  # TODO
    #device = torch.device('cuda:0')  # TODO
    model = Module(args, device=device)
    #model = AutoGesture_RGBD_12layers(init_channels8, init_channels16, init_channels32, num_classes, layers)  # 实例化模型架构
    checkpoint = torch.load(args.init_model, map_location=device)
    model.load_state_dict(checkpoint)  # 加载模型参数
    model.eval() 

    pipeline = oakd.init()

    # Pipeline is now finished, and we need to find an available device to run our pipeline
    # we are using context manager here that will dispose the device after we stop using it
    with depthai.Device(pipeline) as cam:
        # From this point, the Device will be in "running" mode and will start sending data via XLink
        # To consume the device results, we get two output queues from the device, with stream names we assigned earlier
        q_rgb = cam.getOutputQueue("rgb")
        #q_nn = cam.getOutputQueue("nn")
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
        #buffer_rgb = RingBuffer(60)
        #buffer_depth = RingBuffer(60)
        # buffer_rgb = RingBuffer(32)
        # buffer_depth = RingBuffer(32)

        frams_rgb = []
        frams_depth = []
        buffer_rgb_ready = False
        buffer_depth_ready = False
        previous_time = time.time()

        
        while(True):
            # read frame
            ##############################################################################
            # we try to fetch the data from nn/rgb queues. tryGet will return either the data packet or None if there isn't any
            in_rgb = q_rgb.tryGet()
            #in_nn = q_nn.tryGet()
            in_left = q_left.tryGet()
            in_right = q_right.tryGet()
            in_depth = q_depth.tryGet()
            
            # asynchronous
            if in_rgb is not None:
                # If the packet from RGB camera is present, we're retrieving the frame in OpenCV format using getCvFrame
                rgbframe = in_rgb.getCvFrame()
                if not buffer_rgb_ready:
                    preparedRgb = transform_image(rgbframe)
                    #print(f"rgbFrame_resized的形状: {preparedRgb.shape}")
                    preparedRgbTensor = normalizeImageTensor(preparedRgb).view(3, 112, 112, 1)
                    frams_rgb.append(preparedRgbTensor)
                    if len(frams_rgb) == args.sample_duration:
                        # buffer_rgb = [frams_rgb[i] for i in extract_frames_id_list]
                        input_tensor_rgb_no_perm = torch.cat(frams_rgb, dim=3).type(torch.FloatTensor)
                        #print(f"input_tensor_rgb_no_perm的输出结果：{input_tensor_rgb_no_perm.shape}")
                        input_tensor_rgb_perm = input_tensor_rgb_no_perm.permute(0, 3, 1, 2)
                        #print(f"input_tensor_rgb_perm的输出结果：{input_tensor_rgb_perm.shape}")
                        input_tensor_rgb = input_tensor_rgb_perm.unsqueeze(0)
                        #input_tensor_rgb = input_tensor_rgb.permute(0, 1, 4, 2, 3)
                        #print(f"看看input_rgb_depth的输出结果：{input_tensor_rgb.shape}")
                        buffer_rgb_ready = True
                
                #print(f"rgbframe的形状: {rgbframe.shape}")

            if in_left is not None:
                monoLeftFrame = in_left.getCvFrame()

            if in_right is not None:
                monoRightFrame = in_right.getCvFrame()

            if in_depth is not None:
                depthFrame = in_depth.getCvFrame()
                if not buffer_depth_ready:
                    preparedDepth = transform_image(depthFrame)
                    bgr_image = cv2.cvtColor(preparedDepth,cv2.COLOR_GRAY2BGR)
                    #print(f"depthFrame_resized的形状: {bgr_image.shape}")
                    preparedDepthTensor = tensorTransformer(bgr_image).view(3, 112, 112, 1)
                    frams_depth.append(preparedDepthTensor)  
                    if len(frams_depth) == args.sample_duration:
                        # buffer_depth = [frams_depth[i] for i in extract_frames_id_list]
                        input_tensor_depth_no_perm = torch.cat(frams_depth, dim=3).type(torch.FloatTensor)
                        #print(f"input_tensor_depth_no_perm的输出结果：{input_tensor_depth_no_perm.shape}")
                        input_tensor_depth_perm = input_tensor_depth_no_perm.permute(0, 3, 1, 2)
                        #print(f"input_tensor_depth_perm的输出结果：{input_tensor_depth_perm.shape}")
                        input_tensor_depth = input_tensor_depth_perm.unsqueeze(0)
                        #input_tensor_depth = input_tensor_depth.permute(0, 1, 4, 2, 3)
                        #print(f"看看input_tensor_depth的输出结果：{input_tensor_depth.shape}")
                        buffer_depth_ready = True           

            if buffer_depth_ready and buffer_rgb_ready:
                print("---------------------------------")
                start = time.time()
                # if input_tensor_rgb is not None and input_tensor_depth is not None:
                #     # inference
                with torch.no_grad():  # no bp & gradients compute during inference
                    output = model(input_tensor_rgb, input_tensor_depth)  # output是logits tensor
                    predict_tag = torch.argmax(output,dim=1)
                    probs = [float(output[i,p]) for i,p in enumerate(predict_tag)]
                    print(f"probs scores是：{probs}")
                    print(f"录像的预测结果tag:{predict_tag.item()}")
                    cv2.putText(rgbframe, f"Prediction: {predict_tag.item()}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    # open a window to show the frame
                    cv2.imshow('frame', rgbframe)
                    colored = oakd.color_depth_image(depthFrame)
                    cv2.imshow("depth preview", depthFrame)
                    
                dt = time.time() - start
                print("dt: %s" % (dt))
                
                buffer_depth_ready = False
                buffer_rgb_ready = False
                frams_rgb.clear()
                frams_depth.clear()

            #     # real-time rendering display the classification result (244:shake hands) on the screen
                
            
            # # press 'q' quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # if preparedRgb is not None:
            #     cv2.imshow('frame', preparedRgb)
        # release the cam object
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()