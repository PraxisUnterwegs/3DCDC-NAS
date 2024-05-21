import numpy as np
import skimage.exposure
import cv2
from pathlib import Path
import blobconverter
import cv2
import depthai
import numpy as np



def init():
    # Pipeline tells DepthAI what operations to perform when running - you define all of the resources used and flows here
    pipeline = depthai.Pipeline()

    # First, we want the Color camera as the output
    cam_rgb = pipeline.createColorCamera()
    #cam_rgb.setPreviewSize(300, 300)  # 300x300 will be the preview frame size, available as 'preview' output of the node
    cam_rgb.setPreviewSize(320, 240)  
    cam_rgb.setInterleaved(False)
    cam_rgb.setFps(10)


    ####################################################################
    left = pipeline.createMonoCamera()
    left.setCamera("left")
    left.setResolution(depthai.MonoCameraProperties.SensorResolution.THE_400_P)
    left.setFps(10)
    right = pipeline.createMonoCamera()
    right.setCamera("right")
    right.setResolution(depthai.MonoCameraProperties.SensorResolution.THE_400_P)
    right.setFps(10)
    #######################################
    cam_stereo = pipeline.createStereoDepth()
    cam_stereo.initialConfig.setConfidenceThreshold(150)
    # cam_stereo.setDefaultProfilePreset(depthai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    cam_stereo.initialConfig.setMedianFilter(depthai.MedianFilter.KERNEL_3x3)
    # cam_stereo.initialConfig.setLeftRightCheckThreshold(10)
    # Better handling for occlusions:
    cam_stereo.setLeftRightCheck(True)
    # Closer-in minimum depth, disparity range is doubled:
    cam_stereo.setExtendedDisparity(False)
    # Better accuracy for longer distance, fractional disparity 32-levels:
    cam_stereo.setSubpixel(False)
    
    ########################################
    left.out.link(cam_stereo.left)
    right.out.link(cam_stereo.right)
    #####################################################################

    # Next, we want a neural network that will produce the detections
    #detection_nn = pipeline.createMobileNetDetectionNetwork()
    # Blob is the Neural Network file, compiled for MyriadX. It contains both the definition and weights of the model
    # We're using a blobconverter tool to retreive the MobileNetSSD blob automatically from OpenVINO Model Zoo
    #detection_nn.setBlobPath(blobconverter.from_zoo(name='mobilenet-ssd', shaves=6))
    # Next, we filter out the detections that are below a confidence threshold. Confidence can be anywhere between <0..1>
    #detection_nn.setConfidenceThreshold(0.5)
    # Next, we link the camera 'preview' output to the neural network detection input, so that it can produce detections
    #cam_rgb.preview.link(detection_nn.input)

    # XLinkOut is a "way out" from the device. Any data you want to transfer to host need to be send via XLink
    xout_rgb = pipeline.createXLinkOut()
    # For the rgb camera output, we want the XLink stream to be named "rgb"
    xout_rgb.setStreamName("rgb")
    # Linking camera preview to XLink input, so that the frames will be sent to host
    #cam_rgb.video.link(xout_rgb.input)  # full size frame
    cam_rgb.preview.link(xout_rgb.input)

    # The same XLinkOut mechanism will be used to receive nn results
    #xout_nn = pipeline.createXLinkOut()
    #xout_nn.setStreamName("nn")
    #detection_nn.out.link(xout_nn.input)
    ####
    xoutLeft = pipeline.createXLinkOut()
    xoutRight = pipeline.createXLinkOut()
    xoutDepth = pipeline.createXLinkOut()
    xoutLeft.setStreamName("left")
    xoutRight.setStreamName("right")
    xoutDepth.setStreamName("depth")
    cam_stereo.syncedLeft.link(xoutLeft.input)
    cam_stereo.syncedRight.link(xoutRight.input)
    cam_stereo.depth.link(xoutDepth.input)
    cam_stereo.setOutputSize(320, 240)

    return pipeline


def color_depth_image(img):
    # min_depth = np.min(img)
    # max_depth = np.max(img)
    # normalized_depth_image = (img - min_depth) / (max_depth - min_depth)     
    # Normalize depth values to range [0, 1]
    image_float = img.astype(np.float32)
    normalized_depth_image = cv2.normalize(image_float, None, 0.0, 1.0, cv2.NORM_MINMAX)
    # Apply colormap (Jet colormap is commonly used for depth images)
    artificialScalingFactor = 8.0
    colored_depth_image = cv2.applyColorMap(np.uint8(normalized_depth_image * artificialScalingFactor * 255), cv2.COLORMAP_JET)
    return colored_depth_image




