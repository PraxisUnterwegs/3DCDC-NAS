import numpy as np
import skimage.exposure
import cv2
from pathlib import Path
import blobconverter
import cv2
import depthai
import numpy as np


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


# Pipeline tells DepthAI what operations to perform when running - you define all of the resources used and flows here
pipeline = depthai.Pipeline()

# First, we want the Color camera as the output
cam_rgb = pipeline.createColorCamera()

cam_rgb.setPreviewSize(300, 300)  # 300x300 will be the preview frame size, available as 'preview' output of the node
cam_rgb.setInterleaved(False)


####################################################################
left = pipeline.createMonoCamera()
left.setCamera("left")
left.setResolution(depthai.MonoCameraProperties.SensorResolution.THE_800_P)
right = pipeline.createMonoCamera()
right.setCamera("right")
right.setResolution(depthai.MonoCameraProperties.SensorResolution.THE_800_P)
#######################################
cam_stereo = pipeline.createStereoDepth()
cam_stereo.initialConfig.setConfidenceThreshold(150)
# cam_stereo.setDefaultProfilePreset(depthai.node.StereoDepth.PresetMode.HIGH_DENSITY)
cam_stereo.initialConfig.setMedianFilter(depthai.MedianFilter.KERNEL_7x7)
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
detection_nn = pipeline.createMobileNetDetectionNetwork()
# Blob is the Neural Network file, compiled for MyriadX. It contains both the definition and weights of the model
# We're using a blobconverter tool to retreive the MobileNetSSD blob automatically from OpenVINO Model Zoo
detection_nn.setBlobPath(blobconverter.from_zoo(name='mobilenet-ssd', shaves=6))
# Next, we filter out the detections that are below a confidence threshold. Confidence can be anywhere between <0..1>
detection_nn.setConfidenceThreshold(0.5)
# Next, we link the camera 'preview' output to the neural network detection input, so that it can produce detections
cam_rgb.preview.link(detection_nn.input)

# XLinkOut is a "way out" from the device. Any data you want to transfer to host need to be send via XLink
xout_rgb = pipeline.createXLinkOut()
# For the rgb camera output, we want the XLink stream to be named "rgb"
xout_rgb.setStreamName("rgb")
# Linking camera preview to XLink input, so that the frames will be sent to host
cam_rgb.video.link(xout_rgb.input)

# The same XLinkOut mechanism will be used to receive nn results
xout_nn = pipeline.createXLinkOut()
xout_nn.setStreamName("nn")
detection_nn.out.link(xout_nn.input)
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

# Pipeline is now finished, and we need to find an available device to run our pipeline
# we are using context manager here that will dispose the device after we stop using it
with depthai.Device(pipeline) as device:
    # From this point, the Device will be in "running" mode and will start sending data via XLink

    # To consume the device results, we get two output queues from the device, with stream names we assigned earlier
    q_rgb = device.getOutputQueue("rgb")
    q_nn = device.getOutputQueue("nn")
    q_left = device.getOutputQueue("left")
    q_right = device.getOutputQueue("right")
    q_depth = device.getOutputQueue("depth")

    # Here, some of the default values are defined. Frame will be an image from "rgb" stream, detections will contain nn results
    frame = None
    monoLeftFrame = None
    monoRightFrame = None
    depthFrame = None
    detections = []

    # Since the detections returned by nn have values from <0..1> range, they need to be multiplied by frame width/height to
    # receive the actual position of the bounding box on the image
    def frameNorm(frame, bbox):
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)


    # Main host-side application loop
    while True:
        # we try to fetch the data from nn/rgb queues. tryGet will return either the data packet or None if there isn't any
        in_rgb = q_rgb.tryGet()
        in_nn = q_nn.tryGet()
        in_left = q_left.tryGet()
        in_right = q_right.tryGet()
        in_depth = q_depth.tryGet()

        if in_rgb is not None:
            # If the packet from RGB camera is present, we're retrieving the frame in OpenCV format using getCvFrame
            frame = in_rgb.getCvFrame()

        if in_nn is not None:
            # when data from nn is received, we take the detections array that contains mobilenet-ssd results
            detections = in_nn.detections

        if in_left is not None:
            monoLeftFrame = in_left.getCvFrame()

        if in_right is not None:
            monoRightFrame = in_right.getCvFrame()

        if in_depth is not None:
            depthFrame = in_depth.getCvFrame()

        if frame is not None:
            for detection in detections:
                # for each bounding box, we first normalize it to match the frame size
                bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
                # and then draw a rectangle on the frame to show the actual result
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
            # After all the drawing is finished, we show the frame on the screen
            cv2.imshow("rgb preview", frame)
            cv2.moveWindow('rgb preview', 0, 0)  # Adjust the coordinates as needed
        if monoLeftFrame is not None:
            pass
            # cv2.imshow("left preview", monoLeftFrame)
        if monoRightFrame is not None:
            pass
            # cv2.imshow("right preview", monoRightFrame)
        if depthFrame is not None:
            
            # window_size = 3
            # kernel = np.ones((window_size, window_size), np.float32) / (window_size * window_size)
            # filtered_image = cv2.filter2D(depthFrame, -1, kernel)
            # colored = color_depth_image(filtered_image)
            colored = color_depth_image(depthFrame)
            cv2.imshow("depth preview", colored)
            cv2.moveWindow('depth preview', 600, 0)  # Adjust the coordinates as needed

        # at any time, you can press "q" and exit the main loop, therefore exiting the program itself
        if cv2.waitKey(1) == ord('q'):
            break