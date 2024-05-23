import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from message_filters import ApproximateTimeSynchronizer, Subscriber

import cv2
import cv_bridge

import matplotlib.pyplot as plt 
import numpy as np   
import torch
import threading
import time
import copy


### ros packaged imports
from AutoGesture.Multi_modality.utils.config import Config
from AutoGesture.Multi_modality.predict import normalizeImageTensor
from AutoGesture.Multi_modality.predict import transform_image
from AutoGesture.Multi_modality.oak_d_driver import color_depth_image
from AutoGesture_CDC_RGBD_sgd_12layers import Module

#############################################################################################
class InferenceThread(threading.Thread):
    def __init__(self, model, args):
        super().__init__()
        self.model = model
        self.args = args
        self.daemon = True
        self.output = None
        self.clearInputs()
    def run(self):
        while self.running:
            if self.input_image_buf_1 and self.input_image_buf_2:
                ### do inference stuff
                start = time.time()
                with torch.no_grad(): ### no bp & gradients compute during inference
                    output = self.model(self.input_image_buf_1, self.input_image_buf_2)
                    predict_tag = torch.argmax(output,dim=1)
                    probs = [float(output[i,p]) for i,p in enumerate(predict_tag)]
                    # print(f"probs scores：{probs}")
                    # print(f"label:{predict_tag.item()}")
                    self.output = (probs, predict_tag.item())
                dt = time.time() - start
                print("dt: %s" % (dt))
            else:
                ### safety delay for thread to not run amok
                time.sleep(0.010)
    def stop(self):
        self.running = False
    def setInputs(self, inputBuf1, inputBuf2):
        self.input_image_buf_1 = copy.deepcopy(inputBuf1)
        self.input_image_buf_2 = copy.deepcopy(inputBuf2)
    def getOutput(self):
        return self.output
    def clearInputs(self):
        self.input_image_buf_1 = None
        self.input_image_buf_2 = None

#############################################################################################
        
class InferenceRosNode(Node):
    def __init__(self):
        super().__init__('inference_ros_node')
        #################################################
        ### Declare and get parameters
        self.declare_parameter('model_config', 'config.yml')
        self.declare_parameter('rgb_topic', '/rgb/image_raw')
        self.declare_parameter('depth_topic', '/depth/image_raw')   
        configPath = self.get_parameter('model_config').get_parameter_value().string_value      
        rgb_topic = self.get_parameter('rgb_topic').get_parameter_value().string_value
        depth_topic = self.get_parameter('depth_topic').get_parameter_value().string_value
        #################################################
        ### Create publishers/subscribers for the two image topics
        self.rgb_pub = self.create_publisher(Image, '~/rgb/labeled', 1)
        self.depth_pub = self.create_publisher(Image, '~/depth/colored', 1)
        self.rgb_sub = Subscriber(self, Image, rgb_topic)
        self.depth_sub = Subscriber(self, Image, depth_topic)
        ### Create the ApproximateTimeSynchronizer
        self.ts = ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub],
            queue_size=10,
            slop=0.1
        )
        ### Register the callback to be called when both images are received
        self.ts.registerCallback(self.image_callback)
        #################################################
        ### model stuff
        class PseudoArgs:
            def __init__(self, config):
                self.config = config
        self.args = Config(PseudoArgs(configPath))
        device = torch.device('cpu')
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
        model = Module(self.args, device=device)
        checkpoint = torch.load(self.args.init_model, map_location=device)
        model.load_state_dict(checkpoint)
        model.eval()
        #################################################
        ### inference thread
        self.inference = InferenceThread(model, self.args)
        #################################################
        ### misc
        self.bridge = cv_bridge.CvBridge()
        self.rgb_buf = []
        self.depth_buf = []
        
    def image_callback(self, rgb, depth):
        self.get_logger().debug('Synchronized images received')
        preparedRgb = transform_image(rgb)
        preparedDepth = transform_image(depth)
        depth_bgr = cv2.cvtColor(preparedDepth,cv2.COLOR_GRAY2BGR)
        preparedRgbTensor = normalizeImageTensor(preparedRgb).view(3, 112, 112, 1)
        preparedDepthTensor = normalizeImageTensor(depth_bgr).view(3, 112, 112, 1)
        self.rgb_buf.append(preparedRgbTensor)
        self.depth_buf.append(preparedDepthTensor)
        if len(self.rgb_buf) == self.args.sample_duration and len(self.depth_buf) == self.args.sample_duration:
            self.inference.setInputs(self.rgb_buf, self.depth_buf)
        ### grab last available output from model (could be old)
        probs, label = self.inference.getOutput()
        ### publish output images containing last available model output
        cv2.putText(rgb, f"Prediction: {label} > {probs}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        coloredDepth = color_depth_image(depth)
        rgb_img_msg = self.bridge.cv2_to_imgmsg(rgb, "bgr8")
        depth_img_msg = self.bridge.cv2_to_imgmsg(coloredDepth, "bgr8")
        self.rgb_pub.publish(rgb_img_msg)
        self.depth_pub.publish(depth_img_msg)

    def destroy_node(self):
        self.inference.stop()
        super().destroy_node()
#############################################################################################
def main(args=None):
    rclpy.init(args=args)
    node = InferenceRosNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
