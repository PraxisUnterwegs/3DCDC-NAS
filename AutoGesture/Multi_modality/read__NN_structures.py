import torch
from torchviz import make_dot

from models.AutoGesture_RGBD_searched_12layers_DiffChannels import AutoGesture_RGBD_12layers
from utils.config import Config
from train_AutoGesture_CDC_RGBD_sgd_12layers import parse, Module

import os
#os.environ["PATH"] += os.pathsep + 'path/to/graphviz/bin'


args = Config(parse())

device = torch.device('cpu')  # TODO
#device = torch.device('cuda:0')  # TODO
model = Module(args)
#model = AutoGesture_RGBD_12layers(init_channels8, init_channels16, init_channels32, num_classes, layers)  # 实例化模型架构
checkpoint = torch.load('/home/jzf/Tasks__Gestures_Classification/3DCDC-NAS/AutoGesture/checkpoints/epoch26-MK-valid_0.6004-test_0.6224.pth',
                        map_location=device)
model.load_state_dict(checkpoint)  # 加载模型参数
model.eval() 

x_rgb = torch.randn(2, 3, 32, 112, 112)
x_depth = torch.randn(2, 3, 32, 112, 112)

y = model(x_rgb,x_depth)

dot = make_dot(y, params=dict(list(model.named_parameters()) + [('input_rgb', x_rgb), ('input_depth', x_depth)]),show_attrs=True, show_saved=True)
#dot.render("model_structure_graph")  # 将图形保存为文件，默认格式为 PDF
dot.view()  # 生成文件
# 指定文件生成的文件夹
#MyConvNetVis.directory = "data"
