# AutoGesture with 3DCDC
Pytorch code for the TIP paper [**"Searching Multi-Rate and Multi-Modal Temporal Enhanced Networks for Gesture Recognition"**  ](https://arxiv.org/pdf/2008.09412.pdf)


### USAGE ROS

```
ros2 launch AutoGesture inference.launch.py checkpoint:=$HOME/3DCDC-NAS/checkpoints/epoch26-MK-valid_0.6004-test_0.6224.pth
```


### Welcome to plug and play 3DCDC in your networks
```
# -------- Vanilla ---------#
nn.Conv3d(3, 64, kernel_size=3, padding=1)

# -------- 3DCDC ---------#
from 3DCDC import CDC_ST, CDC_T, CDC_TR
CDC_ST(3, 64, kernel_size=3, padding=1, theta=0.6)
CDC_T(3, 64, kernel_size=3, padding=1, theta=0.6)
CDC_TR(3, 64, kernel_size=3, padding=1, theta=0.3)
```


### Citation

If you find our project useful in your research, please consider citing:

```
@article{yu2021searching,
  title={Searching Multi-Rate and Multi-Modal Temporal Enhanced Networks for Gesture Recognition},
  author={Yu, Zitong and Zhou, Benjia and Wan, Jun and Wang, Pichao and Chen, Haoyu and Liu, Xin and Li, Stan Z and Zhao, Guoying},
  journal={IEEE Transactions on Image Processing (TIP)},
  year={2021}
}
```



### Pretrained model on IsoGD
You can download the checkpoints from [google drive](https://drive.google.com/drive/folders/1lFcIXJO7LBZMytlpWgM_r4YeSwO5VLix?usp=sharing)


### Visualization


<div align=center>
<img src="https://github.com/ZitongYu/3DCDC-NAS/blob/master/supplementary%20materials/Searched.png"><br>
Figure 1: The searched architecture from (a) the first stage NAS, and (b) the second stage NAS. The three rows in (a) represent the searched cell structure in the low, mid, and high frame branches, respectively.
</div>


<div align=center>
<img src="https://github.com/ZitongYu/3DCDC-NAS/blob/master/supplementary%20materials/Visualization.jpg"><br>
Figure 2: Features visualization from C3D assembled with varied convolutions on the IsoGD dataset. With (a) RGB and (b) Depth modality inputs, the four rows represent the neural activation with 3D vanilla convolution, 3D-CDC-ST, 3D-CDC-T, and 3D-CDC-TR, respectively.
</div>





