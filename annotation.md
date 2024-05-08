# Single_modality sub-project
## 执行顺序
visulaizer.py
Videodatasets.py
config.py
AutoGesture_searched.py
train_AutoGesture_3DCDC.py

具体来说，{
    Videodatasets.py：是train_AutoGesture_3DCDC的一个依赖，自身并不执行
    visulaizer.py:是train_AutoGesture_3DCDC的一个依赖，自身并不执行
    config.py:是train_AutoGesture_3DCDC的一个依赖，自身并不执行
    AutoGesture_searched.py:是train_AutoGesture_3DCDC的一个依赖，自身并不执行
    

}


## train_AutoGesture_3DCDC.py
if args.type == 'M':
        modality = 'rgb'
    elif args.type == 'K':
        modality = 'depth'
    elif args.type == 'F':
        modality = 'flow'

train_loader👉GetData()👉train_dataloader

train_dataloader👉train_loader👉training()👉train()👉





# Multi_modality sub-project






# Fusion sub-project