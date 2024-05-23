import argparse
import time
import os
import numpy as np
from tqdm import tqdm
import random
# import pprint

from collections import OrderedDict
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
import torch.backends.cudnn as cudnn
import torchvision

# from network import network
# from network import NetworkT_CDC as network
# from network.NetworkT_CDC import T_CDC_Avg, T_CDC, ST_CDC

from utils.visualizer import Visualizer
from Videodatasets_RGBD import Videodatasets_RGBD
from utils.config import Config

from collections import namedtuple


from AutoGesture_CDC_RGBD_sgd_12layers import Module

# GetData的目的就是依据xxx_lst.txt文件里的记录来读取Dataset文件夹下的对应数据集，转化为DataLoader对象，然后投入训练
# The purpose of GetData is to read the corresponding data set in the Dataset folder based on the records in the xxx_lst.txt file, and then put it into training
def GetData(args):
    print('Start load Data...')
    modality1 = 'rgb'
    modality2 = 'depth'

    train_dataset = Videodatasets_RGBD(args.data_dir_root + '/train',
                                args.dataset_splits + '/{0}_train_lst.txt'.format(modality2), modality1,
                                args.dataset_splits + '/{0}_train_lst.txt'.format(modality2), modality2,
                                args.sample_duration, phase='train')
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                  pin_memory=True)

    valid_dataset = Videodatasets_RGBD(args.data_dir_root + '/valid',
                                args.dataset_splits + '/{0}_valid_lst.txt'.format(modality2), modality1,
                                args.dataset_splits + '/{0}_valid_lst.txt'.format(modality2), modality2,
                                args.sample_duration, phase='valid')
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.testing_batch_size, shuffle=False,
                                  num_workers=args.num_workers,
                                  pin_memory=True)
    '''
    Videodatasets_RGBD返回的张量(test_dataset)是4维的，应该是[3, num_sl, 112, 112] , num_sl表示test/003/M_00600文件下.jpg文件的个数
    DataLoader返回的张量(test_dataloader)是5维的，应该是[2, 3, 32, 112, 112]，2应该是batch_size
    
    '''
    try:
        test_dataset = Videodatasets_RGBD(args.data_dir_root + '/test',
                                   args.dataset_splits + '/{0}_test_lst.txt'.format(modality2), modality1,
                                   args.dataset_splits + '/{0}_test_lst.txt'.format(modality2), modality2,
                                   args.sample_duration, phase='test')
        test_dataloader = DataLoader(test_dataset, batch_size=args.testing_batch_size, shuffle=False,
                                     num_workers=args.num_workers,
                                     pin_memory=True)
        print('Load <IsoGD>  Dataset!')
        print('=' * 20)
        return train_dataloader, valid_dataloader, test_dataloader
    except:
        print('Load <INVIDIA>  Dataset!')
        print('=' * 20)
        return train_dataloader, valid_dataloader, None


class train_val:
    def __init__(self, model, args, vis, train_loader, val_loader, test_loader):
        self.model = model
        self.args = args
        self.vis = vis
        self.step = 0
        self.lr = args.learning_rate
        self.dlr = args.learning_rate

        try:
            self.best_val_acc = float(args.resume.split('-')[-2][6:]) if args.resume and not args.pretrain else 0.4
        except:
            self.best_val_acc = 0.4
        self.optimizer, self.lr_scheduler, self.criterion1 = self.LoadOptimizer()

        if args.mode == 'train':
            self.training(train_loader, val_loader, test_loader)
        elif args.mode == 'valid':
            valid_acc = self.valid(1, dataloader=val_loader)
            print('valid_acc:{}'.format(valid_acc))
        elif args.mode == 'test':
            assert test_loader, "test_loader is None"
            test_acc = self.test(1, dataloader=test_loader)
            print('test_acc:{}'.format(test_acc))

        else:
            raise Exception('Error in phase!')

    def LoadOptimizer(self):
        criterion1 = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.learning_rate, momentum=self.args.momentum,
                                    weight_decay=self.args.weight_decay)
        # lr_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, verbose=True, threshold=0.0001,
        #                                  threshold_mode='rel', cooldown=3, min_lr=0.00001, eps=1e-08)
        lr_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, threshold=0.0001,
                                         threshold_mode='rel', cooldown=3, min_lr=0.00001, eps=1e-08)
        lr_scheduler.verbose = True
        return optimizer, lr_scheduler, criterion1

    def update_lr(self, optimizer, step):
        df = 0.7
        ds = 40000.0
        in_dlr = self.dlr
        self.dlr = self.lr * np.power(df, step / ds)

        # Dynamic Modification Learning Rate
        for g in optimizer.param_groups:
            in_lr = g['lr']
            g['lr'] = self.dlr

        self.lr = self.lr * (in_lr / in_dlr)
        if in_lr / in_dlr == 0.1:
            self.step = 0

    # training是为了进行选择，是进行train还是valid，还是test阶段，至于具体的实现会分散在self.train(),self.valid()...中，这些函数在training中会被调用
    def training(self, train_loader, val_loader, test_loader):
        print('Start training...')
        for epoch in range(self.args.init_epochs, self.args.max_epochs):
            if epoch == 0:  # Warm up
                for g in self.optimizer.param_groups:
                    g['lr'] = 0.00001
            elif epoch == 3:
                for g in self.optimizer.param_groups:
                    g['lr'] = self.lr
            '''
            Train:
            '''
            self.train(epoch, dataloader=train_loader)
            torch.save(self.model.state_dict(), "middle.pth")
            if epoch >= 3:
                '''
                valid
                '''
                valid_acc = self.valid(epoch, dataloader=val_loader)
                self.lr_scheduler.step(valid_acc)
                self.step += 1
                '''
                Test
                '''
                if test_loader:
                    test_acc = self.test(epoch, dataloader=test_loader)
                else:
                    test_acc = 0.0

                '''
                Save model
                '''
                model_save_filename = './Checkpoints/model_{}_{}/epoch{}-{}-valid_{}-test_{}.pth'.format(
                    self.args.res_layer, self.args.sample_duration, epoch, self.args.type,
                    round(valid_acc, 4), round(test_acc, 4))
                if not os.path.exists(os.path.split(model_save_filename)[0]): os.makedirs(
                    os.path.split(model_save_filename)[0])
                if valid_acc > self.best_val_acc:  # 如果valid_acc超过0.4 的话，这个模型就会被保存。
                    torch.save(self.model.state_dict(), model_save_filename)
                    with open("log/logfile_{}_{}_{}.txt".format(args.res_layer, args.sample_duration, args.type),
                              'a') as opened_file:
                        opened_file.write(str(time.strftime('%m.%d %H:%M:%S', time.localtime(time.time()))) + ': ' +
                                          os.path.split(model_save_filename)[1] + '\n')
                self.best_val_acc = max(valid_acc, self.best_val_acc)
        print('Training finished in time: @{}'.format(time.strftime('%m.%d %H:%M:%S', time.localtime(time.time()))))

    def train(self, epoch, dataloader):
        self.model.train()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        Avg_loss = AverageMeter()

        end = time.time()
        for i, (d, l, d2) in tqdm(enumerate(dataloader)):
            print(f"d:{d.shape}, l:{l.shape}, d2:{d2.shape}")
            outputs = self.model(d.cuda(), d2.cuda())
            data_time.update(time.time() - end)

            # Forward propagation
            loss = self.criterion1(outputs, l.cuda())

            Avg_loss.update(loss.item(), d.size(0))

            # backward propagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # lr decayed exponentially
            if epoch >= 3:
                self.update_lr(self.optimizer, len(dataloader.dataset) / self.args.batch_size * self.step + i)

            if (i + 1) % self.args.print_freq == 0:
                print(
                    ' Training [%2d/%2d, %4d/%4d] \t Loss: %.4f(%.4f) \t [datatime: %.3f] \t [batchtime: %.3f] \t @%s' % (
                        epoch, self.args.max_epochs, i, len(dataloader.dataset) / self.args.batch_size, loss.item(),
                        Avg_loss.avg,
                        data_time.avg, batch_time.avg, time.strftime('%m.%d %H:%M:%S', time.localtime(time.time()))))

                self.vis.plot_many({'loss': Avg_loss.avg}, 'Train-' + self.args.type)
                self.vis.log(
                    'LR[{m}]: {lr}'.format(m=self.args.type, lr=[g['lr'] for g in self.optimizer.param_groups]))

    def valid(self, epoch, dataloader):
        print('Validation...')
        self.model.eval()
        Avg_loss = AverageMeter()
        with torch.no_grad():
            correct = 0
            max_num = 0
            for i, (d, l, d2) in tqdm(enumerate(dataloader)):
                outputs = self.model(d.cuda(), d2.cuda())

                # Forward propagation
                loss = self.criterion1(outputs, l.cuda())

                Avg_loss.update(loss.item(), d.size(0))

                pred = torch.argmax(outputs, dim=1)
                correct += (pred == l.cuda()).sum().item()
                max_num += len(d)
                acc = float(correct) / max_num

                if (i + 1) % self.args.print_freq == 0:
                    print(' Validing [%2d/%2d, %4d/%4d], Loss %.4f(%.4f), Acc: %.4f \t time: @%s' % (
                        epoch, self.args.max_epochs, i, len(dataloader.dataset) / self.args.testing_batch_size,
                        loss.item(),
                        Avg_loss.avg, acc,
                        time.strftime('%m.%d %H:%M:%S', time.localtime(time.time()))))
                    self.vis.plot_many({'loss': Avg_loss.avg, 'Acc': acc}, 'Valid-' + self.args.type)
        return acc

    def test(self, epoch, dataloader):
        print('Testing...')
        self.model.eval()
        with torch.no_grad():
            correct = 0
            max_num = 0
            t = time.time()
            for i, (d, l, d2) in tqdm(enumerate(dataloader)):
                outputs = self.model(d.cuda(), d2.cuda())
                pred = torch.argmax(outputs, dim=1)
                correct += (pred == l.cuda()).sum().item()  # l在dataloader里肯定是label
                max_num += len(d)
                acc = float(correct) / max_num

                if (i + 1) % self.args.print_freq == 0:
                    print(' Testing [%2d/%2d, %4d/%4d], Acc: %.4f \t time: @%s' % (
                        epoch, self.args.max_epochs, i, len(dataloader.dataset) / self.args.testing_batch_size, acc,
                        time.strftime('%m.%d %H:%M:%S', time.localtime(time.time()))))
        return acc


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yml',
                        dest='config', help='to set the parameters')
    parser.add_argument('-r', '--resume', default='', help='load model')
    parser.add_argument('-m', '--mode', help='train or valid or test')
    parser.add_argument('-t', '--type', help='K or M')
    parser.add_argument('-g', '--gpu_ids', default="0,1", help="gpu")
    parser.add_argument('-l', '--res_layer', default=18, help="ResNet Layer")
    parser.add_argument('-i', '--init_model', default="", help="Pretrained model on 20 BN")

    parser.add_argument('--Mode', default='score', type=str,
                        help='Mode (score | feature). score outputs class scores. feature outputs features (after global average pooling).')
    parser.add_argument('--resnet_shortcut', default='A', type=str, help='Shortcut type of resnet (A | B)')
    parser.set_defaults(verbose=False)
    parser.add_argument('--verbose', action='store_true', help='')
    parser.set_defaults(verbose=False)
    return parser.parse_args()


if __name__ == '__main__':
    seed = 123
    torch.manual_seed(seed)  # Set the random seed for CPU to 123
    # torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Set the random seed for all available GPUs to 123
    random.seed(seed)
    np.random.seed(seed)

    args = Config(parse())  # Config()是config.py这个文件里定义的函数
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    
    
    cudnn.benchmark = True  # This setting enables the automatic optimization of cuDNN, NVIDIA's deep neural network library, to select the best convolutional algorithm to speed up model training.
    vis = Visualizer(args.visname)
    model = Module(args)

    train_loader, valid_loader, test_loader = GetData(args)
    train_val(model, args, vis, train_loader, valid_loader, test_loader)
    print(f"test_loader长啥样：{test_loader}")
    print(f"test_loader的尺寸：{test_loader.shape}")
    # train_val(model, args, train_loader, valid_loader, test_loader)
    
    '''
    the really useful called function in main: Module() 、GetData()、train_val()
        model = Module(args) 
        train_loader, valid_loader, test_loader = GetData(args)
        train_val(model, args, vis, train_loader, valid_loader, test_loader)
    '''