import torch.nn as nn
import torch.nn.modules
import torch.optim as optim
from torch.utils.data import DataLoader
import random
import time
import numpy as np
import argparse
import utils
import os
import datetime
from  tqdm import tqdm
from matplotlib.pyplot import gray
from losses import CharbonnierLoss

from tqdm import tqdm
from warmup_scheduler import GradualWarmupScheduler
from torch.optim.lr_scheduler import StepLR
from timm.utils import NativeScaler

# from get_patch import *
from model.uformer_dcn import Uformer

from dataset.dataset_denoise import *

parser=argparse.ArgumentParser(description='Pytorch UDCN')

# global settings
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--epoch', type=int, default=250, help='training epochs')
parser.add_argument('--train_workers', type=int, default=4, help='train_dataloader workers')
parser.add_argument('--eval_workers', type=int, default=4, help='eval_dataloader workers')
parser.add_argument('--dataset', type=str, default ='SIDD')
parser.add_argument('--pretrain_weights',type=str, default='./log/Uformer_B/models/model_best.pth', help='path of pretrained_weights')
parser.add_argument('--optimizer', type=str, default ='adamw', help='optimizer for training')
parser.add_argument('--lr_initial', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--step_lr', type=int, default=50, help='weight decay')
parser.add_argument('--weight_decay', type=float, default=0.02, help='weight decay')
# GPU
parser.add_argument('--gpu', type=str, default='1', help='GPUs')
# model
parser.add_argument('--model', type=str, default ='Uformer',  help='choose a type of model')
# denoising
parser.add_argument('--mode', type=str, default ='denoising',  help='image restoration mode')
# in_channel
parser.add_argument('--dd_in', type=int, default=3, help='dd_in')

# args for saving
parser.add_argument('--save_dir', type=str, default ='./logs/',  help='save dir')
parser.add_argument('--save_images', action='store_true',default=False)
parser.add_argument('--env', type=str, default ='_',  help='env')
parser.add_argument('--checkpoint', type=int, default=50, help='checkpoint')

# args for Uformer
parser.add_argument('--norm_layer', type=str, default ='nn.LayerNorm', help='normalize layer in transformer')
parser.add_argument('--embed_dim', type=int, default=32, help='dim of emdeding features')
parser.add_argument('--win_size', type=int, default=8, help='window size of self-attention')
parser.add_argument('--token_projection', type=str,default='linear', help='linear/conv token projection')
parser.add_argument('--token_mlp', type=str,default='leff', help='ffn/leff token mlp')  # LeFF
parser.add_argument('--att_se', action='store_true', default=False, help='se after sa')
parser.add_argument('--modulator', action='store_true', default=False, help='multi-scale modulator')  # Uformer modulator

# args for training
parser.add_argument('--train_ps', type=int, default=128, help='patch size of training sample')
parser.add_argument('--val_ps', type=int, default=128, help='patch size of validation sample')
parser.add_argument('--resume', action='store_true',default=False)
parser.add_argument('--train_dir', type=str, default ='./datasets/SIDD/train',  help='dir of train data')
parser.add_argument('--val_dir', type=str, default ='./datasets/SIDD/val',  help='dir of train data')
parser.add_argument('--warmup', action='store_true', default=False, help='warmup')
parser.add_argument('--warmup_epochs', type=int,default=3, help='epochs for warmup')

# ddp
parser.add_argument("--local_rank", type=int,default=-1,help='DDP parameter, do not modify')#不需要赋值，启动命令 torch.distributed.launch会自动赋值
parser.add_argument("--distribute",action='store_true',help='whether using multi gpu train')
parser.add_argument("--distribute_mode",type=str,default='DDP',help="using which mode to ")

#-------------------------------------------------------------------------------------------------

# parser.add_argument('--model', default='DnCNN', type=str, help='choose a type of model')
parser.add_argument('--data_dir', default='data/', type=str, help='path of train data')

parser.add_argument('--sigma', default=25, type=int, help='noise level')
parser.add_argument('--jump', default=1, type=int, help='the space between shot')
parser.add_argument('--download', default=False, type=bool, help='if you will download the dataset from the internet')
parser.add_argument('--datasets', default = 0, type = int, help='the num of datasets you want be download,if download = True')
parser.add_argument('--train_data_num', default=1000, type=int, help='the num of the train_data')
parser.add_argument('--aug_times', default=0, type=int, help='Number of aug operations')

parser.add_argument('--scales', default=[1], type=list, help='data scaling')
parser.add_argument('--agc', default=True, type=int, help='Normalize each trace by amplitude')
parser.add_argument('--verbose1', default=True, type=int, help='Whether to output the progress of data generation')
parser.add_argument('--display', default=10, type=int, help='interval for displaying loss')
parser.add_argument('--n_colors', type=int, default=1,help='number of color channels to use')
parser.add_argument('--reduction', type=int, default=16,help='number of feature maps reduction')
parser.add_argument('--n_resgroups',default=10,type=int,help='the num of RG,original:10')
parser.add_argument('--n_resblocks',default=10,type=int,help='the num of RCAB,original:20')
parser.add_argument('--n_feats',default=64,type=int,help='the channel of CA,original:64')

parser.add_argument('--test_data_dir', default='data/7m_shots_0201_0329', type=str, help='directory of test dataset')
parser.add_argument('--rate', default=2, type=float, help='missing rate')
parser.add_argument('--model_dir', default='model/rf/', help='directory of the model')
parser.add_argument('--model_name', default='model_rf_full_regular_SEG_inter0.5.pkl', type=str, help='the model name')
parser.add_argument('--result_dir', default='NL_SEG/rf/regular_0.5', type=str, help='directory of test result dataset')
parser.add_argument('--save_result', default=1, type=int, help='save the denoised image, 1 or 0')

# parser.add_argument('--growth', type=str, default='momentum', help='Growth mode. Choose from: momentum, random, and momentum_neuron.')
# parser.add_argument('--prune', type=str, default='magnitude', help='Prune mode / pruning mode. Choose from: magnitude, SET.')
# parser.add_argument('--redistribution', type=str, default='momentum', help='Redistribution mode. Choose from: momentum, magnitude, nonzeros, or none.')
# parser.add_argument('--prune-rate', type=float, default=0.50, help='The pruning rate / prune rate.')
# parser.add_argument('--density', type=float, default=0.05, help='The density of the overall sparse network.')
# parser.add_argument('--dense', action='store_true', help='Enable dense mode. Default: False.')
# parser.add_argument('--verbose', action='store_true', help='Prints verbose status of pruning/growth algorithms.')

args = parser.parse_args()

######### Logs dir ###########
log_dir = os.path.join(args.save_dir, 'denoising', args.dataset, args.model+args.env)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
logname = os.path.join(log_dir, datetime.datetime.now().isoformat()+'.txt')
print("Now time is : ",datetime.datetime.now().isoformat())
result_dir = os.path.join(log_dir, 'results')
model_dir  = os.path.join(log_dir, 'models')
utils.mkdir(result_dir)
utils.mkdir(model_dir)

# ######### Set Seeds ###########
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

def set_seed(seed=999):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

set_seed()  # 设置随机种子

######### Model ###########
model_restoration = Uformer(in_chans=3, dd_in=  args.dd_in, embed_dim= args.embed_dim, depths=[2, 2, 2, 2, 2, 2, 2, 2, 2])

with open(logname,'a') as f:
    f.write(str(args)+'\n')
    f.write(str(model_restoration)+'\n')

######### Optimizer ###########
start_epoch = 1
if args.optimizer.lower() == 'adam':
    optimizer = optim.Adam(model_restoration.parameters(), lr=args.lr_initial, betas=(0.9, 0.999), eps=1e-8,
                           weight_decay=args.weight_decay)
elif args.optimizer.lower() == 'adamw':
    optimizer = optim.AdamW(model_restoration.parameters(), lr=args.lr_initial, betas=(0.9, 0.999), eps=1e-8,
                            weight_decay=args.weight_decay)
else:
    raise Exception("Error optimizer...")

######### DataParallel ###########
model_restoration = torch.nn.DataParallel(model_restoration)
torch.cuda.empty_cache()
model_restoration.cuda()

######### Scheduler ###########
if args.warmup:
    print("Using warmup and cosine strategy!")
    warmup_epochs = args.warmup_epochs
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.nepoch - warmup_epochs, eta_min=1e-6)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs,
                                       after_scheduler=scheduler_cosine)
    scheduler.step()
else:
    step = 50
    print("Using StepLR,step={}!".format(step))
    scheduler = StepLR(optimizer, step_size=step, gamma=0.5)
    scheduler.step()

######### Resume ###########
if args.resume:
    path_chk_rest = args.pretrain_weights
    print("Resume from " + path_chk_rest)
    utils.load_checkpoint(model_restoration, path_chk_rest)
    start_epoch = utils.load_start_epoch(path_chk_rest) + 1
    lr = utils.load_optim(optimizer, path_chk_rest)

    # for p in optimizer.param_groups: p['lr'] = lr
    # warmup = False
    # new_lr = lr
    # print('------------------------------------------------------------------------------')
    # print("==> Resuming Training with learning rate:",new_lr)
    # print('------------------------------------------------------------------------------')
    for i in range(1, start_epoch):
        scheduler.step()
    new_lr = scheduler.get_lr()[0]
    print('------------------------------------------------------------------------------')
    print("==> Resuming Training with learning rate:", new_lr)
    print('------------------------------------------------------------------------------')

    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.nepoch-start_epoch+1, eta_min=1e-6)

######### Loss ###########
criterion = CharbonnierLoss().cuda()

######### DataLoader ###########
print('===> Loading datasets')
img_options_train = {'patch_size': args.train_ps}
train_dataset = get_training_data(args.train_dir, img_options_train)
train_loader = DataLoader(dataset=train_dataset, batch_size=opt.batch_size, shuffle=True,
                          num_workers=args.train_workers, pin_memory=False, drop_last=False)
val_dataset = get_validation_data(args.val_dir)
val_loader = DataLoader(dataset=val_dataset, batch_size=opt.batch_size, shuffle=False,
                        num_workers=args.eval_workers, pin_memory=False, drop_last=False)

len_trainset = train_dataset.__len__()
len_valset = val_dataset.__len__()
print("Sizeof training set: ", len_trainset, ", sizeof validation set: ", len_valset)
######### validation ###########
with torch.no_grad():
    model_restoration.eval()
    psnr_dataset = []
    psnr_model_init = []
    for ii, data_val in enumerate((val_loader), 0):
        target = data_val[0].cuda()
        input_ = data_val[1].cuda()
        with torch.cuda.amp.autocast():
            restored = model_restoration(input_)
            restored = torch.clamp(restored, 0, 1)
        psnr_dataset.append(utils.batch_PSNR(input_, target, False).item())
        psnr_model_init.append(utils.batch_PSNR(restored, target, False).item())
    psnr_dataset = sum(psnr_dataset) / len_valset
    psnr_model_init = sum(psnr_model_init) / len_valset
    print('Input & GT (PSNR) -->%.4f dB' % (psnr_dataset), ', Model_init & GT (PSNR) -->%.4f dB' % (psnr_model_init))

######### train ###########
print('===> Start Epoch {} End Epoch {}'.format(start_epoch, args.nepoch))
best_psnr = 0
best_epoch = 0
best_iter = 0
eval_now = len(train_loader) // 4
print("\nEvaluation after every {} Iterations !!!\n".format(eval_now))

loss_scaler = NativeScaler()
torch.cuda.empty_cache()
for epoch in range(start_epoch, args.epoch + 1):
    epoch_start_time = time.time()
    epoch_loss = 0
    train_id = 1

    for i, data in enumerate(tqdm(train_loader), 0):
        # zero_grad
        optimizer.zero_grad()

        target = data[0].cuda()
        input_ = data[1].cuda()

        if epoch > 5:
            target, input_ = utils.MixUp_AUG().aug(target, input_)
        with torch.cuda.amp.autocast():
            restored = model_restoration(input_)
            loss = criterion(restored, target)
        loss_scaler(
            loss, optimizer, parameters=model_restoration.parameters())
        epoch_loss += loss.item()

        #### Evaluation ####
        if (i + 1) % eval_now == 0 and i > 0:
            with torch.no_grad():
                model_restoration.eval()
                psnr_val_rgb = []
                for ii, data_val in enumerate((val_loader), 0):
                    target = data_val[0].cuda()
                    input_ = data_val[1].cuda()
                    filenames = data_val[2]
                    with torch.cuda.amp.autocast():
                        restored = model_restoration(input_)
                    restored = torch.clamp(restored, 0, 1)
                    psnr_val_rgb.append(utils.batch_PSNR(restored, target, False).item())

                psnr_val_rgb = sum(psnr_val_rgb) / len_valset

                if psnr_val_rgb > best_psnr:
                    best_psnr = psnr_val_rgb
                    best_epoch = epoch
                    best_iter = i
                    torch.save({'epoch': epoch,
                                'state_dict': model_restoration.state_dict(),
                                'optimizer': optimizer.state_dict()
                                }, os.path.join(model_dir, "model_best.pth"))

                print(
                    "[Ep %d it %d\t PSNR SIDD: %.4f\t] ----  [best_Ep_SIDD %d best_it_SIDD %d Best_PSNR_SIDD %.4f] " % (
                    epoch, i, psnr_val_rgb, best_epoch, best_iter, best_psnr))
                with open(logname, 'a') as f:
                    f.write(
                        "[Ep %d it %d\t PSNR SIDD: %.4f\t] ----  [best_Ep_SIDD %d best_it_SIDD %d Best_PSNR_SIDD %.4f] " \
                        % (epoch, i, psnr_val_rgb, best_epoch, best_iter, best_psnr) + '\n')
                model_restoration.train()
                torch.cuda.empty_cache()
    scheduler.step()

    print("------------------------------------------------------------------")
    print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time() - epoch_start_time,
                                                                              epoch_loss, scheduler.get_lr()[0]))
    print("------------------------------------------------------------------")
    with open(logname, 'a') as f:
        f.write(
            "Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time() - epoch_start_time,
                                                                                epoch_loss,
                                                                                scheduler.get_lr()[0]) + '\n')

    torch.save({'epoch': epoch,
                'state_dict': model_restoration.state_dict(),
                'optimizer': optimizer.state_dict()
                }, os.path.join(model_dir, "model_latest.pth"))

    if epoch % args.checkpoint == 0:
        torch.save({'epoch': epoch,
                    'state_dict': model_restoration.state_dict(),
                    'optimizer': optimizer.state_dict()
                    }, os.path.join(model_dir, "model_epoch_{}.pth".format(epoch)))
print("Now time is : ", datetime.datetime.now().isoformat())


#
# if __name__ == '__main__':
#     #=========================数据==================================
#     train_data_full = []
#     train_data_full = datagenerator(data_dir=args.data_dir, patch_size=args.patch_size, stride=args.stride,
#                                 train_data_num=args.train_data_num,
#                                 download=args.download, datasets=args.datasets, aug_times=args.aug_times,
#                                  scales=args.scales, verbose=args.verbose1, jump=args.jump, agc=args.agc)
#
#     train_data_full = train_data_full.astype(np.float64)
#     print(train_data_full.shape)
#     train_data_full = torch.from_numpy(train_data_full.transpose((0,3,1,2)))
#     Downsamlpe_train_data = DownsamplingDataset(train_data_full,2,regular=True)
#     train_loader = DataLoader(dataset=Downsamlpe_train_data,drop_last=True,batch_size=batch_size,shuffle=True)
#
#     #=========================模型==================================
#
#     net = Uformer(img_size=64, embed_dim=16,depths=[2, 2, 2, 2, 2, 2, 2, 2, 2],
#                  win_size=4, mlp_ratio=4., token_projection='linear', token_mlp='leff', modulator=True, shift_flag=False)
#
#     #=======================损失函数=================================
#     # criterion=nn.L1Loss(reduce=True,size_average=False)
#     criterion = nn.MSELoss(reduction='sum')
#
#
#     #=======================优化器===================================
#     if torch.cuda.device_count() > 1:
#         print("Use", torch.cuda.device_count(), 'gpus')
#     net = nn.DataParallel(net,[0,1])
#     net.to(device)
#     optimizer = optim.Adam(net.parameters(),lr=args.lr)
#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=20,gamma=0.1)
#
#     #=======================训练=====================================
#     net.train()
#     SNR_list=[]
#     for epoch in range(n_epoch):
#         start_time=time.time()
#         lr = scheduler.get_last_lr()[0]
#         for i,batch_data in enumerate(train_loader):
#             LR, HR= batch_data[0].to(device), batch_data[1].to(device)
#             outputs=net(LR)
#             optimizer.zero_grad()
#             loss=criterion(outputs, HR)
#             loss.backward()
#             optimizer.step()
#             if (i+1)%args.display==0:
#                 print('%4d %4d / loss=%2.4f %2.4f' % (epoch+1,i+1,loss.item()/batch_size,lr))
#
#         scheduler.step()
#         using_time=time.time()-start_time
#         print('epoch time=%2.4f' % (using_time))
#         torch.save(net.state_dict(),os.path.join(args.model_dir, 'model_uf_full_regular_SEG_inter0.5_%03d.pkl' % (epoch+1)))
#         print(SNR_list)
#
