import torch.nn as nn
import torch.nn.modules
import torch.optim as optim
from matplotlib.pyplot import gray


from get_patch import *
import time
import argparse
from SignalProcessing import compare_SNR
from models.model import Uformer
from models.restormer_arch import Restormer

from net import ResidualGroup, RNLNUNCA, default_conv, RNLN

parser=argparse.ArgumentParser(description='Pytorch Seg_rcan')

parser.add_argument('--model', default='DnCNN', type=str, help='choose a type of model')
parser.add_argument('--data_dir', default='data/', type=str, help='path of train data')

parser.add_argument('--data_dir1', default='data/7m_shots_0201_0329', type=str, help='path of train data')
parser.add_argument('--data_dir2', default='data/1997_2.5D_shots', type=str, help='path of train data')
parser.add_argument('--data_dir3', default='data/Anisotropic_FD_Model_Shots_part1', type=str, help='path of train data')
parser.add_argument('--data_dir4', default='data/Model94_shots', type=str, help='path of train data')
parser.add_argument('--data_dir5', default='data/seismic', type=str, help='path of train data')
parser.add_argument('--data_dir6', default='data/shots0001_0200', type=str, help='path of train data')
parser.add_argument('--data_dir7', default='data/timodel_shot_data_II_shot001-320', type=str, help='path of train data')

parser.add_argument('--sigma', default=25, type=int, help='noise level')
parser.add_argument('--epoch', default=100, type=int, help='number of train epoches')
parser.add_argument('--lr', default=0.005, type=float, help='initial learning rate for Adam')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--patch_size', default=(64,64), type=int, help='patch size')
parser.add_argument('--stride', default=(32,32), type=int, help='the step size to slide on the data')
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

parser.add_argument('--growth', type=str, default='momentum', help='Growth mode. Choose from: momentum, random, and momentum_neuron.')
parser.add_argument('--prune', type=str, default='magnitude', help='Prune mode / pruning mode. Choose from: magnitude, SET.')
parser.add_argument('--redistribution', type=str, default='momentum', help='Redistribution mode. Choose from: momentum, magnitude, nonzeros, or none.')
parser.add_argument('--prune-rate', type=float, default=0.50, help='The pruning rate / prune rate.')
parser.add_argument('--density', type=float, default=0.05, help='The density of the overall sparse network.')
parser.add_argument('--dense', action='store_true', help='Enable dense mode. Default: False.')
parser.add_argument('--verbose', action='store_true', help='Prints verbose status of pruning/growth algorithms.')

args = parser.parse_args()

def set_seed(seed=999):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

set_seed()  # 设置随机种子

output_CA=[]
i=0

batch_size = args.batch_size
cuda = torch.cuda.is_available()

Flag=False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
n_epoch=args.epoch
sigma=args.sigma


if __name__ == '__main__':
    #=========================数据==================================
    train_data_full = []
    train_data_full = datagenerator(data_dir=args.data_dir, patch_size=args.patch_size, stride=args.stride,
                                train_data_num=args.train_data_num,
                                download=args.download, datasets=args.datasets, aug_times=args.aug_times,
                                 scales=args.scales, verbose=args.verbose1, jump=args.jump, agc=args.agc)

    # train_data1 = datagenerator(data_dir=args.data_dir1, patch_size=args.patch_size, stride=args.stride,
    #                             train_data_num=args.train_data_num,
    #                             download=args.download, datasets=args.datasets, aug_times=args.aug_times,
    #                             scales=args.scales, verbose=args.verbose1, jump=args.jump, agc=args.agc)
    # # train_data2 = datagenerator(data_dir=args.data_dir2, patch_size=args.patch_size, stride=args.stride,
    # #                             train_data_num=args.train_data_num, download=args.download, datasets=args.datasets,
    # #                             aug_times=args.aug_times,
    # #                             scales=args.scales, verbose=args.verbose1, jump=args.jump, agc=args.agc)
    # train_data3 = datagenerator(data_dir=args.data_dir3, patch_size=args.patch_size, stride=args.stride,
    #                             train_data_num=args.train_data_num,
    #                             download=args.download, datasets=args.datasets, aug_times=args.aug_times,
    #                             scales=args.scales, verbose=args.verbose1, jump=args.jump, agc=args.agc)
    #
    # train_data4 = datagenerator(data_dir=args.data_dir4, patch_size=args.patch_size, stride=args.stride,
    #                             train_data_num=args.train_data_num,
    #                             download=args.download, datasets=args.datasets, aug_times=args.aug_times,
    #                             scales=args.scales, verbose=args.verbose1, jump=args.jump, agc=args.agc)
    #
    # train_data5 = datagenerator(data_dir=args.data_dir5, patch_size=args.patch_size, stride=args.stride,
    #                             train_data_num=args.train_data_num,
    #                             download=args.download, datasets=args.datasets, aug_times=args.aug_times,
    #                             scales=args.scales, verbose=args.verbose1, jump=args.jump, agc=args.agc)
    #
    # train_data6 = datagenerator(data_dir=args.data_dir6, patch_size=args.patch_size, stride=args.stride,
    #                             train_data_num=args.train_data_num,
    #                             download=args.download, datasets=args.datasets, aug_times=args.aug_times,
    #                             scales=args.scales, verbose=args.verbose1, jump=args.jump, agc=args.agc)
    # train_data7 = datagenerator(data_dir=args.data_dir7, patch_size=args.patch_size, stride=args.stride,
    #                             train_data_num=args.train_data_num,
    #                             download=args.download, datasets=args.datasets, aug_times=args.aug_times,
    #                             scales=args.scales, verbose=args.verbose1, jump=args.jump, agc=args.agc)
    # 加载数据
    # train_data_full = np.append(train_data1, train_data3, 0)
    # train_data_full = np.append(train_data_full, train_data4, 0)
    # train_data_full = np.append(train_data_full, train_data5, 0)
    # train_data_full = np.append(train_data_full, train_data6, 0)
    # train_data_full = datagenerator(data_dir=args.data_dir, patch_size=args.patch_size, stride=args.stride,
    #                             train_data_num=args.train_data_num,
    #                             download=args.download, datasets=args.datasets, aug_times=args.aug_times,
    #                             scales=args.scales, verbose=args.verbose1, jump=args.jump, agc=args.agc)


    train_data_full = train_data_full.astype(np.float64)
    print(train_data_full.shape)
    train_data_full = torch.from_numpy(train_data_full.transpose((0,3,1,2)))
    Downsamlpe_train_data = DownsamplingDataset(train_data_full,2,regular=True)
    train_loader = DataLoader(dataset=Downsamlpe_train_data,drop_last=True,batch_size=batch_size,shuffle=True)

    #=========================模型==================================

    net = Uformer(img_size=64, embed_dim=16,depths=[2, 2, 2, 2, 2, 2, 2, 2, 2],
                 win_size=4, mlp_ratio=4., token_projection='linear', token_mlp='leff', modulator=True, shift_flag=False)

    #=======================损失函数=================================
    # criterion=nn.L1Loss(reduce=True,size_average=False)
    criterion = nn.MSELoss(reduction='sum')


    #=======================优化器===================================
    if torch.cuda.device_count() > 1:
        print("Use", torch.cuda.device_count(), 'gpus')
    net = nn.DataParallel(net,[0,1])
    net.to(device)
    optimizer = optim.Adam(net.parameters(),lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=20,gamma=0.1)

    #=======================训练=====================================
    net.train()
    SNR_list=[]
    for epoch in range(n_epoch):
        start_time=time.time()
        lr = scheduler.get_last_lr()[0]
        for i,batch_data in enumerate(train_loader):
            LR, HR= batch_data[0].to(device), batch_data[1].to(device)
            outputs=net(LR)
            optimizer.zero_grad()
            loss=criterion(outputs, HR)
            loss.backward()
            optimizer.step()
            if (i+1)%args.display==0:
                print('%4d %4d / loss=%2.4f %2.4f' % (epoch+1,i+1,loss.item()/batch_size,lr))

        scheduler.step()
        using_time=time.time()-start_time
        print('epoch time=%2.4f' % (using_time))
        torch.save(net.state_dict(),os.path.join(args.model_dir, 'model_uf_full_regular_SEG_inter0.5_%03d.pkl' % (epoch+1)))
        print(SNR_list)

