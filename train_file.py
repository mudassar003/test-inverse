import scipy.io as sio
import os
from torch.utils.data import Dataset, DataLoader
import platform
from argparse import ArgumentParser
from utils import *
from network.net import APG
from time import time
import glob
import subprocess


parser = ArgumentParser(description='')

parser.add_argument('--mark_str', type=str, default='Cardiac_APG', help='result directory')

parser.add_argument('--start_epoch', type=int, default=0, help='epoch number of start training')
#训练轮数
parser.add_argument('--end_epoch', type=int, default=300, help='epoch number of end training')
# parser.add_argument('--end_epoch', type=int, default=5000, help='epoch number of end training')
parser.add_argument('--gpu_list', type=str, default='0', help='gpu index')
#batchsize
parser.add_argument('--batch_size', type=int, default=4, help='')
#mask
parser.add_argument('--cs_ratio', type=int, default=10, help='from {5,10, 20, 30, 40, 50}')
#保存模型
parser.add_argument('--save_interval', type=int, default=5, help='eg, save model every 10 epochs')
# parser.add_argument('--save_interval', type=int, default=5, help='eg, save model every 10 epochs')
#训练数据集路径
parser.add_argument('--data_dir', type=str, default='data', help='training data directory')
#模型路径
parser.add_argument('--model_dir', type=str, default='model', help='trained or pre-trained model directory')
#日志路径
parser.add_argument('--log_dir', type=str, default='log', help='log directory')
#训练集名字
parser.add_argument('--trainset_name', type=str, default='cardiac.mat', help='train dataset')
#训练集标签
parser.add_argument('--trainset_name_matkey', type=str, default='labels', help='')

parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
parser.add_argument('--show_model_paras', type=int, default=0, help='')
#层数
parser.add_argument('--layer_num', type=int, default=9, help='phase number')
parser.add_argument('--rb_num', type=int, default=2, help='')

#什么时候测试
parser.add_argument('--test_while_train', type=int, default=1, help='whether to test when saving model')
#测试集名字
parser.add_argument('--test_name', type=str, default='card_test', help='name of test set')
#测试集的图片类型png/tif
parser.add_argument('--test_img_type', type=str, default='png', help='name of test set')



args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_list
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

start_epoch = args.start_epoch
end_epoch = args.end_epoch
learning_rate = args.learning_rate
layer_num = args.layer_num
rb_num = args.rb_num
Training_data_Name = args.trainset_name
batch_size = args.batch_size
cs_ratio = args.cs_ratio
test_while_train = (args.test_while_train == 1)




model_full_name = '%s_ratio_%d_layer_%d' % (
    args.mark_str, cs_ratio, layer_num)
model_dir = os.path.join(args.model_dir, model_full_name)
log_file_name = "./%s/Log_TRAIN_%s.txt" % (args.log_dir, model_full_name)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)

# load data
Training_data = sio.loadmat('./%s/%s' % (args.data_dir, Training_data_Name))
Training_labels = Training_data[args.trainset_name_matkey]
nrtrain = Training_labels.shape[0]

Phi_data_Name = 'mask/mask_10.mat'
Phi_data = sio.loadmat(Phi_data_Name)
mask_matrix = Phi_data['mask_matrix']

mask_matrix = torch.from_numpy(mask_matrix).type(torch.FloatTensor)
mask = mask_matrix.unsqueeze(0).unsqueeze(-1)

mask = mask.to(device)

model = APG(layer_num, rb_num)
model = nn.DataParallel(model)
model = model.to(device)
show_model_paras(model, args.show_model_paras == 1)


class RandomDataset(Dataset):
    def __init__(self, data, length):
        self.data = data
        self.len = length

    def __getitem__(self, index):
        return torch.Tensor(self.data[index, :]).float()

    def __len__(self):
        return self.len


if (platform.system() == "Windows"):
    rand_loader = DataLoader(dataset=RandomDataset(Training_labels, nrtrain), batch_size=batch_size, num_workers=0,
                             shuffle=True)
else:
    rand_loader = DataLoader(dataset=RandomDataset(Training_labels, nrtrain), batch_size=batch_size, num_workers=4,
                             shuffle=True)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

if start_epoch == -1:
    pre_model_dir = model_dir
    filelist = sorted(glob.glob('./%s/net_params_*.pkl' % pre_model_dir))
    if len(filelist) == 0:
        start_epoch = 0
        print('start epoch is -1, i.e., starting from epoch %d' % start_epoch)
    else:
        int_list = []
        for i in range(len(filelist)):
            model_path = filelist[i]
            this_epoch = int(os.path.split(model_path)[-1].split('.')[0].split('_')[-1])
            int_list.append(this_epoch)
        start_epoch = max(int_list)
        print('start epoch is -1, i.e., starting from epoch %d' % start_epoch)
if start_epoch > 0:
    pre_model_dir = model_dir
    model_path = './%s/net_params_%d.pkl' % (pre_model_dir, start_epoch)
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(model_path))
    else:
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))





TRAIN_TIME_All = np.zeros([1, (end_epoch - start_epoch)], dtype=np.float32)
TRAIN_TIME_All_i = 0

for epoch_i in range(start_epoch + 1, end_epoch + 1):
    batch_cnt = 0
    loss_discrepancy_accum = 0
    train_start = time()
    model.train()

    for data in rand_loader:
        gt = data.to(device)
        gt = gt.view(gt.shape[0], 1, gt.shape[1], gt.shape[2])

        model_output = model(gt,mask)

        x_output = model_output


        loss_discrepancy = torch.mean(torch.pow(x_output - gt, 2))

        phi_thr = torch.Tensor([cs_ratio / 100.0]).to(device)

        loss_all = loss_discrepancy

        optimizer.zero_grad()
        loss_all.backward()
        optimizer.step()

        batch_cnt += 1
        loss_discrepancy_accum += loss_discrepancy.item()

    train_end = time()
    train_epoch_time = train_end - train_start
    TRAIN_TIME_All[0, TRAIN_TIME_All_i] = train_epoch_time
    TRAIN_TIME_All_i += 1

    output_data = "[%02d/%02d] Discrepancy Loss: %.5f, Train Time: %.4f, mask ratio: %.5f \n" % (
        epoch_i, end_epoch, loss_discrepancy_accum / batch_cnt, train_epoch_time, torch.mean(mask_matrix).item())
    print(output_data, end='')
    output_file = open(log_file_name, 'a')
    output_file.write(output_data)
    output_file.close()

    if epoch_i % args.save_interval == 0:
        torch.save(model.state_dict(), "./%s/net_params_%d.pkl" % (model_dir, epoch_i))
        if test_while_train:
            cmd_png = 'python test_file.py --epoch_num %d --layer_num %d --cs_ratio %d' \
                      ' --test_name %s --test_img_type %s --mark_str %s' \
                      ' --rb_num %d --model_dir %s --log_dir %s' % (
                          epoch_i, layer_num, cs_ratio, args.test_name, args.test_img_type,
                          args.mark_str, rb_num, args.model_dir, args.log_dir)
            print(cmd_png, end='\n')
            process_png = subprocess.Popen(cmd_png, shell=True)
            process_png.wait()

output_data = 'From epoch %d to epoch %d, Average Train Epoch Time: %.4f \n' % (
start_epoch, end_epoch, np.mean(TRAIN_TIME_All))
output_file = open(log_file_name, 'a')
output_file.write(output_data)
output_file.close()