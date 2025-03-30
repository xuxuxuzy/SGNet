import torch
from torch.autograd import Variable
import os
import argparse
from datetime import datetime
from lib.pvt2 import SGNet
from utils.dataloader import get_loader, test_dataset
from utils.utils import clip_gradient, adjust_lr, AvgMeter,adjust_lr_d
import torch.nn.functional as F
import numpy as np
import logging


import matplotlib.pyplot as plt


train_loss_list = []
mean_dice_list = []


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()


def test(model, path, dataset):

    data_path = os.path.join(path, dataset)
    image_root = '{}/images/'.format(data_path)
    gt_root = '{}/masks/'.format(data_path)
    model.eval()
    num1 = len(os.listdir(gt_root))
    test_loader = test_dataset(image_root, gt_root, 352)
    DSC = 0.0
    # test_loss = 0.0  
    # num_batches = 0  
    for i in range(num1):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        res ,_,_,_,_,_= model(image)
        # eval Dice
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        #pcs
        res[torch.where(res>0)] /= (res>0).float().mean()
        res[torch.where(res<0)] /= (res<0).float().mean()
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        input = res
        target = np.array(gt)
        N = gt.shape
        smooth = 1
        input_flat = np.reshape(input, (-1))
        target_flat = np.reshape(target, (-1))
        intersection = (input_flat * target_flat)
        dice = (2 * intersection.sum() + smooth) / (input.sum() + target.sum() + smooth)
        dice = '{:.4f}'.format(dice)
        dice = float(dice)
        DSC = DSC + dice

        # pred_tensor = torch.tensor(res, dtype=torch.float32).cuda().unsqueeze(0).unsqueeze(0)
        # gt_tensor = torch.tensor(gt, dtype=torch.float32).cuda().unsqueeze(0).unsqueeze(0)
        # loss = structure_loss(pred_tensor, gt_tensor)
        # test_loss += loss.item()
        # num_batches += 1  
    # avg_test_loss = test_loss / num_batches if num_batches > 0 else 0

    return DSC / num1



def train(train_loader, model, optimizer, epoch, test_path , edge_loss):
    model.train()
    global best
    size_rates = [0.75, 1, 1.25] 
    loss_P1_record = AvgMeter()
    epoch_loss = 0.0 
    # loss_record = AvgMeter()
    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            optimizer.zero_grad()
            # ---- data prepare ----
            images, gts ,egs,tts= pack
            images = Variable(images).cuda()
            gts = Variable(gts).cuda()
            egs = Variable(egs).cuda()
            tts = Variable(tts).cuda()
            # ---- rescale ----
            trainsize = int(round(opt.trainsize * rate / 32) * 32)
            if rate != 1:
                images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                egs = F.upsample(egs, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                tts = F.upsample(tts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            # ---- forward ----
            P1,E,T,P3,P4,P5= model(images)

            # ---- loss function ----
            loss_P1 = structure_loss(P1, gts)
            # loss_P2 = structure_loss(P2, gts)
            loss_P3 = structure_loss(P3, gts)
            loss_P4 = structure_loss(P4, gts)
            loss_P5 = structure_loss(P5, gts)
            loss_E = edge_loss(E, egs)
            loss_T = edge_loss(T, tts)
            loss = loss_P1 +loss_E+loss_T+loss_P3+loss_P4+loss_P5


            # ---- backward ----
            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            # ---- recording loss ----
            epoch_loss += loss.item()
            if rate == 1:
                loss_P1_record.update(loss_P1.data, opt.batchsize)

                # loss_record.update(loss.data, opt.batchsize)
        # ---- train visualization ----
        if i % 20 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                  ' lateral-5: {:0.4f}]'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step,
                         loss_P1_record.show()))

    avg_train_loss = epoch_loss / (len(train_loader) * len(size_rates))
    train_loss_list.append(avg_train_loss)
    # save model 
    save_path = (opt.train_save)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # if epoch > 50 :
    #     torch.save(model.state_dict(), save_path +str(epoch)+ 'SGNet.pth')
    # choose the best model

    global dict_plot
   
    test1path = ''
    if (epoch + 1) % 1 == 0:
        sum = 0.000000
        for dataset in ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']:
            dataset_dice = test(model, test1path, dataset)
            sum += dataset_dice
            logging.info('epoch: {}, dataset: {}, dice: {}'.format(epoch, dataset, dataset_dice))
            print(dataset, ': ', dataset_dice)
            dict_plot[dataset].append(dataset_dice)
        # meandice = test(model, test_path, 'test')
        # val_loss_list.append(avg_test_loss) 
        # dict_plot['test'].append(meandice)
        meandice = sum / 5
        mean_dice_list.append(meandice)
        if meandice > best:
            best = meandice
            torch.save(model.state_dict(), save_path + 'SGNet.pth')
            # torch.save(model.state_dict(), save_path +str(epoch)+ 'SGNet-best.pth')
            print('##############################################################################best', best)
            logging.info('##############################################################################best:{}'.format(best))

#
# def plot_train(dict_plot=None, name = None):
#     color = ['red', 'lawngreen', 'lime', 'gold', 'm', 'plum', 'blue']
#     line = ['-', "--"]
#     for i in range(len(name)):
#         plt.plot(dict_plot[name[i]], label=name[i], color=color[i], linestyle=line[(i + 1) % 2])
#         transfuse = {'CVC-300': 0.902, 'CVC-ClinicDB': 0.918, 'Kvasir': 0.918, 'CVC-ColonDB': 0.773,'ETIS-LaribPolypDB': 0.733, 'test':0.83}
#         plt.axhline(y=transfuse[name[i]], color=color[i], linestyle='-')
#     plt.xlabel("epoch")
#     plt.ylabel("dice")
#     plt.title('Train')
#     plt.legend()
#     plt.savefig('eval.png')
    # plt.show()
def plot_train():
    plt.figure(figsize=(8, 6))

  
    plt.plot(train_loss_list, label="Train Loss", color="blue", linestyle="-")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.legend()

    plt.grid(True)
    plt.savefig("train_loss_curve.png")
    plt.show()

    plt.figure(figsize=(8, 6))

  
    plt.plot(mean_dice_list, label="Test Mean Dice", color="red", linestyle="-")

    plt.xlabel("Epoch")
    plt.ylabel("Mean dice")
    # plt.title("Training & Validation Loss")
    plt.legend()

    plt.grid(True)
    plt.savefig("Meandice.png")
    plt.show()

    
if __name__ == '__main__':
    dict_plot = {'CVC-300':[], 'CVC-ClinicDB':[], 'Kvasir':[], 'CVC-ColonDB':[], 'ETIS-LaribPolypDB':[], 'test':[] }

    name = ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB', 'test']
    ##################model_name#############################
    model_name = 'SGNet'
    ###############################################
    parser = argparse.ArgumentParser()

    parser.add_argument('--epoch', type=int,
                        default=150, help='epoch number')

    parser.add_argument('--lr', type=float,
                        default=1e-4, help='learning rate')

    parser.add_argument('--optimizer', type=str,
                        default='AdamW', help='choosing optimizer AdamW or SGD')

    parser.add_argument('--augmentation',
                        default=False, help='choose to do random flip rotation')

    parser.add_argument('--batchsize', type=int,
                        default=16, help='training batch size')
    #352
    parser.add_argument('--trainsize', type=int,
                        default=352, help='training dataset size')

    parser.add_argument('--clip', type=float,
                        default=0.5, help='gradient clipping margin')

    parser.add_argument('--decay_rate', type=float,
                        default=0.5, help='decay rate of learning rate')

    parser.add_argument('--decay_epoch', type=int,
                        default=50, help='every n epochs decay learning rate')

    parser.add_argument('--train_path', type=str,
                        default='',
                        help='path to train dataset')

    parser.add_argument('--test_path', type=str,
                        default='',
                        help='path to testing Kvasir dataset')

    parser.add_argument('--train_save', type=str,
                        default='./model_pth/'+'SGNet'+'/')

    opt = parser.parse_args()
    logging.basicConfig(filename='train.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')

    # ---- build models ----
    torch.cuda.set_device(0)  # set your gpu device
    model = SGNet().cuda()

    best = 0

    params = model.parameters()

    if opt.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(params, opt.lr, weight_decay=1e-4)
    else:
        optimizer = torch.optim.SGD(params, opt.lr, weight_decay=1e-4, momentum=0.9)

    print(optimizer)
    image_root = '{}/images/'.format(opt.train_path)
    gt_root = '{}/masks/'.format(opt.train_path)
    edge_root = '{}/edges/'.format(opt.train_path)
    texture_root = '{}/textures/'.format(opt.train_path)
    train_loader = get_loader(image_root, gt_root,edge_root,texture_root, batchsize=opt.batchsize, trainsize=opt.trainsize,
                              augmentation=opt.augmentation)
    total_step = len(train_loader)

    print("#" * 20, "Start Training", "#" * 20)
    edge_loss = torch.nn.BCEWithLogitsLoss()
    for epoch in range(1, opt.epoch):
        if epoch in [50,90,120]:
            adjust_lr_d(optimizer, opt.lr, epoch, 0.5)

        train(train_loader, model, optimizer, epoch, opt.test_path,edge_loss)
    
    # plot the eval.png in the training stage
    plot_train()
