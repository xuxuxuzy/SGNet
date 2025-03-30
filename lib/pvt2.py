import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.pvtv2 import pvt_v2_b2
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from model import *
from torchvision.transforms.functional import rgb_to_grayscale
from coordatt import CoordAtt
from msag import MSAG
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class RF(nn.Module):

    def __init__(self, in_channel, out_channel):
        super(RF, self).__init__()
        self.relu = nn.ReLU(True)

        # self.branch0 = nn.Sequential(
        #     BasicConv2d(in_channel, out_channel, 1),
        # )
        self.branch1 = nn.Sequential(

            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(

            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(

            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )

        self.conv_cat =BasicConv2d(3*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)
        self.ca = CoordAtt(in_channel, out_channel)
    def forward(self, x):
        # x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        tmp = x1+x2+x3
        x_cat = self.conv_cat(torch.cat((x3, x1, x2), dim=1))

        x = self.ca(x_cat + tmp)+x
        return x


class TAM(nn.Module):
    def __init__(self,hidden_dim):
        super(TAM,self).__init__()
        self.bn = nn.BatchNorm2d(hidden_dim)
        self.relu6 = nn.ReLU6(inplace=True)

        self.conv3 = BasicConv2d(hidden_dim*2, hidden_dim, 1, 1, 0)
        # self.conv4_1 = nn.Conv2d(hidden_dim * 4, hidden_dim, 1, 1, 0)
        self.conv3_1 =nn.Conv2d(hidden_dim * 3, hidden_dim, 1, 1, 0)
        self.conv2_1 = nn.Conv2d(hidden_dim * 2, hidden_dim, 1, 1, 0)
        self.conv1 = BasicConv2d(512, 64, 1, 1, 0)
        self.conv5 = BasicConv2d(hidden_dim, hidden_dim , 3, 1, 1)
        self.conv4 = BasicConv2d(128, 64, 1, 1, 0)
        self.conv7 = BasicConv2d(64, 64, 3, 1, 1)


        self.sigmoid = nn.Sigmoid()
        self.ca = CoordAtt(hidden_dim, hidden_dim)
        self.conv_f = nn.Sequential(
            BasicConv2d(hidden_dim, hidden_dim // 4, 1),
            nn.Conv2d(hidden_dim // 4, 1, 1)
        )

        self.up4 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up3 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.msag = MSAG(hidden_dim)


    def forward(self,x,x1,x2):
        tmp = x
        x = torch.cat([x,x1,x2],dim=1)
        x = self.conv3_1(x)+tmp

        x_bg = (1 - self.sigmoid(x)) * tmp

        pred_f = self.sigmoid(self.conv_f(x))
        pred_edge = x * (make_laplace(pred_f, 1))+x
        pred_edge = torch.cat([pred_edge,x_bg],dim=1)
        x_out = self.conv3(pred_edge)
        x_out = self.conv5(x_out)
        x_out = self.ca(x_out)

        return x_out

class EAM(nn.Module):
    def __init__(self,hidden_dim):
        super(EAM,self).__init__()
        self.bn = nn.BatchNorm2d(hidden_dim)
        self.relu6 = nn.ReLU6(inplace=True)

        self.conv3 = BasicConv2d(hidden_dim*2, hidden_dim, 1, 1, 0)
        self.conv3_1 =nn.Conv2d(hidden_dim * 3, hidden_dim, 1, 1, 0)
        # self.conv4_1 = nn.Conv2d(hidden_dim * 4, hidden_dim, 1, 1, 0)
        # self.conv2_1 = nn.Conv2d(hidden_dim * 2, hidden_dim, 1, 1, 0)
        self.conv1 = BasicConv2d(512, 64, 1, 1, 0)
        self.conv5 = BasicConv2d(hidden_dim, hidden_dim , 3, 1, 1)
        self.conv4 = BasicConv2d(128, 64, 1, 1, 0)
        self.conv7 = BasicConv2d(64, 64, 3, 1, 1)


        self.sigmoid = nn.Sigmoid()
        self.ca = CoordAtt(hidden_dim, hidden_dim)
        self.conv_f = nn.Sequential(
            BasicConv2d(hidden_dim, hidden_dim // 4, 1),
            nn.Conv2d(hidden_dim // 4, 1, 1)
        )

        self.up4 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up3 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.msag = MSAG(hidden_dim)


    def forward(self,x,x1,x2):
        tmp = x
        x = torch.cat([x,x1,x2],dim=1)
        x = self.conv3_1(x)+tmp
        x_bg = (1 - self.sigmoid(x)) * tmp

        pred_f = self.sigmoid(self.conv_f(x))
        pred_edge = x * (make_laplace(pred_f, 1))+x
        pred_edge = torch.cat([pred_edge,x_bg],dim=1)
        x_out = self.conv3(pred_edge)
        x_out = self.conv5(x_out)
        x_out = self.ca(x_out)

        return x_out



class FilterLayer(nn.Module):
    def __init__(self, in_planes, out_planes, reduction=16):
        super(FilterLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_planes, out_planes // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(out_planes // reduction, out_planes),
            nn.Sigmoid()
        )
        self.out_planes = out_planes

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y2 = self.max_pool(x).view(b, c)
        y = self.fc(y+y2).view(b, self.out_planes, 1, 1)
        return y

class FSP(nn.Module):
    def __init__(self, in_planes, out_planes, reduction=16):
        super(FSP, self).__init__()
        self.filter = FilterLayer(2*in_planes, out_planes, reduction)

    def forward(self, x, e,t):
        xe = torch.cat([x,e], dim=1)
        xt = torch.cat([x, t], dim=1)
        xe_w = self.filter(xe)
        xt_w = self.filter(xt)
        rec_e = xe_w*e
        rec_t = xt_w*t
        rec_x = xe_w*x + xt_w*x+x
        return rec_x,rec_e,rec_t


class FEEM(nn.Module):
    def __init__(self, in_planes, reduction=16):
        self.init__ = super(FEEM, self).__init__()
        self.in_planes = in_planes


        self.fsp_xet = FSP(in_planes, in_planes, reduction)

        self.conv1= nn.Conv2d(in_planes*2, in_planes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x,e,t):

        rec_x,rec_e,rec_t = self.fsp_xet(x,e,t)

        rec_xe = torch.cat([rec_x+rec_e,rec_x+e],dim=1)
        rec_xe = self.conv1(rec_xe)
        rec_xt = torch.cat([rec_x+rec_t,rec_x+t],dim=1)
        rec_xt = self.conv1(rec_xt)
        rec_out = torch.cat([rec_xt,rec_xe],dim=1)
        rec_out = self.conv1(rec_out)+ rec_x


        return rec_out



class Attention_block(nn.Module):
    def __init__(self, F_g):
        super(Attention_block, self).__init__()
        self.W_x = nn.Conv2d(F_g, F_g, 3, 1, 1)


        self.psi = nn.Sequential(
            nn.Conv2d(F_g, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )


        self.relu = nn.ReLU(inplace=True)
        self.sigmoid= nn.Sigmoid()
        self.msag =MSAG(F_g)
    def forward(self, x, g):
        g1 = self.msag(g)
        x1 = self.W_x(x)
        psi = g1 + x1
        psi = self.psi(psi)
        # return x+g
        return x1* psi+g1
class SGNet(nn.Module):
    def __init__(self, channel=32):
        super(SGNet, self).__init__()

        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        path = './pretrained_pth/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)


        self.mccm4 = RF(512, 512)
        self.mccm3 = RF(320, 320)
        self.mccm2 = RF(128, 128)
        self.mccm1 = RF(64, 64)
        self.tam = EAM(64)
        self.eam = TAM(64)

        self.msag = MSAG(64)
        self.msag2 = MSAG(512)


        self.ff4 = MSAG(512)
        self.ff3 = Attention_block(320)
        self.ff2 = Attention_block(128)
        self.ff1 = Attention_block(64)

        self.feem1 = FEEM(64)
        self.feem2 = FEEM(128)
        self.feem3 = FEEM(320)
        self.feem4 = FEEM(512)

        self.up =nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
        self.up2 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        self.up1 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True)
        self.up3 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.out4 = nn.Conv2d(512,320,1)
        self.out1 =  nn.Conv2d(256, 128, 1)
        self.out3 = nn.Conv2d(320, 128, 1)
        self.out2 =  nn.Conv2d(128, 64, 1)
        self.out_m =  nn.Conv2d(256, 64, 1)
        self.final_out = nn.Sequential(
            BasicConv2d(in_planes=64,out_planes=16,kernel_size=1,padding=0),
            nn.Conv2d(in_channels=16,out_channels=1,kernel_size=1)
        )
        self.final_out2 = nn.Sequential(
            BasicConv2d(512, 128, kernel_size=1,padding=0),
            BasicConv2d(128, 16, kernel_size=1, padding=0),
            nn.Conv2d(16, 1, 1),
        )

        self.final_out3 = nn.Sequential(
            BasicConv2d(320, 16, kernel_size=1, padding=0),
            nn.Conv2d(16, 1, 1)
        )
        self.final_out4 = nn.Sequential(
            BasicConv2d(128, 16, kernel_size=1, padding=0),
            nn.Conv2d(16, 1, 1)
        )

        self.relu = nn.ReLU(inplace=True)
        self.conv_up =  nn.Conv2d(64,128,1)
        self.conv_up2 = nn.Conv2d(128, 320, 1)
        self.conv_up3 =nn.Conv2d(320, 512, 1)
        self.maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.ca = CoordAtt(64, 64)
        self.W_x = nn.Conv2d(128, 128, 3, 1, 1)

        self.psi = nn.Sequential(
            nn.Conv2d(128, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.W_x2 = nn.Conv2d(64, 64, 3, 1, 1)

        self.psi2 = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        # self.ptem = PTFEM()
    def forward(self, x):

        # backbone
        pvt = self.backbone(x)
        x1 = pvt[0]
        x2 = pvt[1]
        x3 = pvt[2]
        x4 = pvt[3]
        x4_up = self.up(x4)
        x4_up = self.out4(x4_up)
        x4_up = self.up(x4_up)
        x4_up = self.out3(x4_up)
        x4_up = self.up(x4_up)
        x4_up = self.out2(x4_up)

        x3_up = self.up(x3)
        x3_up = self.out3(x3_up)
        x3_up = self.up(x3_up)
        x3_up = self.out2(x3_up)

        x2_up = self.up(x2)
        x2_up = self.out2(x2_up)


        e = self.eam(x1,x2_up,x4_up)
        t = self.tam(x1,x3_up,x4_up)


        p2_up = self.maxpool(e)
        p2_up = self.conv_up(p2_up)

        p2_up2 = self.maxpool(p2_up)
        p2_up2 = self.conv_up2(p2_up2)

        p2_up3 = self.maxpool(p2_up2)
        p2_up3 = self.conv_up3(p2_up3)
        #
        p3_up = self.maxpool(t)
        p3_up = self.conv_up(p3_up)

        p3_up2 = self.maxpool(p3_up)
        p3_up2 = self.conv_up2(p3_up2)

        p3_up3 = self.maxpool(p3_up2)
        p3_up3 = self.conv_up3(p3_up3)

        x1_fe = self.mccm1(x1)

        x1_fe = self.feem1(x1_fe,e,t)

        x2_fe = self.mccm2(x2)

        x2_fe = self.feem2(x2_fe, p2_up,  p3_up)

        x3_fe = self.mccm3(x3)

        x3_fe = self.feem3(x3_fe, p2_up2,p3_up2)

        x4_fe = self.mccm4(x4)

        x4_fe = self.feem4(x4_fe,  p2_up3, p3_up3)

        x4_out = x4_fe
        x4_out = self.ff4(x4_out)
        x4_out_up = self.up(x4_out)
        x4_out_up = self.out4(x4_out_up)
        x4_out_up2 = self.up(x4_out_up)
        x4_out_up2 = self.out3(x4_out_up2)
        x4_out_up3 = self.up(x4_out_up2)
        x4_out_up3 = self.out2(x4_out_up3)

        x3_out = self.ff3(x3_fe, x4_out_up)
        x3_out_up = self.up(x3_out)
        x3_out_up = self.out3(x3_out_up)
        x3_out_up2 = self.up(x3_out_up)
        x3_out_up2 = self.out2(x3_out_up2)

        x2_out = self.ff2(x2_fe, x3_out_up)
        x4_out_up2 = self.W_x(x4_out_up2)
        x2_out = self.psi(x2_out+x4_out_up2)*x4_out_up2+x2_out

        x2_out_up = self.up(x2_out)
        x2_out_up = self.out2(x2_out_up)

        x1_out = self.ff1(x1_fe, x2_out_up)
        x3_out_up2 = self.W_x2(x3_out_up2)
        x1_out = self.psi2(x1_out + x3_out_up2) * x3_out_up2 + x1_out
        x4_out_up3 = self.W_x2(x4_out_up3)
        x1_out = self.psi2(x1_out + x4_out_up3) * x4_out_up3 + x1_out

        x2_out_s= self.final_out4(x2_out)
        p3=self.up1(x2_out_s)
        #
        x3_out_s = self.final_out3(x3_out)
        p4 = self.up2(x3_out_s)

        x4_out_s = self.final_out2(x4_out)
        p5 = self.up4(x4_out_s)

        x1_out = self.up3(x1_out)

        # lsk = len(os.listdir("./outhot_my"))
        # if not os.path.exists("./outhot_my/" + str(lsk)):
        #     os.makedirs("./outhot_my/" + str(lsk))
        #     np.save("./outhot_my/" + str(lsk) + "/newp_1.npy", x1_out.detach().cpu().numpy())

        p1 = self.final_out(x1_out)

        e = self.final_out(e)

        e = self.up3(e)
        #
        #
        t = self.final_out(t)
        t = self.up3(t)

        return p1, e, t, p3, p4, p5



if __name__ == '__main__':
    model = SGNet().cuda()
    input_tensor = torch.randn(1, 3, 352, 352).cuda()

    prediction1, prediction2 = model(input_tensor)
    print(prediction1.size(), prediction2.size())
