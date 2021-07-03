# coding: utf-8
import torch
from torch.nn import init
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

'''
ENet: A Deep Neural Network Architecture for Real-Time Semantic Segmentation
Refence: https://arxiv.org/pdf/1606.02147.pdf

Code is written by Iroh Cao
'''

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

class InitialBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(InitialBlock, self).__init__()
        self.input_channel = in_ch
        self.conv_channel = out_ch - in_ch

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch - in_ch, kernel_size = 3, stride = 2, padding=1),
            nn.BatchNorm2d(out_ch - in_ch),
            nn.PReLU()
        )
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        conv_branch = self.conv(x)
        maxp_branch = self.maxpool(x)
        return torch.cat([conv_branch, maxp_branch], 1)

class BottleneckModule(nn.Module):
    def __init__(self, in_ch, out_ch, module_type, padding = 1, dilated = 0, asymmetric = 5, dropout_prob = 0):
        super(BottleneckModule, self).__init__()
        self.input_channel = in_ch
        self.activate = nn.PReLU()

        self.module_type = module_type
        if self.module_type == 'downsampling':
            self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size = 2, stride = 2),
                nn.BatchNorm2d(out_ch),
                nn.PReLU(),
                nn.Conv2d(out_ch, out_ch, kernel_size = 3, stride=1, padding=padding),
                nn.BatchNorm2d(out_ch),
                nn.PReLU(),
                nn.Conv2d(out_ch, out_ch, kernel_size = 1),
                nn.BatchNorm2d(out_ch),
                nn.PReLU(),
                nn.Dropout2d(p=dropout_prob)
            )
        elif self.module_type == 'upsampling':
            self.maxunpool = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size = 1),
                nn.BatchNorm2d(out_ch),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)    # Use upsample instead of maxunpooling
            )
            
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size = 1),
                nn.BatchNorm2d(out_ch),
                nn.PReLU(),
                nn.ConvTranspose2d(out_ch, out_ch, kernel_size=2, stride=2),
                nn.BatchNorm2d(out_ch),
                nn.PReLU(),
                nn.Conv2d(out_ch, out_ch, kernel_size = 1),
                nn.BatchNorm2d(out_ch),
                nn.PReLU(),
                nn.Dropout2d(p=dropout_prob)
            )
        elif self.module_type == 'regular':
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size = 1),
                nn.BatchNorm2d(out_ch),
                nn.PReLU(),
                nn.Conv2d(out_ch, out_ch, kernel_size = 3, stride=1, padding=padding),
                nn.BatchNorm2d(out_ch),
                nn.PReLU(),
                nn.Conv2d(out_ch, out_ch, kernel_size = 1),
                nn.BatchNorm2d(out_ch),
                nn.PReLU(),
                nn.Dropout2d(p=dropout_prob)
            )
        elif self.module_type == 'asymmetric':
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size = 1),
                nn.BatchNorm2d(out_ch),
                nn.PReLU(),
                nn.Conv2d(out_ch, out_ch, (asymmetric, 1), stride=1, padding=(padding, 0)),
                nn.Conv2d(out_ch, out_ch, (1, asymmetric), stride=1, padding=(0, padding)),
                nn.BatchNorm2d(out_ch),
                nn.PReLU(),
                nn.Conv2d(out_ch, out_ch, kernel_size = 1),
                nn.BatchNorm2d(out_ch),
                nn.PReLU(),
                nn.Dropout2d(p=dropout_prob)
            )
        elif self.module_type == 'dilated':
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size = 1),
                nn.BatchNorm2d(out_ch),
                nn.PReLU(),
                nn.Conv2d(out_ch, out_ch, kernel_size = 3, stride=1, padding=padding, dilation=dilated),
                nn.BatchNorm2d(out_ch),
                nn.PReLU(),
                nn.Conv2d(out_ch, out_ch, kernel_size = 1),
                nn.BatchNorm2d(out_ch),
                nn.PReLU(),
                nn.Dropout2d(p=dropout_prob)
            )
        else:
            raise("Module Type error")

    def forward(self, x):
        if self.module_type == 'downsampling':
            conv_branch = self.conv(x)
            maxp_branch = self.maxpool(x)
            bs, conv_ch, h, w = conv_branch.size()
            maxp_ch = maxp_branch.size()[1]
            padding = torch.zeros(bs, conv_ch - maxp_ch, h, w).to(DEVICE)

            maxp_branch = torch.cat([maxp_branch, padding], 1).to(DEVICE)
            output = maxp_branch + conv_branch
        elif self.module_type == 'upsampling':
            conv_branch = self.conv(x)
            maxunp_branch = self.maxunpool(x)
            output = maxunp_branch + conv_branch
        else:
            output = self.conv(x) + x
        
        return self.activate(output)

class ENet_Encoder(nn.Module):
    
    def __init__(self, in_ch=3, dropout_prob=0):
        super(ENet_Encoder, self).__init__()

        # Encoder

        self.initial_block = InitialBlock(in_ch, 16)

        self.bottleneck1_0 = BottleneckModule(16, 64, module_type = 'downsampling', padding = 1, dropout_prob = dropout_prob)
        self.bottleneck1_1 = BottleneckModule(64, 64, module_type = 'regular', padding = 1, dropout_prob = dropout_prob)
        self.bottleneck1_2 = BottleneckModule(64, 64, module_type = 'regular', padding = 1, dropout_prob = dropout_prob)
        self.bottleneck1_3 = BottleneckModule(64, 64, module_type = 'regular', padding = 1, dropout_prob = dropout_prob)
        self.bottleneck1_4 = BottleneckModule(64, 64, module_type = 'regular', padding = 1, dropout_prob = dropout_prob)

        self.bottleneck2_0 = BottleneckModule(64, 128, module_type = 'downsampling', padding = 1, dropout_prob = dropout_prob)
        self.bottleneck2_1 = BottleneckModule(128, 128, module_type = 'regular', padding = 1, dropout_prob = dropout_prob)
        self.bottleneck2_2 = BottleneckModule(128, 128, module_type = 'dilated', padding = 2, dilated = 2, dropout_prob = dropout_prob)
        self.bottleneck2_3 = BottleneckModule(128, 128, module_type = 'asymmetric', padding = 2, asymmetric=5, dropout_prob = dropout_prob)
        self.bottleneck2_4 = BottleneckModule(128, 128, module_type = 'dilated', padding = 4, dilated = 4, dropout_prob = dropout_prob)
        self.bottleneck2_5 = BottleneckModule(128, 128, module_type = 'regular', padding = 1, dropout_prob = dropout_prob)
        self.bottleneck2_6 = BottleneckModule(128, 128, module_type = 'dilated', padding = 8, dilated = 8, dropout_prob = dropout_prob)
        self.bottleneck2_7 = BottleneckModule(128, 128, module_type = 'asymmetric', padding = 2, asymmetric=5, dropout_prob = dropout_prob)
        self.bottleneck2_8 = BottleneckModule(128, 128, module_type = 'dilated', padding = 16, dilated = 16, dropout_prob = dropout_prob)

        self.bottleneck3_0 = BottleneckModule(128, 128, module_type = 'regular', padding = 1, dropout_prob = dropout_prob)
        self.bottleneck3_1 = BottleneckModule(128, 128, module_type = 'dilated', padding = 2, dilated = 2, dropout_prob = dropout_prob)
        self.bottleneck3_2 = BottleneckModule(128, 128, module_type = 'asymmetric', padding = 2, asymmetric=5, dropout_prob = dropout_prob)
        self.bottleneck3_3 = BottleneckModule(128, 128, module_type = 'dilated', padding = 4, dilated = 4, dropout_prob = dropout_prob)
        self.bottleneck3_4 = BottleneckModule(128, 128, module_type = 'regular', padding = 1, dropout_prob = dropout_prob)
        self.bottleneck3_5 = BottleneckModule(128, 128, module_type = 'dilated', padding = 8, dilated = 8, dropout_prob = dropout_prob)
        self.bottleneck3_6 = BottleneckModule(128, 128, module_type = 'asymmetric', padding = 2, asymmetric=5, dropout_prob = dropout_prob)
        self.bottleneck3_7 = BottleneckModule(128, 128, module_type = 'dilated', padding = 16, dilated = 16, dropout_prob = dropout_prob)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                weights_init_kaiming(m)
            elif isinstance(m, nn.BatchNorm2d):
                weights_init_kaiming(m)
    
    def forward(self, x):
        x = self.initial_block(x)

        x = self.bottleneck1_0(x)
        x = self.bottleneck1_1(x)
        x = self.bottleneck1_2(x)
        x = self.bottleneck1_3(x)
        x = self.bottleneck1_4(x)

        x = self.bottleneck2_0(x)
        x = self.bottleneck2_1(x)
        x = self.bottleneck2_2(x)
        x = self.bottleneck2_3(x)
        x = self.bottleneck2_4(x)
        x = self.bottleneck2_5(x)
        x = self.bottleneck2_6(x)
        x = self.bottleneck2_7(x)
        x = self.bottleneck2_8(x)

        x = self.bottleneck3_0(x)
        x = self.bottleneck3_1(x)
        x = self.bottleneck3_2(x)
        x = self.bottleneck3_3(x)
        x = self.bottleneck3_4(x)
        x = self.bottleneck3_5(x)
        x = self.bottleneck3_6(x)
        x = self.bottleneck3_7(x)

        return x


class ENet_Decoder(nn.Module):
    
    def __init__(self, out_ch=1, dropout_prob=0):
        super(ENet_Decoder, self).__init__()


        self.bottleneck4_0 = BottleneckModule(128, 64, module_type = 'upsampling', padding = 1, dropout_prob = dropout_prob)
        self.bottleneck4_1 = BottleneckModule(64, 64, module_type = 'regular', padding = 1, dropout_prob = dropout_prob)
        self.bottleneck4_2 = BottleneckModule(64, 64, module_type = 'regular', padding = 1, dropout_prob = dropout_prob)

        self.bottleneck5_0 = BottleneckModule(64, 16, module_type = 'upsampling', padding = 1, dropout_prob = dropout_prob)
        self.bottleneck5_1 = BottleneckModule(16, 16, module_type = 'regular', padding = 1, dropout_prob = dropout_prob)

        self.fullconv = nn.ConvTranspose2d(16, out_ch, kernel_size=2, stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                weights_init_kaiming(m)
            elif isinstance(m, nn.BatchNorm2d):
                weights_init_kaiming(m)
    
    def forward(self, x):

        x = self.bottleneck4_0(x)
        x = self.bottleneck4_1(x)
        x = self.bottleneck4_2(x)

        x = self.bottleneck5_0(x)
        x = self.bottleneck5_1(x)

        x = self.fullconv(x)

        return x


class ENet(nn.Module):
    
    def __init__(self, in_ch=3, out_ch=1):
        super(ENet, self).__init__()

        # Encoder

        self.encoder = ENet_Encoder(in_ch)

        # self.initial_block = InitialBlock(in_ch, 16)

        # self.bottleneck1_0 = BottleneckModule(16, 64, module_type = 'downsampling', padding = 1, dropout_prob = 0.1)
        # self.bottleneck1_1 = BottleneckModule(64, 64, module_type = 'regular', padding = 1, dropout_prob = 0.1)
        # self.bottleneck1_2 = BottleneckModule(64, 64, module_type = 'regular', padding = 1, dropout_prob = 0.1)
        # self.bottleneck1_3 = BottleneckModule(64, 64, module_type = 'regular', padding = 1, dropout_prob = 0.1)
        # self.bottleneck1_4 = BottleneckModule(64, 64, module_type = 'regular', padding = 1, dropout_prob = 0.1)

        # self.bottleneck2_0 = BottleneckModule(64, 128, module_type = 'downsampling', padding = 1, dropout_prob = 0.1)
        # self.bottleneck2_1 = BottleneckModule(128, 128, module_type = 'regular', padding = 1, dropout_prob = 0.1)
        # self.bottleneck2_2 = BottleneckModule(128, 128, module_type = 'dilated', padding = 2, dilated = 2, dropout_prob = 0.1)
        # self.bottleneck2_3 = BottleneckModule(128, 128, module_type = 'asymmetric', padding = 2, asymmetric=5, dropout_prob = 0.1)
        # self.bottleneck2_4 = BottleneckModule(128, 128, module_type = 'dilated', padding = 4, dilated = 4, dropout_prob = 0.1)
        # self.bottleneck2_5 = BottleneckModule(128, 128, module_type = 'regular', padding = 1, dropout_prob = 0.1)
        # self.bottleneck2_6 = BottleneckModule(128, 128, module_type = 'dilated', padding = 8, dilated = 8, dropout_prob = 0.1)
        # self.bottleneck2_7 = BottleneckModule(128, 128, module_type = 'asymmetric', padding = 2, asymmetric=5, dropout_prob = 0.1)
        # self.bottleneck2_8 = BottleneckModule(128, 128, module_type = 'dilated', padding = 16, dilated = 16, dropout_prob = 0.1)

        # self.bottleneck3_0 = BottleneckModule(128, 128, module_type = 'regular', padding = 1, dropout_prob = 0.1)
        # self.bottleneck3_1 = BottleneckModule(128, 128, module_type = 'dilated', padding = 2, dilated = 2, dropout_prob = 0.1)
        # self.bottleneck3_2 = BottleneckModule(128, 128, module_type = 'asymmetric', padding = 2, asymmetric=5, dropout_prob = 0.1)
        # self.bottleneck3_3 = BottleneckModule(128, 128, module_type = 'dilated', padding = 4, dilated = 4, dropout_prob = 0.1)
        # self.bottleneck3_4 = BottleneckModule(128, 128, module_type = 'regular', padding = 1, dropout_prob = 0.1)
        # self.bottleneck3_5 = BottleneckModule(128, 128, module_type = 'dilated', padding = 8, dilated = 8, dropout_prob = 0.1)
        # self.bottleneck3_6 = BottleneckModule(128, 128, module_type = 'asymmetric', padding = 2, asymmetric=5, dropout_prob = 0.1)
        # self.bottleneck3_7 = BottleneckModule(128, 128, module_type = 'dilated', padding = 16, dilated = 16, dropout_prob = 0.1)

        # Decoder

        self.decoder = ENet_Decoder(out_ch)

        # self.bottleneck4_0 = BottleneckModule(128, 64, module_type = 'upsampling', padding = 1, dropout_prob = 0.1)
        # self.bottleneck4_1 = BottleneckModule(64, 64, module_type = 'regular', padding = 1, dropout_prob = 0.1)
        # self.bottleneck4_2 = BottleneckModule(64, 64, module_type = 'regular', padding = 1, dropout_prob = 0.1)

        # self.bottleneck5_0 = BottleneckModule(64, 16, module_type = 'upsampling', padding = 1, dropout_prob = 0.1)
        # self.bottleneck5_1 = BottleneckModule(16, 16, module_type = 'regular', padding = 1, dropout_prob = 0.1)

        # self.fullconv = nn.ConvTranspose2d(16, out_ch, kernel_size=2, stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                weights_init_kaiming(m)
            elif isinstance(m, nn.BatchNorm2d):
                weights_init_kaiming(m)
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)


        # x = self.initial_block(x)

        # x = self.bottleneck1_0(x)
        # x = self.bottleneck1_1(x)
        # x = self.bottleneck1_2(x)
        # x = self.bottleneck1_3(x)
        # x = self.bottleneck1_4(x)

        # x = self.bottleneck2_0(x)
        # x = self.bottleneck2_1(x)
        # x = self.bottleneck2_2(x)
        # x = self.bottleneck2_3(x)
        # x = self.bottleneck2_4(x)
        # x = self.bottleneck2_5(x)
        # x = self.bottleneck2_6(x)
        # x = self.bottleneck2_7(x)
        # x = self.bottleneck2_8(x)

        # x = self.bottleneck3_0(x)
        # x = self.bottleneck3_1(x)
        # x = self.bottleneck3_2(x)
        # x = self.bottleneck3_3(x)
        # x = self.bottleneck3_4(x)
        # x = self.bottleneck3_5(x)
        # x = self.bottleneck3_6(x)
        # x = self.bottleneck3_7(x)

        # x = self.bottleneck4_0(x)
        # x = self.bottleneck4_1(x)
        # x = self.bottleneck4_2(x)

        # x = self.bottleneck5_0(x)
        # x = self.bottleneck5_1(x)

        # x = self.fullconv(x)

        return x

#########################################################################
'''
============================================================================
Test the module type.
============================================================================

'''

if __name__ == "__main__":
    input_var = Variable(torch.randn(5, 3, 512, 512))
    # model = BottleneckModule(128, 64, module_type = 'upsampling', padding = 2, dilated = 2, asymmetric = 5, dropout_prob = 0.1)
    model = ENet(3, 2)
    print(model)
    output = model(input_var)
    # print(output)
    print(output.shape)