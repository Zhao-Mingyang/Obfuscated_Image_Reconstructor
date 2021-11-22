import torch
import torch.nn as nn


import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

class ResnetGenerator(nn.Module):
    def __init__(self,Discriminator_compare, input_nc=3, output_nc=3, ngf=64, n_blocks=6, img_size=128, light=False,
                ):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.n_blocks = n_blocks
        self.img_size = img_size
        self.light = light

        DownBlock = []
        DownBlock += [nn.ReflectionPad2d(3),
                      nn.Conv2d(input_nc, ngf, kernel_size=7, stride=1, padding=0, bias=False),
                      nn.InstanceNorm2d(ngf),
                      nn.LeakyReLU(0.1)]

        # Down-Sampling
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            DownBlock += [nn.ReflectionPad2d(1),
                          nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=0, bias=False),
                          nn.InstanceNorm2d(ngf * mult * 2),
                          nn.LeakyReLU(0.1)]

#         # Down-Sampling Bottleneck
        mult = 2**n_downsampling
        # Class Activation Map
        
        self.attention_layer_max = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(256, 256, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=1),
            nn.LeakyReLU(0.15)
        )
        self.attention_layer_avg = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(256, 256, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=1),
            nn.LeakyReLU(0.15)
        )
#         self.gmp_fc = nn.Linear(ngf * mult, 1, bias=False)
        self.conv1x1 = nn.Conv2d(ngf * mult * 2, ngf * mult, kernel_size=1, stride=1, bias=True)
        self.relu = nn.LeakyReLU(0.1)

        # Up-Sampling
        UpBlock2 = []
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            UpBlock2 += [nn.Upsample(scale_factor=2, mode='nearest'),
                         nn.ReflectionPad2d(1),
                         nn.Conv2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=1, padding=0, bias=False),
                         nn.InstanceNorm2d(int(ngf * mult / 2)),
                         nn.ReLU]

        UpBlock2 += [nn.ReflectionPad2d(3),
                     nn.Conv2d(ngf, output_nc, kernel_size=7, stride=1, padding=0, bias=False),
                     nn.ReLU]
        
        self.gap_fc = nn.Linear(256, 1, bias=False)
        self.gmp_fc = nn.Linear(256, 1, bias=False)
        self.DownBlock = nn.Sequential(*DownBlock)
#         self.ResBlock = nn.Sequential(*ResBlock)
#         self.FC = nn.Sequential(*FC)
        self.UpBlock2 = nn.Sequential(*UpBlock2)
        
        self.discriminator = Discriminator_compare
        
        Comblocks = []
        for _ in range(5):
            block = ResnetBlock(256, 1)
            Comblocks.append(block)
        
        self.head = nn.Sequential(*Comblocks)
        
#         Deliblocks = []
#         for _ in range(5):
#             block = ResnetBlock(128, use_bias=False)
#             Deliblocks.append(block)
            
#         self.Delimiddle = nn.Sequential(*Deliblocks)
        
        UpResblocks = []
        for _ in range(3):
            block = ResnetBlock2(256, 1)
            UpResblocks.append(block)
        self.tail = nn.Sequential(*UpResblocks)
        
        self.InputComBlock = CombConv(256, 512,kernel_size=3,padding=1)
        self.ResComBlock = CombConv(512, 256,kernel_size=3,padding=1)
       
        
        
    def forward(self, input):
        x = self.DownBlock(input)
        x = self.InputComBlock(x)
        x_res = self.head(x)
#         x = torch.cat((x_res,x),1)

#         x = self.Delimiddle(x)
#         x = torch.cat((x_res,x),1)
        x = self.ResComBlock(x)
        gap = torch.nn.functional.adaptive_avg_pool2d(x, 1)
        gap_weight = list(self.gap_fc.parameters())[0]
        gap = x * gap_weight.unsqueeze(2).unsqueeze(3)

        gmp = torch.nn.functional.adaptive_max_pool2d(x, 1)
        gmp_weight = list(self.gmp_fc.parameters())[0]
        gmp = x * gmp_weight.unsqueeze(2).unsqueeze(3)

        x = torch.cat([gap, gmp], 1)
        x = self.relu(self.conv1x1(x))
#         gap_weight = self.attention_layer_avg(x)
#         gap = gap_weight * x


#         gmp_weight = self.attention_layer_max(x)
#         gmp = gmp_weight * x

        x = torch.cat([gap, gmp], 1)
        x = self.relu(self.conv1x1(x))

        x = self.tail(x)
        out = self.UpBlock2(x)
        outclass = self.discriminator(out)
        return out, outclass

class CombConv(nn.Module):
    def __init__(self, dim1, dim2, kernel_size=3, padding=1):
        super(CombConv, self).__init__()
        ComBlock = []
        ComBlock += [nn.Conv2d(dim1, dim2, kernel_size=kernel_size, stride=1, padding=padding, bias=False),
                      nn.InstanceNorm2d(dim2),
                      nn.LeakyReLU(0.1)]
        self.ComBlock = nn.Sequential(*ComBlock)
        
        
    def forward(self, x):
        out = self.ComBlock(x)
        return out
    
    
class ResnetBlock(nn.Module):
    def __init__(self, dim, use_bias):
        super(ResnetBlock, self).__init__()
        conv_block1 = []
        conv_block2 = []
        conv_block4 = []
        conv_block1 += [nn.ReflectionPad2d(2),
                       nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=0, dilation=2, bias=use_bias),
                       nn.InstanceNorm2d(dim),
                       nn.LeakyReLU(0.1)]

        conv_block1 += [nn.ReflectionPad2d(1),
                       nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=0, dilation=1, bias=use_bias),
                       nn.InstanceNorm2d(dim)]

        conv_block2 += [nn.ReflectionPad2d(1),
                       nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, dilation=1, bias=use_bias),
                       nn.InstanceNorm2d(dim),
                       nn.LeakyReLU(0.1)]

        conv_block2 += [nn.ReflectionPad2d(3),
                       nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, dilation=3, bias=use_bias),
                       nn.InstanceNorm2d(dim)]

        conv_block4 += [nn.ReflectionPad2d(3),
                       nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, dilation=3, bias=use_bias),
                       nn.InstanceNorm2d(dim),
                       nn.LeakyReLU(0.1)]

        conv_block4 += [nn.ReflectionPad2d(1),
                       nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, dilation=1, bias=use_bias),
                       nn.InstanceNorm2d(dim)]

        self.conv_block1 = nn.Sequential(*conv_block1)
        self.conv_block2 = nn.Sequential(*conv_block2)
        self.conv_block4 = nn.Sequential(*conv_block4)
        
        self.reducechannel = CombConv(dim*2, dim,kernel_size=3,padding=1)
#         self.convchannel = CombConv(dim*2, dim*2,kernel_size=3,padding=0)

    def forward(self, x):
        inputt = x
        x = self.reducechannel(x)
#         x1 = self.conv_block1(x)
        x2 = self.conv_block2(x)
        x4 = self.conv_block4(x)
        x = torch.cat((x2,x4),1)
#         x = torch.cat((x,x5),1)
#         x = self.convchannel(x)
#         print(inputt.size())
#         print(x.size())
        x = inputt + x
        out = self.conv_block1(x) + x
        return out



            
class ResnetBlock2(nn.Module):
    def __init__(self, dim, dilation=1, use_spectral_norm=False):
        super(ResnetBlock2, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=1),
            nn.InstanceNorm2d(dim, track_running_stats=False),
            nn.ReLU

            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=1),
            nn.InstanceNorm2d(dim, track_running_stats=False),
#             nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, padding=0),
#             nn.Sigmoid()
        )


    def forward(self, x):
        output = self.conv_block(x)
#         output = output*x+x

        output = output+x

        return output