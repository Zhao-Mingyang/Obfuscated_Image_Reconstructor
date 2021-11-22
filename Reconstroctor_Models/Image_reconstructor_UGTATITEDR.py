

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
                      nn.ReLU(True)]

        # Down-Sampling
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            DownBlock += [nn.ReflectionPad2d(1),
                          nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=0, bias=False),
                          nn.InstanceNorm2d(ngf * mult * 2),
                          nn.ReLU(True)]

        # Down-Sampling Bottleneck
        mult = 2**n_downsampling
        Comblocks = []
        for _ in range(5):
            block = ResnetBlock(256, use_bias=False)
            Comblocks.append(block)
        
        self.head = Kernalcom(256, 1)
        self.middle = nn.Sequential(*Comblocks)

        # Class Activation Map
        self.gap_fc = nn.Linear(ngf * mult, 1, bias=False)
        self.gmp_fc = nn.Linear(ngf * mult, 1, bias=False)
        self.conv1x1 = nn.Conv2d(ngf * mult * 2, ngf * mult, kernel_size=1, stride=1, bias=True)
        self.relu = nn.ReLU(True)

        # Gamma, Beta block
        if self.light:
            FC = [nn.Linear(ngf * mult, ngf * mult, bias=False),
                  nn.ReLU(True),
                  nn.Linear(ngf * mult, ngf * mult, bias=False),
                  nn.ReLU(True)]
        else:
            FC = [nn.Linear(img_size // mult * img_size // mult * ngf * mult, ngf * mult, bias=False),
                  nn.ReLU(True),
                  nn.Linear(ngf * mult, ngf * mult, bias=False),
                  nn.ReLU(True)]
        self.gamma = nn.Linear(ngf * mult, ngf * mult, bias=False)
        self.beta = nn.Linear(ngf * mult, ngf * mult, bias=False)

        # Up-Sampling Bottleneck
        for i in range(n_blocks):
            setattr(self, 'UpBlock1_' + str(i+1), ResnetAdaILNBlock(ngf * mult, use_bias=False))

        # Up-Sampling
        UpBlock2 = []
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            UpBlock2 += [nn.Upsample(scale_factor=2, mode='nearest'),
                         nn.ReflectionPad2d(1),
                         nn.Conv2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=1, padding=0, bias=False),
                         ILN(int(ngf * mult / 2)),
                         nn.ReLU(True)]

        UpBlock2 += [nn.ReflectionPad2d(3),
                     nn.Conv2d(ngf, output_nc, kernel_size=7, stride=1, padding=0, bias=False),
                     nn.Tanh()]

        self.DownBlock = nn.Sequential(*DownBlock)
        self.FC = nn.Sequential(*FC)
        self.UpBlock2 = nn.Sequential(*UpBlock2)
        self.InputComBlock = CombConv(256, 512,kernel_size=3,padding=1)
        self.ResComBlock = CombConv(512, 256,kernel_size=3,padding=1)
        self.discriminator = Discriminator_compare

    def forward(self, input):
        x = self.DownBlock(input)
        x = self.InputComBlock(x)
        x = self.head(x)
        x = self.middle(x)

        gap = torch.nn.functional.adaptive_avg_pool2d(x, 1)
        gap_weight = list(self.gap_fc.parameters())[0]
        gap = x * gap_weight.unsqueeze(2).unsqueeze(3)

        gmp = torch.nn.functional.adaptive_max_pool2d(x, 1)
        gmp_weight = list(self.gmp_fc.parameters())[0]
        gmp = x * gmp_weight.unsqueeze(2).unsqueeze(3)

        x = torch.cat([gap, gmp], 1)
        x = self.relu(self.conv1x1(x))


        if self.light:
            x_ = torch.nn.functional.adaptive_avg_pool2d(x, 1)
            x_ = self.FC(x_.view(x_.shape[0], -1))
        else:
            x_ = self.FC(x.view(x.shape[0], -1))
        gamma, beta = self.gamma(x_), self.beta(x_)


        for i in range(self.n_blocks):
            x = getattr(self, 'UpBlock1_' + str(i+1))(x, gamma, beta)
        out = self.UpBlock2(x)
        outclass = self.discriminator(out)
        return out, outclass


class ResnetBlock(nn.Module):
    def __init__(self, dim, use_bias):
        super(ResnetBlock, self).__init__()
        conv_block = []
        conv_block += [nn.ReflectionPad2d(1),
                       nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias),
                       nn.InstanceNorm2d(dim),
                       nn.ReLU(True)]

        conv_block += [nn.ReflectionPad2d(1),
                       nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias),
                       nn.InstanceNorm2d(dim)]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class ResnetAdaILNBlock(nn.Module):
    def __init__(self, dim, use_bias):
        super(ResnetAdaILNBlock, self).__init__()
        self.pad1 = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias)
        self.norm1 = adaILN(dim)
        self.relu1 = nn.ReLU(True)

        self.pad2 = nn.ReflectionPad2d(1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias)
        self.norm2 = adaILN(dim)

    def forward(self, x, gamma, beta):
        out = self.pad1(x)
        out = self.conv1(out)
        out = self.norm1(out, gamma, beta)
        out = self.relu1(out)
        out = self.pad2(out)
        out = self.conv2(out)
        out = self.norm2(out, gamma, beta)

        return out + x


class adaILN(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(adaILN, self).__init__()
        self.eps = eps
        self.rho = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.rho.data.fill_(0.9)

    def forward(self, input, gamma, beta):
        in_mean, in_var = torch.mean(input, dim=[2, 3], keepdim=True), torch.var(input, dim=[2, 3], keepdim=True)
        out_in = (input - in_mean) / torch.sqrt(in_var + self.eps)
        ln_mean, ln_var = torch.mean(input, dim=[1, 2, 3], keepdim=True), torch.var(input, dim=[1, 2, 3], keepdim=True)
        out_ln = (input - ln_mean) / torch.sqrt(ln_var + self.eps)
        out = self.rho.expand(input.shape[0], -1, -1, -1) * out_in + (1-self.rho.expand(input.shape[0], -1, -1, -1)) * out_ln
        out = out * gamma.unsqueeze(2).unsqueeze(3) + beta.unsqueeze(2).unsqueeze(3)

        return out


class ILN(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(ILN, self).__init__()
        self.eps = eps
        self.rho = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.gamma = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.beta = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.rho.data.fill_(0.0)
        self.gamma.data.fill_(1.0)
        self.beta.data.fill_(0.0)

    def forward(self, input):
        in_mean, in_var = torch.mean(input, dim=[2, 3], keepdim=True), torch.var(input, dim=[2, 3], keepdim=True)
        out_in = (input - in_mean) / torch.sqrt(in_var + self.eps)
        ln_mean, ln_var = torch.mean(input, dim=[1, 2, 3], keepdim=True), torch.var(input, dim=[1, 2, 3], keepdim=True)
        out_ln = (input - ln_mean) / torch.sqrt(ln_var + self.eps)
        out = self.rho.expand(input.shape[0], -1, -1, -1) * out_in + (1-self.rho.expand(input.shape[0], -1, -1, -1)) * out_ln
        out = out * self.gamma.expand(input.shape[0], -1, -1, -1) + self.beta.expand(input.shape[0], -1, -1, -1)

        return out
            


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
    
    
class Kernalcom(nn.Module):
    def __init__(self, dim, use_bias):
        super(Kernalcom, self).__init__()
        conv_block1 = []
        conv_block2 = []
        conv_block4 = []
        conv_block1 += [nn.ReflectionPad2d(2),
                       nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, dilation=1, bias=use_bias),
                       nn.InstanceNorm2d(dim),
                       nn.LeakyReLU(0.1)]

        conv_block1 += [nn.ReflectionPad2d(1),
                       nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, dilation=1, bias=use_bias),
                       nn.InstanceNorm2d(dim)]

        conv_block2 += [nn.ReflectionPad2d(1),
                       nn.Conv2d(dim, dim, kernel_size=2, stride=1, padding=0, dilation=1, bias=use_bias),
                       nn.InstanceNorm2d(dim),
                       nn.LeakyReLU(0.1)]

        conv_block2 += [nn.ReflectionPad2d(1),
                       nn.Conv2d(dim, dim, kernel_size=2, stride=1, padding=0, dilation=1, bias=use_bias),
                       nn.InstanceNorm2d(dim)]

        conv_block4 += [nn.ReflectionPad2d(2),
                       nn.Conv2d(dim, dim, kernel_size=4, stride=1, padding=0, dilation=1, bias=use_bias),
                       nn.InstanceNorm2d(dim),
                       nn.LeakyReLU(0.1)]

        conv_block4 += [nn.ReflectionPad2d(2),
                       nn.Conv2d(dim, dim, kernel_size=4, stride=1, padding=0, dilation=1, bias=use_bias),
                       nn.InstanceNorm2d(dim)]

        self.conv_block1 = nn.Sequential(*conv_block1)
        self.conv_block2 = nn.Sequential(*conv_block2)
        self.conv_block4 = nn.Sequential(*conv_block4)
        
        self.reducechannel = CombConv(dim*2, dim,kernel_size=3,padding=1)
        self.convchannel = CombConv(dim*3, dim,kernel_size=3,padding=0)

    def forward(self, x):
        inputt = x
        x = self.reducechannel(x)
#         x1 = self.conv_block1(x)
        x1 = self.conv_block1(x)
        x2 = self.conv_block2(x)
        x4 = self.conv_block4(x)
        x = torch.cat((x1,x2),1)
        out = torch.cat((x,x4),1)
        out = self.convchannel(out)
#         x = torch.cat((x,x5),1)
#         x = self.convchannel(x)
#         print(inputt.size())
#         print(x.size())

        return out

