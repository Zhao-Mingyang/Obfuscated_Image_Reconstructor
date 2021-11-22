
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

class ResnetGenerator(nn.Module):
    def __init__(self,Discriminator_compare,FaceoutlineDetector, input_nc=1, output_nc=1, ngf=64, n_blocks=6, img_size=128, light=False,
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
#         ResBlock = []
#         for i in range(n_blocks):
#             ResBlock += [ResnetBlock(ngf * mult, use_bias=False)]
        self.ResBlock_1 = ResnetBlock(128, use_bias=False)
        self.ResBlock_2 = ResnetBlock(128, use_bias=False)
        self.ResBlock_3 = ResnetBlock(128, use_bias=False)
        self.ResBlock_4 = ResnetBlock(128, use_bias=False)
#         self.ResBlock_5 = ResnetBlock(128, use_bias=False)
        # Class Activation Map
        self.gap_fc = nn.Linear(ngf * mult, 1, bias=False)
        self.gmp_fc = nn.Linear(ngf * mult, 1, bias=False)
        self.conv1x1 = nn.Conv2d(ngf * mult * 2, ngf * mult, kernel_size=1, stride=1, bias=True)
        self.relu = nn.LeakyReLU(0.1)

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
#         self.ResBlock = nn.Sequential(*ResBlock)
        self.FC = nn.Sequential(*FC)
        self.UpBlock2 = nn.Sequential(*UpBlock2)
        
        self.discriminator = Discriminator_compare
        self.outlineDetector = FaceoutlineDetector
        
        blocks = []
        for _ in range(5):
            block = ResnetBlock2(256, 1)
            blocks.append(block)
        
        self.middle = nn.Sequential(*blocks)
        
        self.InputComBlock = CombConv(512, 128)
        self.ResComBlock = CombConv(256, 128)
        self.MiddleComBlock_1 = CombConv(256, 128)
        self.MiddleComBlock_2 = CombConv(256, 128)
        self.MiddleComBlock_3 = CombConv(256, 128)
#         self.MiddleComBlock_4 = CombConv(256, 128)
        self.MiddleComBlock_4 = CombConv(256, 256)

        
        
    def forward(self, input):
        x = self.DownBlock(input)
        x_res = self.middle(x)
        dect_out1 = self.UpBlock2(x_res)
        dect_out = grayscale_RGB(dect_out1)
        dect_out = self.outlineDetector(dect_out)
        x = torch.cat((x_res,x),1)
        x = self.InputComBlock(x)
        x = self.ResBlock_1(x)
        x_res = self.ResComBlock(x_res)
        x = torch.cat((x_res,x),1)
        x = self.MiddleComBlock_1(x)
        x = self.ResBlock_2(x)
        x = torch.cat((x_res,x),1)
        x = self.MiddleComBlock_2(x)
        x = self.ResBlock_3(x)
        x = torch.cat((x_res,x),1)
        x = self.MiddleComBlock_3(x)
        x = self.ResBlock_4(x)
        x = torch.cat((x_res,x),1)
        x = self.MiddleComBlock_4(x)
#         x = self.ResBlock_5(x)
#         x = torch.cat((x_res,x),1)
#         x = self.MiddleComBlock_5(x)
#         print(x.size())
        
        
#         print(x_res.size())
        
#         print(x_skip.size())
        
#         print(x.size())
#         x = self.ResBlock(x)
        gap = torch.nn.functional.adaptive_avg_pool2d(x, 1)
        gap_logit = self.gap_fc(gap.view(x.shape[0], -1))
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
        return out, outclass, dect_out, dect_out1

class CombConv(nn.Module):
    def __init__(self, dim1, dim2):
        super(CombConv, self).__init__()
        ComBlock = []
        ComBlock += [nn.Conv2d(dim1, dim2, kernel_size=3, stride=1, padding=1, bias=False),
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
        conv_block3 = []
        conv_block5 = []
        conv_block1 += [nn.ReflectionPad2d(1),
                       nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, dilation=1, bias=use_bias),
                       nn.InstanceNorm2d(dim),
                       nn.LeakyReLU(0.1)]

        conv_block1 += [nn.ReflectionPad2d(1),
                       nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, dilation=1, bias=use_bias),
                       nn.InstanceNorm2d(dim)]

        conv_block3 += [nn.ReflectionPad2d(3),
                       nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, dilation=3, bias=use_bias),
                       nn.InstanceNorm2d(dim),
                       nn.LeakyReLU(0.1)]

        conv_block3 += [nn.ReflectionPad2d(3),
                       nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, dilation=3, bias=use_bias),
                       nn.InstanceNorm2d(dim)]

        conv_block5 += [nn.ReflectionPad2d(5),
                       nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, dilation=5, bias=use_bias),
                       nn.InstanceNorm2d(dim),
                       nn.LeakyReLU(0.1)]

        conv_block5 += [nn.ReflectionPad2d(5),
                       nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, dilation=5, bias=use_bias),
                       nn.InstanceNorm2d(dim)]

        self.conv_block1 = nn.Sequential(*conv_block1)
        self.conv_block3 = nn.Sequential(*conv_block3)
        self.conv_block5 = nn.Sequential(*conv_block5)
        
        self.convlayer = nn.Conv2d(dim*3, dim, kernel_size=3, stride=1, padding=1, dilation=1, bias=use_bias)

    def forward(self, x):
        inputt = x
        x1 = self.conv_block1(x)
        x3 = self.conv_block3(x)
        x5 = self.conv_block5(x)
        x = torch.cat((x1,x3),1)
        x = torch.cat((x,x5),1)
        x = self.convlayer(x)
#         print(inputt.size())
#         print(x.size())
        out = inputt + x
        return out


class ResnetAdaILNBlock(nn.Module):
    def __init__(self, dim, use_bias):
        super(ResnetAdaILNBlock, self).__init__()
        self.pad1 = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias)
        self.norm1 = adaILN(dim)
        self.relu1 = nn.LeakyReLU(0.1)

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



class RhoClipper(object):

    def __init__(self, min, max):
        self.clip_min = min
        self.clip_max = max
        assert min < max

    def __call__(self, module):

        if hasattr(module, 'rho'):
            w = module.rho.data
            w = w.clamp(self.clip_min, self.clip_max)
            module.rho.data = w
            
class ResnetBlock2(nn.Module):
    def __init__(self, dim, dilation=1, use_spectral_norm=False):
        super(ResnetBlock2, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=1),
            nn.InstanceNorm2d(dim, track_running_stats=False),
            nn.LeakyReLU(0.15),

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

        # Remove ReLU at the end of the residual block
        # http://torch.ch/blog/2016/02/04/resnets.html

        return output