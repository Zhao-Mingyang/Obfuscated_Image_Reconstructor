import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

class ResnetGenerator(nn.Module):
    def __init__(self,Discriminator_compare, input_nc=1, output_nc=1, ngf=64, n_blocks=6, img_size=128, light=False,
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
            
        encoder = []
        encoder += [nn.ReflectionPad2d(3),
                      nn.Conv2d(input_nc, ngf, kernel_size=7, stride=1, padding=0, bias=False),
                      nn.InstanceNorm2d(ngf),
                      nn.LeakyReLU(0.1)]

        for i in range(n_downsampling):
            if i == n_downsampling-1:
                stride = 2
            else:
                stride = 1
            mult = 2**i
            encoder += [nn.ReflectionPad2d(1),
                          nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=stride, padding=0, bias=False),
                          nn.InstanceNorm2d(ngf * mult * 2),
                          nn.LeakyReLU(0.1)]
            

        # Down-Sampling Bottleneck
        mult = 2**n_downsampling
        head_blocks = []
        for _ in range(4):
            block = ResnetBlock(256, 2)
            head_blocks.append(block)
            
        self.head = nn.Sequential(*head_blocks)
        self.HeadComBlock = CombConv(512, 256)
        
        middle_blocks = []
        for _ in range(6):
            block = ResnetBlock(256, 1)
            middle_blocks.append(block)
          
        self.middle = nn.Sequential(*middle_blocks)
        # Class Activation Map
        self.gap_fc = nn.Linear(ngf * mult, 1, bias=False)
        self.gmp_fc = nn.Linear(ngf * mult, 1, bias=False)
        self.conv1x1 = nn.Conv2d(ngf * mult * 2, ngf * mult, kernel_size=1, stride=1, bias=True)
        self.relu = nn.ReLU(True)
        
        self.down_gap_fc = nn.Linear(ngf * mult, 1, bias=False)
        self.down_gmp_fc = nn.Linear(ngf * mult, 1, bias=False)
        self.down_conv1x1 = nn.Conv2d(ngf * mult * 2, ngf * mult, kernel_size=1, stride=1, bias=True)

        # Gamma, Beta block
        FC = [nn.Linear(img_size // mult * img_size // mult * ngf * mult, ngf * mult, bias=False),
              nn.ReLU(True),
              nn.Linear(ngf * mult, ngf * mult, bias=False),
              nn.ReLU(True)]
            
        down_FC = [nn.Linear(img_size // mult * img_size // mult * ngf * mult, ngf * mult, bias=False),
                  nn.ReLU(True),
                  nn.Linear(ngf * mult, ngf * mult, bias=False),
                  nn.ReLU(True)]
        
        self.gamma = nn.Linear(ngf * mult, ngf * mult, bias=False)
        self.beta = nn.Linear(ngf * mult, ngf * mult, bias=False)
        
        self.down_gamma = nn.Linear(ngf * mult, ngf * mult, bias=False)
        self.down_beta = nn.Linear(ngf * mult, ngf * mult, bias=False)

        # Up-Sampling Bottleneck
        for i in range(n_blocks):
            setattr(self, 'UpBlock1_' + str(i+1), ResnetAdaILNBlock(ngf * mult, use_bias=False))

        for i in range(4):
            setattr(self, 'Down_UpBlock1_' + str(i+1), ResnetAdaILNBlock(ngf * mult, use_bias=False))    
        
        # Up-Sampling
        decoder = []
        for i in range(n_downsampling):
            if i == 0:
                stride = 2
            else:
                stride = 1
            mult = 2**(n_downsampling - i)
            decoder += [nn.Upsample(scale_factor=2, mode='nearest'),
                         nn.ReflectionPad2d(1),
                         nn.Conv2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=stride, padding=0, bias=False),
                         ILN(int(ngf * mult / 2)),
                         nn.ReLU(True)]

        decoder += [nn.ReflectionPad2d(3),
                     nn.Conv2d(ngf, output_nc, kernel_size=7, stride=1, padding=0, bias=False),
                     nn.Tanh()]
        
        
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
        self.decoder = nn.Sequential(*decoder)
        self.down_FC = nn.Sequential(*down_FC)
        self.FC = nn.Sequential(*FC)
        self.encoder = nn.Sequential(*encoder)
        self.UpBlock2 = nn.Sequential(*UpBlock2)
        
        self.discriminator = Discriminator_compare
 

    def forward(self, input):
        x_down = F.interpolate(input, scale_factor=0.5)

        x_down = self.encoder(x_down)
        x_down = self.head(x_down)

        x = self.DownBlock(input)
        inputt = x
        gap_weight_down = list(self.down_gap_fc.parameters())[0]
        gap_down = x_down * gap_weight_down.unsqueeze(2).unsqueeze(3)

        gmp_weight_down = list(self.down_gmp_fc.parameters())[0]
        gmp_down = x_down * gmp_weight_down.unsqueeze(2).unsqueeze(3)

        x_down = torch.cat([gap_down, gmp_down], 1)
        x_down = self.relu(self.down_conv1x1(x_down))


        x_down_ = self.down_FC(x_down.view(x_down.shape[0], -1))
        down_gamma, down_beta = self.down_gamma(x_down_), self.down_beta(x_down_)


        for i in range(4):
            x_down = getattr(self, 'Down_UpBlock1_' + str(i+1))(x_down, down_gamma, down_beta)     
        
        out_down = self.decoder(x_down)
#         out_outline = grayscale_RGB(out_outline)
#         out_outline = self.faceoutlinedetector(out_outline)
        x = torch.cat((x_down,inputt),1)
        x = self.HeadComBlock(x)
        x = self.middle(x)

        gap_weight = list(self.gap_fc.parameters())[0]
        gap = x * gap_weight.unsqueeze(2).unsqueeze(3)

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
        return out, outclass, out_down

    
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
    def __init__(self, dim, dilation=1, use_spectral_norm=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(dilation),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=dilation, bias=False),
            nn.InstanceNorm2d(dim, track_running_stats=False),
            nn.LeakyReLU(0.2),

            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=1, bias=False),
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

