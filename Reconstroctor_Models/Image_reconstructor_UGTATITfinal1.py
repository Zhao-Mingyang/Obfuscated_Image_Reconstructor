import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

class Encoder_Decoder_Residual(nn.Module):
    def __init__(self,Discriminator_compare,faceoutlinedetector, rs_blocks=8):
        super(Encoder_Decoder_Residual, self).__init__()

        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, padding=0),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.LeakyReLU(0.1),
#             nn.ReLU(True),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=6, stride=2, padding=0),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.LeakyReLU(0.1),
#             nn.ReLU(True),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256, track_running_stats=False),
            nn.LeakyReLU(0.1),
#             nn.ReLU(True)
        )

        head_blocks = []
        for _ in range(8):
            block = ResnetBlock(256, 2)
            head_blocks.append(block)
            
        self.head = nn.Sequential(*head_blocks)
        
        self.HeadComBlock = CombConv(512, 256)
        
        middle_blocks = []
        for _ in range(3):
            block = ResnetBlock(256, 1)
            middle_blocks.append(block)
          
        self.middle = nn.Sequential(*middle_blocks)
        
#         tail_blocks = []
#         for _ in range(3):
#             block = ResnetBlock(256, 1)
#             tail_blocks.append(block)
          
#         self.tail = nn.Sequential(*tail_blocks)
        
        self.gap_fc = nn.Linear(256, 1, bias=False)
        self.gmp_fc = nn.Linear(256, 1, bias=False)
        self.conv1x1 = nn.Conv2d(512, 256, kernel_size=1, stride=1, bias=True)
        self.relu = nn.ReLU(True)
        
        FC = [nn.Linear(230400, 256, bias=False),
                  nn.ReLU(True),
                  nn.Linear(256, 256, bias=False),
                  nn.ReLU(True)]
        self.FC = nn.Sequential(*FC)
        self.gamma = nn.Linear(256, 256, bias=False)
        self.beta = nn.Linear(256, 256, bias=False)

        # Up-Sampling Bottleneck
        for i in range(6):
            setattr(self, 'UpBlock1_' + str(i+1), ResnetAdaILNBlock(256, use_bias=False))
        

        self.decoder = nn.Sequential(

            # replace ConvTranspose2d with upsample + conv as in paper
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.ReflectionPad2d(1),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=1, padding=2),

            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.LeakyReLU(0.1),

            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=6, stride=2, padding=3),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.LeakyReLU(0.1),

            nn.ReflectionPad2d(3),
            nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=9, padding=3),
        )
        self.discriminator = Discriminator_compare

    def forward(self, x):
        x = self.encoder(x)
        inputt = x
        x_outline = self.head(x)
        out_outline = self.decoder(x_outline)
#         out_outline = grayscale_RGB(out_outline)
        out_outline = faceoutlinedetector(out_outline)
        x = x_outline
        x = torch.cat((inputt,x),1)
        x = self.HeadComBlock(x)
        x = self.middle(x)

        gap_weight = list(self.gap_fc.parameters())[0]
        gap = x * gap_weight.unsqueeze(2).unsqueeze(3)

        gmp_weight = list(self.gmp_fc.parameters())[0]
        gmp = x * gmp_weight.unsqueeze(2).unsqueeze(3)

        x = torch.cat([gap, gmp], 1)
        x = self.relu(self.conv1x1(x))
        
#         print(x.view(x.shape[0], -1).size())
        x_ = self.FC(x.view(x.shape[0], -1))
        gamma, beta = self.gamma(x_), self.beta(x_)


        for i in range(6):
            x = getattr(self, 'UpBlock1_' + str(i+1))(x, gamma, beta)
            
        
        x = self.decoder(x)
        out = (torch.tanh(x) + 1) / 2
        outclass = self.discriminator(out)

        return out, outclass, out_outline

    
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
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=dilation),
            nn.InstanceNorm2d(dim, track_running_stats=False),
            nn.LeakyReLU(0.2),

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

