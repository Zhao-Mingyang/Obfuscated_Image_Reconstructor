import torch.nn as nn
import torch

class en_deconv_ressuper(nn.Module):
    def __init__(self):
        super(en_deconv_ressuper, self).__init__()
        
        self.leakyrelu =nn.LeakyReLU(0.2)
        self.ReflectionPad2d=nn.ReflectionPad2d(3)
        self.d_conv_2 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=2, padding=2)
        self.d_bn_2 = nn.BatchNorm2d(16)

        self.d_conv_3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=2)
        self.d_bn_3 = nn.BatchNorm2d(32)
        self.s_conv_3 = nn.Conv2d(in_channels=32, out_channels=4, kernel_size=5, stride=1, padding=2)

        self.d_conv_4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2)
        self.d_bn_4 = nn.BatchNorm2d(64)
        self.s_conv_4 = nn.Conv2d(in_channels=64, out_channels=4, kernel_size=5, stride=1, padding=2)

        self.d_conv_5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2)
        self.d_bn_5 = nn.BatchNorm2d(128)
        self.s_conv_5 = nn.Conv2d(in_channels=128, out_channels=4, kernel_size=5, stride=1, padding=2)

        self.d_conv_6 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=2)
        self.d_bn_6 = nn.BatchNorm2d(256)

        self.u_deconv_5 = nn.ConvTranspose2d(in_channels=256, out_channels=124, kernel_size=4, stride=2, padding=1)
        self.u_bn_5 = nn.BatchNorm2d(128)

        self.u_deconv_4 = nn.ConvTranspose2d(in_channels=128, out_channels=60, kernel_size=4, stride=2, padding=1)
        self.u_bn_4 = nn.BatchNorm2d(64)

        self.u_deconv_3 = nn.ConvTranspose2d(in_channels=64, out_channels=28, kernel_size=4, stride=2, padding=1)
        self.u_bn_3 = nn.BatchNorm2d(32)

        self.u_deconv_2 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=1)
        self.u_bn_2 = nn.BatchNorm2d(16)

        self.u_deconv_1 = nn.ConvTranspose2d(in_channels=16, out_channels=3, kernel_size=4, stride=2, padding=1)
        self.u_bn_1 = nn.BatchNorm2d(3)

        blocks = []
        for i in range(8):
            if (i>1)&(i<7):
                block = ResnetBlock(256, 2)
                blocks.append(block)
            elif i==7:
                block = ResnetBlock(256, 3)
                blocks.append(block)
            else:
                block = ResnetBlock(256, 1)
                blocks.append(block)                

        self.middle = nn.Sequential(*blocks)
        
    def forward(self, noise):

        down_2 = self.d_conv_2(noise)
        down_2 = self.d_bn_2(down_2)
        down_2 = self.leakyrelu(down_2)
#         print(down_2.size())
        down_3 = self.d_conv_3(down_2)
        down_3 = self.d_bn_3(down_3)
        down_3 = self.leakyrelu(down_3)
        skip_3 = self.s_conv_3(down_3)
#         print(down_3.size())
        down_4 = self.d_conv_4(down_3)
        down_4 = self.d_bn_4(down_4)
        down_4 = self.leakyrelu(down_4)
        skip_4 = self.s_conv_4(down_4)
#         print(down_4.size())
        down_5 = self.d_conv_5(down_4)
        down_5 = self.d_bn_5(down_5)
        down_5 = self.leakyrelu(down_5)
        skip_5 = self.s_conv_5(down_5)
#         print(down_5.size())
        down_6 = self.d_conv_6(down_5)
        down_6 = self.d_bn_6(down_6)
        down_6 = self.leakyrelu(down_6)
#         print(down_6.size())
        res_out = self.middle(down_6)
#         print(res_out.size())
        up_5 = self.u_deconv_5(res_out)
#         up_5 = self.u_deconv_5(down_6)
#         print(up_5.size())
#         print(skip_5.size())
        
        up_5 = torch.cat([up_5, skip_5], 1)
        up_5 = self.u_bn_5(up_5)
        up_5 = self.leakyrelu(up_5)
        
        up_4 = self.u_deconv_4(up_5)
        up_4 = torch.cat([up_4, skip_4], 1)
        up_4 = self.u_bn_4(up_4)
        up_4 = self.leakyrelu(up_4)
#         print(up_4.size())
        up_3 = self.u_deconv_3(up_4)
        up_3 = torch.cat([up_3, skip_3], 1)
        up_3 = self.u_bn_3(up_3)
        up_3 = self.leakyrelu(up_3)
#         print(up_3.size())
        up_2 = self.u_deconv_2(up_3)
        up_2 = self.u_bn_2(up_2)
        up_2 = self.leakyrelu(up_2)
#         print(up_2.size())
        up_1 = self.u_deconv_1(up_2)
        out = self.u_bn_1(up_1)
#         up_1 = self.leakyrelu(up_1)
#         print(up_1.size())
#         out = self.out_deconv(up_2)
#         out = self.out_in(out)
#         out = F.sigmoid(up_1)

        return out



class ResnetBlock(nn.Module):
    def __init__(self, dim, dilation=1, use_spectral_norm=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(dilation),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=dilation,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),

            nn.ReflectionPad2d(1),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.BatchNorm2d(dim),
        )

    def forward(self, x):
        output = x + self.conv_block(x)
#         print(output.size())
        # Remove ReLU at the end of the residual block
        # http://torch.ch/blog/2016/02/04/resnets.html

        return output



def spectral_norm(module, use_spectral_norm=True):
    if use_spectral_norm:
        return nn.utils.spectral_norm(module)

    return module
