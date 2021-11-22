import torch
import torch.nn as nn

class Encoder_Decoder_Residual(nn.Module):
    def __init__(self,Discriminator_compare, rs_blocks=8):
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

        blocks = []
        for _ in range(8):
            block = ResnetBlock(256, 2)
            blocks.append(block)
          

        self.middle = nn.Sequential(*blocks)

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
        x = self.middle(x)
        x = self.decoder(x)
        out = (torch.tanh(x) + 1) / 2
        outclass = self.discriminator(out)

        return out, outclass





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



# class Encoder_Decoder_Residual(nn.Module):
#     def __init__(self, rs_blocks=8):
#         super(Encoder_Decoder_Residual, self).__init__()

#         self.encoder = nn.Sequential(
#             nn.ReflectionPad2d(3),
#             nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, padding=0),
#             nn.InstanceNorm2d(64, track_running_stats=False),
#             nn.LeakyReLU(0.1),
# #             nn.ReLU(True),

#             nn.Conv2d(in_channels=64, out_channels=128, kernel_size=6, stride=2, padding=0),
#             nn.InstanceNorm2d(128, track_running_stats=False),
#             nn.LeakyReLU(0.1),
# #             nn.ReLU(True),

#             nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
#             nn.InstanceNorm2d(256, track_running_stats=False),
#             nn.LeakyReLU(0.1),
# #             nn.ReLU(True)
#         )

#         blocks = []
#         for _ in range(8):
#             block = ResnetBlock(256, 2)
#             blocks.append(block)
          

#         self.middle = nn.Sequential(*blocks)

#         self.decoder = nn.Sequential(

#             # replace ConvTranspose2d with upsample + conv as in paper
#             nn.Upsample(scale_factor=2, mode='bilinear'),
#             nn.ReflectionPad2d(1),
#             nn.ConvTranspose2d(256, 128, kernel_size=4, stride=1, padding=2),

#             nn.InstanceNorm2d(128, track_running_stats=False),
#             nn.LeakyReLU(0.1),

#             nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=6, stride=2, padding=3),
#             nn.InstanceNorm2d(64, track_running_stats=False),
#             nn.LeakyReLU(0.1),

#             nn.ReflectionPad2d(3),
#             nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=9, padding=3),
#         )

#     def forward(self, x):
#         x = self.encoder(x)
#         x = self.middle(x)
#         x = self.decoder(x)
#         x = (torch.tanh(x) + 1) / 2

#         return x





# class ResnetBlock(nn.Module):
#     def __init__(self, dim, dilation=1, use_spectral_norm=False):
#         super(ResnetBlock, self).__init__()
#         self.conv_block = nn.Sequential(
#             nn.ReflectionPad2d(dilation),
#             nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=dilation),
#             nn.InstanceNorm2d(dim, track_running_stats=False),
#             nn.LeakyReLU(0.2),

#             nn.ReflectionPad2d(1),
#             nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=1),
#             nn.InstanceNorm2d(dim, track_running_stats=False),
# #             nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, padding=0),
# #             nn.Sigmoid()
#         )


#     def forward(self, x):
#         output = self.conv_block(x)
# #         output = output*x+x
#         output = output+x

#         # Remove ReLU at the end of the residual block
#         # http://torch.ch/blog/2016/02/04/resnets.html

#         return output


# # def spectral_norm(module, use_spectral_norm=True):
# #     if use_spectral_norm:
# #         return nn.utils.spectral_norm(module)

# #     return module