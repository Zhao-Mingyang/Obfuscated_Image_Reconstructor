class PyNET(nn.Module):

    def __init__(self, instance_norm=True, instance_norm_level_1=False):
        super(PyNET, self).__init__()


        self.conv_l1_d1 = ConvLayer(3, 32, 3,1, instance_norm=False)

        self.conv_l2_d1 = ConvLayer(32, 64, 3, 1,instance_norm=instance_norm)


        self.conv_l3_d1 = ConvLayer(64, 128, 3,1, instance_norm=instance_norm)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv_l4_d1 = ConvLayer(128, 256, 3, 1,instance_norm=instance_norm)

        # -------------------------------------

        self.conv_l5_d1 = ConvLayer(256, 512, 3, 1,instance_norm=instance_norm)
        self.conv_l5_d2 = ConvLayer(512, 512, 3,1, instance_norm=instance_norm)
#         self.conv_l5_d3 = ConvMultiBlock(512, 512, 3, instance_norm=instance_norm)
#         self.conv_l5_d4 = ConvMultiBlock(512, 512, 3, instance_norm=instance_norm)
        self.conv_l5_d5 = ConvLayer(512, 256, 3, 1,instance_norm=instance_norm)
        

        self.conv_t4a = UpsampleConvLayer(256, 128, 3)
        self.conv_t4b = UpsampleConvLayer(256, 128, 3)

#         self.conv_t4a = UpsampleConvLayer(512, 128, 3)
#         self.conv_t4b = UpsampleConvLayer(512, 128, 3)

        self.conv_l5_out = ConvLayer(256, 3, kernel_size=3, stride=1, relu=False)
        self.output_l5 = nn.Sigmoid()


        # -------------------------------------

        self.conv_l3_d3a = ConvLayer(256, 128, 5,1, instance_norm=instance_norm)
        self.conv_l3_d3b = ConvLayer(256, 128, 3, 1,instance_norm=instance_norm)
#         self.conv_l3_d4 = ConvMultiBlock(256, 128, 5, instance_norm=instance_norm)
#         self.conv_l3_d5 = ConvMultiBlock(256, 128, 5, instance_norm=instance_norm)
        self.conv_l3_d6a = ConvLayer(256, 128, 5, 1,instance_norm=instance_norm)
        self.conv_l3_d6b = ConvLayer(256, 128, 3,1, instance_norm=instance_norm)
        self.conv_l3_d8 = ConvLayer(512, 128, 3,1, instance_norm=instance_norm)
        
        self.conv_l3_d9 = ConvLayer(128, 64, 5,1)
        self.conv_l3_d10 = ConvLayer(64, 32, 5,1)
        self.conv_l3_d11 = ConvLayer(32, 16, 5,1)


        # -------------------------------------

        self.conv_l0_d1 = ConvLayer(16, 3, kernel_size=3, stride=1, relu=False)
        self.output_l0 = nn.Sigmoid()

    def level_5(self, pool4):

        conv_l5_d1 = self.conv_l5_d1(pool4)
        conv_l5_d2 = self.conv_l5_d2(conv_l5_d1)
#         conv_l5_d3 = self.conv_l5_d3(conv_l5_d2)
#         conv_l5_d4 = self.conv_l5_d4(conv_l5_d3)
        conv_l5_d5 = self.conv_l5_d5(conv_l5_d2)

        conv_t4a = self.conv_t4a(conv_l5_d5)
        conv_t4b = self.conv_t4b(conv_l5_d5)

#         conv_t4a = self.conv_t4a(conv_l5_d4)
#         conv_t4b = self.conv_t4b(conv_l5_d4)

        conv_l5_out = self.conv_l5_out(conv_l5_d5)
#         conv_l5_out = self.conv_l5_out(conv_l5_d4)
        output_l5 = self.output_l5(conv_l5_out)

        return output_l5, conv_t4a, conv_t4b

    def level_3(self, conv_l3_d1, conv_t4a, conv_t4b):
#         print(conv_l3_d1.size())
#         print(conv_t4a.size())
#         print(conv_t4b.size())
        conv_l3_d2 = torch.cat([conv_l3_d1, conv_t4a], 1)

        conv_l3_d3a = self.conv_l3_d3a(conv_l3_d2)
        conv_l3_d3b = self.conv_l3_d3b(conv_l3_d2)
        conv_l3_d3 = torch.cat([conv_l3_d3b,conv_l3_d3a], 1)
        conv_l3_d3 = conv_l3_d3 + conv_l3_d2
#         conv_l3_d4 = self.conv_l3_d4(conv_l3_d3) + conv_l3_d3
#         conv_l3_d5 = self.conv_l3_d5(conv_l3_d4) + conv_l3_d4
        conv_l3_d6a = self.conv_l3_d6a(conv_l3_d3)
        conv_l3_d6b = self.conv_l3_d6b(conv_l3_d3)
        conv_l3_d6 = torch.cat([conv_l3_d6b,conv_l3_d6a], 1)

        conv_l3_d7 = torch.cat([conv_l3_d6, conv_l3_d1, conv_t4b], 1)
        conv_l3_d8 = self.conv_l3_d8(conv_l3_d7)
        conv_l3_d9 = self.conv_l3_d9(conv_l3_d8)
        conv_l3_d10 = self.conv_l3_d10(conv_l3_d9)
        output_l3= self.conv_l3_d11(conv_l3_d10)

        return output_l3

    def level_0(self, conv_t0):

        conv_l0_d1 = self.conv_l0_d1(conv_t0)
        output_l0 = self.output_l0(conv_l0_d1)
        

        return output_l0

    def forward(self, x):

#         conv_l1_d1 = self.conv_l1_d1(x)
#         pool1 = self.pool1(conv_l1_d1)

#         conv_l2_d1 = self.conv_l2_d1(pool1)
#         pool2 = self.pool2(conv_l2_d1)

        conv_l1_d1 = self.conv_l1_d1(x)

        conv_l2_d1 = self.conv_l2_d1(conv_l1_d1)

        conv_l3_d1 = self.conv_l3_d1(conv_l2_d1)
        pool3 = self.pool3(conv_l3_d1)

        conv_l4_d1 = self.conv_l4_d1(pool3)
        output_l5, conv_t4a, conv_t4b = self.level_5(conv_l4_d1)       

        output_l3= self.level_3(conv_l3_d1, conv_t4a, conv_t4b)
        
        enhanced = self.level_0(output_l3)

        return enhanced



class ConvLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, relu=True, instance_norm=False):

        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2

        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

        self.instance_norm = instance_norm
        self.instance = None
        self.relu = None
        

        if instance_norm:
            self.instance = nn.InstanceNorm2d(out_channels, affine=True)

        if relu:
            self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):

        out = self.reflection_pad(x)
        out = self.conv2d(out)

        if self.instance_norm:
            out = self.instance(out)

        if self.relu:
            out = self.relu(out)

        return out


class UpsampleConvLayer(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, upsample=2, stride=1, relu=True):

        super(UpsampleConvLayer, self).__init__()
        self.upsample = nn.Upsample(scale_factor=upsample, mode='bilinear', align_corners=True)

        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)

        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

        if relu:
            self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):

        out = self.upsample(x)
        out = self.reflection_pad(out)
        out = self.conv2d(out)

        if self.relu:
            out = self.relu(out)

        return out
