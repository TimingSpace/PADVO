class HardMax(nn.Module):
    def __init__(self):
        super(HardMax,self).__init__()
    def forward(self,input):
        res = input/torch.sum(input,0)
        print(input,torch.sum(input,0),res)
        return res

class VOSimple(nn.Module):

    def __init__(self, nb_ref_imgs=2):
        super(VOSimple, self).__init__()
        self.nb_ref_imgs = nb_ref_imgs

        conv_planes = [16, 32, 64, 128, 256, 256, 256]
        self.conv1 = conv(1*(1+self.nb_ref_imgs), conv_planes[0], kernel_size=7)
        self.conv2 = conv(conv_planes[0], conv_planes[1], kernel_size=5)
        self.conv3 = conv(conv_planes[1], conv_planes[2])
        self.conv4 = conv(conv_planes[2], conv_planes[3])

        self.conv5 = conv(conv_planes[3], conv_planes[4])
        self.conv6 = conv(conv_planes[4], conv_planes[5])
        self.vo_pred = nn.Conv2d(conv_planes[5], 6*self.nb_ref_imgs, kernel_size=1, padding=0)
    # weights initialization
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTransvo2d):
                nn.init.xavier_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, image_pairs):
        #assert(len(ref_imgs) == self.nb_ref_imgs)
        #input = [target_image]
        #input.extend(ref_imgs)
        #input = torch.cat(input, 1)
        input = image_pairs
        out_conv1 = self.conv1(input)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_conv4 = self.conv4(out_conv3)

        out_conv5 = self.conv5(out_conv4)
        out_conv6 = self.conv6(out_conv5)

        vo = self.vo_pred(out_conv6)
        vo = vo.mean(3).mean(2)
        # why multify 0.01
        vo = 0.01 * vo.view(vo.size(0), self.nb_ref_imgs, 6)

        return vo

class VONetDia(nn.Module):

    def __init__(self, nb_ref_imgs=2):
        super(VONetDia, self).__init__()
        self.nb_ref_imgs = nb_ref_imgs

        conv_planes = [16, 32, 64, 128, 256, 256, 256]
        self.conv1 = dia_conv(1*(1+self.nb_ref_imgs), conv_planes[0], kernel_size=7)
        self.conv2 = dia_conv(conv_planes[0], conv_planes[1], kernel_size=5)
        self.conv3 = dia_conv(conv_planes[1], conv_planes[2])
        self.conv4 = dia_conv(conv_planes[2], conv_planes[3])
        self.conv5 = dia_conv(conv_planes[3], conv_planes[4])
        self.conv6 = dia_conv(conv_planes[4], conv_planes[5])
        #self.conv7 = dia_conv(conv_planes[5], conv_planes[6])

        self.vo_pred = nn.Conv2d(conv_planes[5], 6*self.nb_ref_imgs, kernel_size=1, padding=0)
    # weights initialization
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTransvo2d):
                nn.init.xavier_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, image_pairs):
        #assert(len(ref_imgs) == self.nb_ref_imgs)
        #input = [target_image]
        #input.extend(ref_imgs)
        #input = torch.cat(input, 1)
        input = image_pairs
        out_conv1 = self.conv1(input)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_conv4 = self.conv4(out_conv3)
        out_conv5 = self.conv5(out_conv4)
        out_conv6 = self.conv6(out_conv5)
        #out_conv7 = self.conv7(out_conv6)

        vo = self.vo_pred(out_conv6)
        vo = vo.mean(3).mean(2)
        # why multify 0.01
        vo = 0.01 * vo.view(vo.size(0), self.nb_ref_imgs, 6)

        return vo
class VONetDrop2(nn.Module):

    def __init__(self, nb_ref_imgs=2):
        super(VONetDrop2, self).__init__()
        self.nb_ref_imgs = nb_ref_imgs

        conv_planes = [16, 32, 64, 128, 256, 256, 256]
        self.conv1 = conv(1*(1+self.nb_ref_imgs), conv_planes[0], kernel_size=7)
        self.drop1  = nn.Dropout2d(p=0.2)
        self.conv2 = conv(conv_planes[0], conv_planes[1], kernel_size=5)
        self.drop2  = nn.Dropout2d(p=0.2)
        self.conv3 = conv(conv_planes[1], conv_planes[2])
        self.conv4 = conv(conv_planes[2], conv_planes[3])
        self.conv5 = conv(conv_planes[3], conv_planes[4])
        self.conv6 = conv(conv_planes[4], conv_planes[5])
        self.conv7 = conv(conv_planes[5], conv_planes[6])

        self.vo_pred = nn.Conv2d(conv_planes[6], 6*self.nb_ref_imgs, kernel_size=1, padding=0)
    # weights initialization
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTransvo2d):
                nn.init.xavier_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, image_pairs):
        #assert(len(ref_imgs) == self.nb_ref_imgs)
        #input = [target_image]
        #input.extend(ref_imgs)
        #input = torch.cat(input, 1)
        input = image_pairs
        out_conv1 = self.conv1(input)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_conv4 = self.conv4(out_conv3)
        out_conv5 = self.conv5(out_conv4)
        out_drop5 = self.drop1(out_conv5)
        out_conv6 = self.conv6(out_drop5)
        out_drop6 = self.drop2(out_conv6)
        out_conv7 = self.conv7(out_drop6)
        vo = self.vo_pred(out_conv7)
        vo = vo.mean(3).mean(2)
        # why multify 0.01
        vo = 0.01 * vo.view(vo.size(0), self.nb_ref_imgs, 6)

        return vo


class VONetDrop(nn.Module):

    def __init__(self, nb_ref_imgs=2):
        super(VONetDrop, self).__init__()
        self.nb_ref_imgs = nb_ref_imgs

        conv_planes = [16, 32, 64, 128, 256, 256, 256]
        self.conv1 = conv(1*(1+self.nb_ref_imgs), conv_planes[0], kernel_size=7)
        self.drop1  = nn.Dropout2d(p=0.2)
        self.conv2 = conv(conv_planes[0], conv_planes[1], kernel_size=5)
        self.drop2  = nn.Dropout2d(p=0.2)
        self.conv3 = conv(conv_planes[1], conv_planes[2])
        self.conv4 = conv(conv_planes[2], conv_planes[3])
        self.conv5 = conv(conv_planes[3], conv_planes[4])
        self.conv6 = conv(conv_planes[4], conv_planes[5])
        self.conv7 = conv(conv_planes[5], conv_planes[6])

        self.vo_pred = nn.Conv2d(conv_planes[6], 6*self.nb_ref_imgs, kernel_size=1, padding=0)
    # weights initialization
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTransvo2d):
                nn.init.xavier_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, image_pairs):
        #assert(len(ref_imgs) == self.nb_ref_imgs)
        #input = [target_image]
        #input.extend(ref_imgs)
        #input = torch.cat(input, 1)
        input = image_pairs
        out_conv1 = self.conv1(input)
        out_conv2 = self.conv2(out_conv1)
        out_drop2 = self.drop1(out_conv2)
        out_conv3 = self.conv3(out_drop2)
        out_conv4 = self.conv4(out_conv3)
        out_conv5 = self.conv5(out_conv4)
        out_conv6 = self.conv6(out_conv5)
        out_conv7 = self.conv7(out_conv6)

        vo = self.vo_pred(out_conv7)
        vo = vo.mean(3).mean(2)
        # why multify 0.01
        vo = 0.01 * vo.view(vo.size(0), self.nb_ref_imgs, 6)

        return vo


class VONet(nn.Module):

    def __init__(self):
        super(VONet, self).__init__()

        conv_planes = [16, 32, 64, 128, 256, 256, 256]
        self.conv1 = conv(6, conv_planes[0], kernel_size=7)
        self.conv2 = conv(conv_planes[0], conv_planes[1], kernel_size=5)
        self.conv3 = conv(conv_planes[1], conv_planes[2])
        self.conv4 = conv(conv_planes[2], conv_planes[3])
        self.conv5 = conv(conv_planes[3], conv_planes[4])
        self.conv6 = conv(conv_planes[4], conv_planes[5])
        self.conv7 = conv(conv_planes[5], conv_planes[6])

        self.vo_pred = nn.Conv2d(conv_planes[6], 6, kernel_size=1, padding=0)
    # weights initialization
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTransvo2d):
                nn.init.xavier_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, image_pairs):
        input = image_pairs
        out_conv1 = self.conv1(input)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_conv4 = self.conv4(out_conv3)
        out_conv5 = self.conv5(out_conv4)
        out_conv6 = self.conv6(out_conv5)
        out_conv7 = self.conv7(out_conv6)

        vo = self.vo_pred(out_conv7)
        vo = vo.mean(3).mean(2)
        vo = 0.01 * vo.view(vo.size(0), 1, 6)

        return vo

class SSVONet(nn.Module):

    def __init__(self):
        super(SSVONet, self).__init__()

        conv_planes = [16, 32, 64, 128, 256, 256, 256]
        self.conv1 = conv(8, conv_planes[0], kernel_size=7)
        self.conv2 = conv(conv_planes[0], conv_planes[1], kernel_size=5)
        self.conv3 = conv(conv_planes[1], conv_planes[2])
        self.conv4 = conv(conv_planes[2], conv_planes[3])
        self.conv5 = conv(conv_planes[3], conv_planes[4])
        self.conv6 = conv(conv_planes[4], conv_planes[5])
        self.conv7 = conv(conv_planes[5], conv_planes[6])

        self.vo_pred = nn.Conv2d(conv_planes[6], 6, kernel_size=1, padding=0)
    # weights initialization
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTransvo2d):
                nn.init.xavier_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, image_pairs):
        input = image_pairs
        out_conv1 = self.conv1(input)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_conv4 = self.conv4(out_conv3)
        out_conv5 = self.conv5(out_conv4)
        out_conv6 = self.conv6(out_conv5)
        out_conv7 = self.conv7(out_conv6)

        vo = self.vo_pred(out_conv7)
        #vo = vo.mean(3).mean(2)
        #vo = 0.01 * vo.view(vo.size(0), 1, 6)

        return vo
class AttentionNet(nn.Module):

    def __init__(self):
        super(AttentionNet, self).__init__()

        conv_planes = [16, 32, 64, 128, 256, 128, 64]
        self.conv1 = conv(8, conv_planes[0], kernel_size=7)
        self.conv2 = conv(conv_planes[0], conv_planes[1], kernel_size=5)
        self.conv3 = conv(conv_planes[1], conv_planes[2])
        self.conv4 = conv(conv_planes[2], conv_planes[3])
        self.conv5 = conv(conv_planes[3], conv_planes[4])
        self.conv6 = conv(conv_planes[4], conv_planes[5])
        self.conv7 = conv(conv_planes[5], conv_planes[6])

        self.att_pred = nn.Conv2d(conv_planes[6], 1, kernel_size=1, padding=0)
        self.soft_max = nn.Softmax(2)
    # weights initialization
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTransvo2d):
                nn.init.xavier_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, image_pairs):
        input = image_pairs
        out_conv1 = self.conv1(input)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_conv4 = self.conv4(out_conv3)
        out_conv5 = self.conv5(out_conv4)
        out_conv6 = self.conv6(out_conv5)
        out_conv7 = self.conv7(out_conv6)

        att = self.att_pred(out_conv7)
        att_solf = self.soft_max(att.view(att.size(0),att.size(1),att.size(2)*att.size(3))).view(att.size(0),att.size(1),att.size(2),att.size(3))
        #vo = vo.mean(3).mean(2)
        #vo = 0.01 * vo.view(vo.size(0), 1, 6)

        return att_solf

