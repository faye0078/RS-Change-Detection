import paddle
import paddle.nn as nn
from model.backbone import hrnet, vit, swt, UNetv1, UNetv2
from model.header import fcn

class SplitNet(nn.Layer):

    def __init__(self, encoder1, encoder2, decoder,
                 in_channels=3,
                 num_classes=2,
                 is_batchnorm=True):
        super(SplitNet, self).__init__()
        if encoder1 == 'hrnet':
            encoder1_model = hrnet.HRNet_W48()
        if encoder2 == 'hrnet':
            encoder2_model = hrnet.HRNet_W48()
        if decoder == 'fcn':
            decoder_model = fcn.FCNHead(num_classes, 200, 200)

        # internal definition
        self.filters = [64, 128, 256, 512, 1024]
        self.cat_channels = self.filters[0]
        self.cat_blocks = 5
        self.up_channels = self.cat_channels * self.cat_blocks
        # layers
        if encoder1 == encoder2:
            self.encoder = encoder1_model
        else:
            self.encoder1 = encoder1_model
            self.encoder2 = encoder2_model
        self.decoder = decoder_model

    def forward(self, x1, x2):
        if self.encoder:
            features1 = self.encoder(x1)
            features2 = self.encoder(x2)
        else:
            features1 = self.encoder1(x1)
            features2 = self.encoder2(x2)
        features = features1 - features2
        output = self.decoder(features)
        return output

def get_split_model(nclass, backbone1, backbone2, header):
    return SplitNet(backbone1, backbone2, header, num_classes=nclass)
