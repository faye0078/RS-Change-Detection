from model.backbone import hrnet, UNetv1, UNetv2
from paddleseg.models import ocrnet, deeplab, fcn


def get_concat_model(nclass, backbone, header):
    if backbone == 'hrnet':
        backbone_model = hrnet.HRNet_W48()

    if backbone == 'unetv1':
        return UNetv1.UNet3Plus(in_channels=6, num_classes=2)

    if header == 'fcn':
        header_model = fcn.FCN(nclass, backbone_model)

    return header_model
