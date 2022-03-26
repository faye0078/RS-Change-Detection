from model.backbone import hrnet, vit, swt, UNetv1, UNetv2
from paddleseg.models import ocrnet, deeplab, fcn


def get_concat_model(nclass, backbone, header):
    if backbone == 'hrnet':
        backbone_model = hrnet.HRNet_W48()

    if header == 'fcn':
        header_model = fcn.FCN(nclass, backbone_model)

    return header_model
