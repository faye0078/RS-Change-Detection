import paddlers as pdrs
import paddlers.utils.logging as logging
def make_model(model_name):
    if model_name == 'BIT':
        # 调用PaddleRS API一键构建模型
        model = pdrs.tasks.BIT(
            # 模型输出类别数
            num_classes=2,
            # 是否使用混合损失函数，默认使用交叉熵损失函数训练
            use_mixed_loss=False,
            # 模型输入通道数
            in_channels=3,
            # 模型使用的骨干网络，支持'resnet18'或'resnet34'
            backbone='resnet34',
            # 骨干网络中的resnet stage数量
            n_stages=4,
            # 是否使用tokenizer获取语义token
            use_tokenizer=True,
            # token的长度
            token_len=4,
            # 若不使用tokenizer，则使用池化方式获取token。此参数设置池化模式，有'max'和'avg'两种选项，分别对应最大池化与平均池化
            pool_mode='max',
            # 池化操作输出特征图的宽和高（池化方式得到的token的长度为pool_size的平方）
            pool_size=2,
            # 是否在Transformer编码器中加入位置编码（positional embedding）
            enc_with_pos=True,
            # Transformer编码器使用的注意力模块（attention block）个数
            enc_depth=1,
            # Transformer编码器中每个注意力头的嵌入维度（embedding dimension）
            enc_head_dim=64,
            # Transformer解码器使用的注意力模块个数
            dec_depth=8,
            # Transformer解码器中每个注意力头的嵌入维度
            dec_head_dim=8
        )
    elif model_name == 'DSIFN':
        model = pdrs.tasks.DSIFN(
            num_classes=2,
            use_mixed_loss=False,
            use_dropout=True
        )
    elif model_name == 'STANet':
        model = pdrs.tasks.STANet(
            num_classes=2,
            use_mixed_loss=False,
            in_channels=3,
            att_type='BAM',
            ds_factor=1,
        )
    elif model_name == 'SNUNet':
        model = pdrs.tasks.SNUNet(
            in_channels=3,
            num_classes=2,
            width=32
        )
    elif model_name == 'DSAMNet':
        model = pdrs.tasks.DSAMNet(
            in_channels=3,
            num_classes=2,
            ca_ratio=8,
            sa_kernel=7
        )
    elif model_name == 'ChangeStar':
        model = pdrs.tasks.ChangeStar(
            num_classes=2,
            mid_channels=256,
            inner_channels=16,
            num_convs=4,
            scale_factor=4.0,
        )
    elif model_name == 'FCSiamConc':
        model = pdrs.tasks.FCSiamConc(
            in_channels=3,
            num_classes=2,
            use_dropout=False
        )
    elif model_name == 'FCSiamDiff':
        model = pdrs.tasks.FCSiamDiff(
            in_channels=3,
            num_classes=2,
            use_dropout=False
        )
    elif model_name == 'FCEarlyFusion':
        model = pdrs.tasks.FCEarlyFusion(
            in_channels=3,
            num_classes=2,
            use_dropout=False
        )
    logging.info('[TRAIN] model name is: {} .'
                 .format(model_name))

    return model