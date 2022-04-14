import sys
sys.path.append('./PaddleRS')
import argparse
import paddle
from utils.yaml import _parse_from_yaml
from dataloader.dataset_maker import make_train_dataset
from model.model_maker import make_model
class Trainer(object):
    def __init__(self, args):
        self.args = args


        self.train_dataset, self.eval_dataset = make_train_dataset(args)

        self.model = make_model(self.args["MODEL_NAME"])

        # 制定定步长学习率衰减策略
        self.lr_scheduler = paddle.optimizer.lr.StepDecay(
            args["LR"],
            step_size=args["DECAY_STEP"],
            # 学习率衰减系数，这里指定每次减半
            gamma=0.5
        )
        # 构造Adam优化器
        self.optimizer = paddle.optimizer.Adam(
            learning_rate=self.lr_scheduler,
            # 在PaddleRS中，可通过ChangeDetector对象的net属性获取paddle.nn.Layer类型组网
            parameters=self.model.net.parameters()
        )


    def train(self):
        # 调用PaddleRS API实现一键训练
        self.model.train(
            num_epochs=self.args["NUM_EPOCHS"],
            train_dataset=self.train_dataset,
            train_batch_size=self.args["BATCH_SIZE"],
            eval_dataset=self.eval_dataset,
            optimizer=self.optimizer,
            save_interval_epochs=self.args["SAVE_INTERVAL_EPOCHS"],
            # 每多少次迭代记录一次日志
            log_interval_steps=20,
            save_dir=self.args["EXP_DIR"],
            # 是否使用early stopping策略，当精度不再改善时提前终止训练
            early_stop=False,
            # 是否启用VisualDL日志功能
            use_vdl=True,
            # 指定从某个检查点继续训练
            resume_checkpoint=self.args["RESUME"]
        )


if __name__ == '__main__':
    # 命令行读取yaml文件名
    parser = argparse.ArgumentParser(description='Model training')
    parser.add_argument("--config", dest="cfg", help="The config file.", default='./experiment/BIT.yml', type=str)
    # 读取yaml参数
    args = parser.parse_args()
    args = _parse_from_yaml(args.cfg)
    test_trainer = Trainer(args)