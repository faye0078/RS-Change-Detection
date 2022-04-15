from model.model_maker import make_model
from utils.dataTrans import quantize, info
from dataloader.test_dataset import recons_prob_map
import sys
sys.path.append('../PaddleRS')
import argparse
from utils.yaml import _parse_from_yaml
import os
import os.path as osp
import paddle
from dataloader.dataset_maker import make_test_dataset
from skimage.io import imread, imsave
from tqdm import tqdm

class Predictor(object):
    def __init__(self, args):
        self.args = args
        self.args["BEST_CKP_PATH"] = os.path.join(self.args["EXP_DIR"], 'best_model', 'model.pdparams')

        self.test_dataset, self.test_dataloader = make_test_dataset(args)

        self.model = make_model(self.args["MODEL_NAME"])
        # 为模型加载历史最佳权重
        state_dict = paddle.load(args["BEST_CKP_PATH"])
        # 同样通过net属性访问组网对象
        self.model.net.set_state_dict(state_dict)


    def predict(self, out_dir):
        # 推理过程主循环

        self.model.net.eval()
        len_test = len(self.test_dataset.names)
        with paddle.no_grad():
            for name, (t1, t2) in tqdm(zip(self.test_dataset.names, self.test_dataloader), total=len_test):
                shape = paddle.shape(t1)
                pred = paddle.zeros(shape=(shape[0], 2, *shape[2:]))
                for i in range(0, shape[0], self.args["INFER_BATCH_SIZE"]):
                    pred[i:i+self.args["INFER_BATCH_SIZE"]] = self.model.net(t1[i:i+self.args["INFER_BATCH_SIZE"]], t2[i:i+self.args["INFER_BATCH_SIZE"]])[0]
                # 取softmax结果的第1（从0开始计数）个通道的输出作为变化概率
                prob = paddle.nn.functional.softmax(pred, axis=1)[:, 1]
                # 由patch重建完整概率图
                prob = recons_prob_map(prob.numpy(), self.args["ORIGINAL_SIZE"], self.args["CROP_SIZE"], self.args["STRIDE"])
                # 默认将阈值设置为0.5，即，将变化概率大于0.5的像素点分为变化类
                out = quantize(prob > self.args["THRESHOLD"])

                imsave(osp.join(out_dir, name), out, check_contrast=False)

        info("模型推理完成")



if __name__ == '__main__':
    # 命令行读取yaml文件名
    parser = argparse.ArgumentParser(description='Model training')
    parser.add_argument("--config", dest="cfg", help="The config file.", default='./experiment/BIT.yml', type=str)
    # 读取yaml参数
    args = parser.parse_args()
    args = _parse_from_yaml(args.cfg)

    test_trainer = Predictor(args)