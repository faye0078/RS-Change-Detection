import paddle
import os
import time
import shutil
import numpy as np
from utils.f1 import f1
from paddleseg.core import infer
from configs.config import Config
from paddleseg.utils import TimeAverager, calculate_eta, resume, logger, progbar, metrics
from collections import deque
from utils.yaml import _parse_from_yaml
from utils.loss import loss_computation
from utils.preprocess import make_transform
from configs.MyConfig import get_trainer_config
from dataloader import make_dataloader

class Trainer(object):
    def __init__(self, args):
        self.args = args
        cfg = Config(args.cfg)
        self.origin_config = _parse_from_yaml(args.cfg)

        self.train_set, self.val_set, self.train_loader, self.val_loader = make_dataloader(self.origin_config['dataset'], True)

        self.val_transforms = make_transform(self.origin_config['dataset']['val_dataset']['transforms']) #
        self.nclasses = self.origin_config['model']['num_classes']

        self.optimizer = cfg.optimizer
        self.losses = cfg.loss
        self.model = cfg.model

        if self.args.resume_model is not None:
            self.start_iter = resume(self.model, self.optimizer, self.args.resume_model)
        self.start_iter = 0


    def train(self, iters=None):
        if iters == None:
            iters = self.origin_config['iters']
        self.model.train()
        nranks = paddle.distributed.ParallelEnv().nranks
        local_rank = paddle.distributed.ParallelEnv().local_rank

        if not os.path.isdir(self.args.save_dir):
            if os.path.exists(self.args.save_dir):
                os.remove(self.args.save_dir)
            os.makedirs(self.args.save_dir)

        if nranks > 1:
            # Initialize parallel environment if not done.
            if not paddle.distributed.parallel.parallel_helper._is_parallel_ctx_initialized(
            ):
                paddle.distributed.init_parallel_env()
                ddp_model = paddle.DataParallel(self.model)
            else:
                ddp_model = paddle.DataParallel(self.model)

        if self.args.use_vdl:
            from visualdl import LogWriter
            log_writer = LogWriter(self.args.save_dir)

        avg_loss = 0.0
        avg_loss_list = []
        iters_per_epoch = len(self.train_loader)
        best_f1 = -1.0
        best_model_iter = -1
        reader_cost_averager = TimeAverager()
        batch_cost_averager = TimeAverager()
        save_models = deque()
        batch_start = time.time()

        iter = self.start_iter
        while iter < iters:
            for data in self.train_loader:
                iter += 1
                if iter > iters:
                    break
                reader_cost_averager.record(time.time() - batch_start)
                images = data[0]
                labels = data[1].astype('int64')

                if nranks > 1:
                    logits_list = ddp_model(images)
                else:
                    logits_list = self.model(images)
                loss_list = loss_computation(
                    logits_list=logits_list,
                    labels=labels,
                    losses=self.losses)
                loss = sum(loss_list)
                loss.backward()

                self.optimizer.step()
                lr = self.optimizer.get_lr()
                if isinstance(self.optimizer._learning_rate,
                              paddle.optimizer.lr.LRScheduler):
                    self.optimizer._learning_rate.step()
                self.model.clear_gradients()
                avg_loss += loss.numpy()[0]
                if not avg_loss_list:
                    avg_loss_list = [l.numpy() for l in loss_list]
                else:
                    for i in range(len(loss_list)):
                        avg_loss_list[i] += loss_list[i].numpy()
                batch_cost_averager.record(
                    time.time() - batch_start, num_samples=self.origin_config['batch_size'])

                if (iter) % self.args.log_iters == 0 and local_rank == 0:
                    avg_loss /= self.args.log_iters
                    avg_loss_list = [l[0] / self.args.log_iters for l in avg_loss_list]
                    remain_iters = iters - iter
                    avg_train_batch_cost = batch_cost_averager.get_average()
                    avg_train_reader_cost = reader_cost_averager.get_average()
                    eta = calculate_eta(remain_iters, avg_train_batch_cost)
                    logger.info(
                        "[TRAIN] epoch: {}, iter: {}/{}, loss: {:.4f}, lr: {:.6f}, batch_cost: {:.4f}, reader_cost: {:.5f}, ips: {:.4f} samples/sec | ETA {}"
                            .format((iter - 1) // iters_per_epoch + 1, iter, iters,
                                    avg_loss, lr, avg_train_batch_cost,
                                    avg_train_reader_cost,
                                    batch_cost_averager.get_ips_average(), eta))
                    if self.args.use_vdl:
                        log_writer.add_scalar('Train/loss', avg_loss, iter)
                        # Record all losses if there are more than 2 losses.
                        if len(avg_loss_list) > 1:
                            avg_loss_dict = {}
                            for i, value in enumerate(avg_loss_list):
                                avg_loss_dict['loss_' + str(i)] = value
                            for key, value in avg_loss_dict.items():
                                log_tag = 'Train/' + key
                                log_writer.add_scalar(log_tag, value, iter)

                        log_writer.add_scalar('Train/lr', lr, iter)
                        log_writer.add_scalar('Train/batch_cost',
                                              avg_train_batch_cost, iter)
                        log_writer.add_scalar('Train/reader_cost',
                                              avg_train_reader_cost, iter)
                    avg_loss = 0.0
                    avg_loss_list = []
                    reader_cost_averager.reset()
                    batch_cost_averager.reset()

                if (iter % self.args.save_interval == 0 or iter == iters) and (self.val_loader is not None):
                    f1, class_f1, mean_iou, acc, class_iou, _, _ = self.val()
                    self.model.train()

                if (iter % self.args.save_interval == 0 or iter == iters) and local_rank == 0:
                    current_save_dir = os.path.join(self.args.save_dir,
                                                    "iter_{}".format(iter))
                    if not os.path.isdir(current_save_dir):
                        os.makedirs(current_save_dir)
                    paddle.save(self.model.state_dict(),
                                os.path.join(current_save_dir, 'model.pdparams'))
                    paddle.save(self.optimizer.state_dict(),
                                os.path.join(current_save_dir, 'model.pdopt'))
                    save_models.append(current_save_dir)
                    if len(save_models) > self.args.keep_checkpoint_max > 0:
                        model_to_remove = save_models.popleft()
                        shutil.rmtree(model_to_remove)

                    if self.val_loader is not None:
                        if f1 > best_f1:
                            best_f1 = f1
                            best_model_iter = iter
                            best_model_dir = os.path.join(self.args.save_dir, "best_model")
                            paddle.save(
                                self.model.state_dict(),
                                os.path.join(best_model_dir, 'model.pdparams'))
                        logger.info(
                            '[EVAL] The model with the best validation mIoU ({:.4f}) was saved at iter {}.'
                                .format(best_f1, best_model_iter))

                        if self.args.use_vdl:
                            log_writer.add_scalar('Evaluate/F1', f1, iter)
                            for i, f1 in enumerate(class_f1):
                                log_writer.add_scalar('Evaluate/IoU {}'.format(i),
                                                      float(f1), iter)

                            log_writer.add_scalar('Evaluate/Acc', acc, iter)
                batch_start = time.time()

        # Calculate flops.
        if local_rank == 0:
            def count_syncbn(m, x, y):
                x = x[0]
                nelements = x.numel()
                m.total_ops += int(2 * nelements)

            _, c, h, w = images.shape
            self.flops = paddle.flops(
                self.model, [1, c, h, w],
                custom_ops={paddle.nn.SyncBatchNorm: count_syncbn})

        # Sleep for half a second to let dataloader release resources.
        time.sleep(0.5)
        if self.use_vdl:
            log_writer.close()

    def val(self):
        self.model.eval()
        nranks = paddle.distributed.ParallelEnv().nranks
        local_rank = paddle.distributed.ParallelEnv().local_rank
        if nranks > 1:
            # Initialize parallel environment if not done.
            if not paddle.distributed.parallel.parallel_helper._is_parallel_ctx_initialized(
            ):
                paddle.distributed.init_parallel_env()

        total_iters = len(self.val_loader)
        intersect_area_all = 0
        pred_area_all = 0
        label_area_all = 0
        logits_all = None
        label_all = None

        logger.info(
            "Start evaluating (total_iters: {})...".format(total_iters))
        progbar_val = progbar.Progbar(
            target=total_iters, verbose=1 if nranks < 2 else 2)
        reader_cost_averager = TimeAverager()
        batch_cost_averager = TimeAverager()
        batch_start = time.time()
        with paddle.no_grad():
            for iter, (im, label) in enumerate(self.val_loader):
                reader_cost_averager.record(time.time() - batch_start)
                label = label.astype('int64')

                ori_shape = label.shape[-2:]
                pred, logits = infer.inference(
                    self.model,
                    im,
                    ori_shape=ori_shape,
                    transforms=self.val_transforms)

                intersect_area, pred_area, label_area = metrics.calculate_area(
                    pred,
                    label,
                    self.nclasses)

                # Gather from all ranks
                if nranks > 1:
                    intersect_area_list = []
                    pred_area_list = []
                    label_area_list = []
                    paddle.distributed.all_gather(intersect_area_list,
                                                  intersect_area)
                    paddle.distributed.all_gather(pred_area_list, pred_area)
                    paddle.distributed.all_gather(label_area_list, label_area)

                    # Some image has been evaluated and should be eliminated in last iter
                    if (iter + 1) * nranks > len(self.val_set):
                        valid = len(self.val_set) - iter * nranks
                        intersect_area_list = intersect_area_list[:valid]
                        pred_area_list = pred_area_list[:valid]
                        label_area_list = label_area_list[:valid]

                    for i in range(len(intersect_area_list)):
                        intersect_area_all = intersect_area_all + intersect_area_list[
                            i]
                        pred_area_all = pred_area_all + pred_area_list[i]
                        label_area_all = label_area_all + label_area_list[i]
                else:
                    intersect_area_all = intersect_area_all + intersect_area
                    pred_area_all = pred_area_all + pred_area
                    label_area_all = label_area_all + label_area

                batch_cost_averager.record(
                    time.time() - batch_start, num_samples=len(label))
                batch_cost = batch_cost_averager.get_average()
                reader_cost = reader_cost_averager.get_average()

                if local_rank == 0:
                    progbar_val.update(iter + 1, [('batch_cost', batch_cost),
                                                  ('reader cost', reader_cost)])
                reader_cost_averager.reset()
                batch_cost_averager.reset()
                batch_start = time.time()

        class_iou, miou = metrics.mean_iou(intersect_area_all, pred_area_all, label_area_all)
        class_acc, acc = metrics.accuracy(intersect_area_all, pred_area_all)
        kappa = metrics.kappa(intersect_area_all, pred_area_all, label_area_all)
        class_dice, mdice = metrics.dice(intersect_area_all, pred_area_all, label_area_all)
        class_f1, all_f1 = f1(intersect_area_all, pred_area_all, label_area_all)

        infor = "[EVAL] #Images: {} F1: {:.4f} class_f1: {} mIoU: {:.4f} Acc: {:.4f} Kappa: {:.4f} Dice: {:.4f}".format(
            len(self.val_set), all_f1,class_f1, miou, acc, kappa, mdice)
        infor = infor
        logger.info(infor)
        logger.info("[EVAL] Class IoU: \n" + str(np.round(class_iou, 4)))
        logger.info("[EVAL] Class Acc: \n" + str(np.round(class_acc, 4)))
        return all_f1, class_f1, miou, acc, class_iou, class_acc, kappa

if __name__ == '__main__':
    args = get_trainer_config()
    test_trainer = Trainer(args)
