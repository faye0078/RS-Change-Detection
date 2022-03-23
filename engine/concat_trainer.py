import paddle
import os
import time
import shutil
from paddleseg.cvlibs import Config
from paddleseg.utils import TimeAverager, calculate_eta, resume, logger
from collections import deque
from utils.yaml import _parse_from_yaml
from utils.loss import loss_computation
from configs.Config import get_trainer_config
from dataloader import make_dataloader

class Trainer(object):
    def __init__(self, args):
        self.args = args
        cfg = Config(args.cfg)
        dataset_config = _parse_from_yaml(args.cfg)

        self.train_loader, self.val_loader = make_dataloader(dataset_config, True)
        self.optimizer = cfg.optimizer
        self.losses = cfg.loss
        self.model = cfg.model

        if self.args.resume_model is not None:
            self.start_iter = resume(self.model, self.optimizer, self.args.resume_model)
        self.start_iter = 0
        train(
            cfg.model,
            train_dataset,
            val_dataset=val_dataset,
            optimizer=cfg.optimizer,
            save_dir=args.save_dir,
            iters=cfg.iters,
            batch_size=cfg.batch_size,
            resume_model=args.resume_model,
            save_interval=args.save_interval,
            log_iters=args.log_iters,
            num_workers=args.num_workers,
            use_vdl=args.use_vdl,
            losses=losses,
            keep_checkpoint_max=args.keep_checkpoint_max)
    def train(self, iters):
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
        best_mean_iou = -1.0
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
                    time.time() - batch_start, num_samples=batch_size)

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

                if (iter % self.args.save_interval == 0
                    or iter == iters) and (self.val_loader is not None):
                    num_workers = 1 if num_workers > 0 else 0
                    mean_iou, acc, class_iou, _, _ = self.val()
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
                        if mean_iou > best_mean_iou:
                            best_mean_iou = mean_iou
                            best_model_iter = iter
                            best_model_dir = os.path.join(self.args.save_dir, "best_model")
                            paddle.save(
                                self.model.state_dict(),
                                os.path.join(best_model_dir, 'model.pdparams'))
                        logger.info(
                            '[EVAL] The model with the best validation mIoU ({:.4f}) was saved at iter {}.'
                                .format(best_mean_iou, best_model_iter))

                        if self.args.use_vdl:
                            log_writer.add_scalar('Evaluate/mIoU', mean_iou, iter)
                            for i, iou in enumerate(class_iou):
                                log_writer.add_scalar('Evaluate/IoU {}'.format(i),
                                                      float(iou), iter)

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

    def val(self):# TODO: validation function
        a, b, c, d, e = 0
        return a, b, c, d, e

if __name__ == '__main__':
    args = get_trainer_config()
    test_trainer = Trainer(args)
