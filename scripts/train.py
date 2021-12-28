import gc
import os
import sys
import time
import shutil
import datetime
import argparse

import numpy as np
from cv2 import cv2 as cv2

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn
from torchvision import transforms
from light.utils.distributed import *
from light.utils.logger import setup_logger
from light.utils.lr_scheduler import WarmupPolyLR
from light.utils.metric import SegmentationMetric
from light.data import get_segmentation_dataset
from light.model import get_segmentation_model
from light.nn import MixSoftmaxCrossEntropyLoss, MixSoftmaxCrossEntropyOHEMLoss
from light.utils.my_utils import *


def parse_args():
    parser = argparse.ArgumentParser(description='Light Model for Segmentation')
    # model and dataset
    parser.add_argument('--model', type=str, default='mobilenetv3_small',
                        help='model name (default: mobilenet)')
    parser.add_argument('--dataset', type=str, default='elements',
                        help='dataset name (choices: citys, elements)')
    parser.add_argument('--base-size', type=int, default=700,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=700,
                        help='crop image size')
    parser.add_argument('--workers', '-j', type=int, default=2,          # ------------------WORKERS------------
                        metavar='N', help='dataloader threads')
    # training hyper params
    parser.add_argument('--ohem', action='store_true', default=False,
                        help='OHEM Loss for cityscapes dataset')
    parser.add_argument('--aux', action='store_true', default=False,
                        help='Auxiliary loss')
    parser.add_argument('--aux-weight', type=float, default=0.4,
                        help='auxiliary loss weight')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',          # ---------BATCH------------
                        help='input batch size for training (default: 4)')
    parser.add_argument('--start-epoch', type=int, default=71,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--epochs', type=int, default=999, metavar='N',
                        help='number of epochs to train (default: 240)')
    parser.add_argument('--lr', type=float, default=0.00005, metavar='LR',
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=1e-4, metavar='M',
                        help='w-decay (default: 5e-4)')
    parser.add_argument('--warmup-iters', type=int, default=0,
                        help='warmup iters')
    parser.add_argument('--warmup-factor', type=float, default=1.0 / 3,
                        help='lr = warmup_factor * lr')
    parser.add_argument('--warmup-method', type=str, default='linear',
                        help='method of warmup')
    parser.add_argument('--max_nonimprovement', type=int, default=2,
                        help='Maximum number of validations without improvement. Exceeding it stops the training.')
    # cuda setting
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--local_rank', type=int, default=0)
    # checkpoint and log
    parser.add_argument('--resume', type=str, default='C:/Users/Admin/PycharmProjects/lightweight/runs/saved_state/best/best_model.pth',
                        help='put the path to resuming file if needed')
    parser.add_argument('--save-dir', default='c:/users/admin/pycharmprojects/lightweight/runs/saved_state/best',
                        help='Directory for saving checkpoint models')
    parser.add_argument('--save-epoch', type=int, default=10,
                        help='save model every checkpoint-epoch')
    parser.add_argument('--log-dir', default='../runs/logs/',
                        help='Directory for saving checkpoint models')
    parser.add_argument('--log-iter', type=int, default=1,
                        help='print log every log-iter')
    # evaluation only
    parser.add_argument('--skip-val', action='store_true', default=False,
                        help='skip validation during training')
    parser.add_argument('--val-epoch', type=int, default=5,
                        help='run validation every val-epoch')
    args = parser.parse_args()

    # default settings for epochs, batch_size and lr
    # args.lr = args.lr / 4 * args.batch_size  # INFO: orig

    return args


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        self.no_improvement = 0  # number of validations without improvement

        # image transform
        input_transform = transforms.Compose([
            transforms.ToTensor()#,
            #transforms.Normalize([.485, .456, .406], [.229, .224, .225]),  # INFO: orig
        ])
        # dataset and dataloader
        data_kwargs = {'transform': input_transform, 'base_size': args.base_size, 'crop_size': args.crop_size}
        trainset = get_segmentation_dataset(args.dataset, split='train', mode='train', **data_kwargs)
        args.iters_per_epoch = len(trainset) // (args.num_gpus * args.batch_size)
        args.max_iters = args.epochs * args.iters_per_epoch

        train_sampler = make_data_sampler(trainset, shuffle=True, distributed=args.distributed)
        train_batch_sampler = make_batch_data_sampler(train_sampler, args.batch_size, args.max_iters)
        self.train_loader = data.DataLoader(dataset=trainset,
                                            batch_sampler=train_batch_sampler,
                                            num_workers=args.workers,
                                            pin_memory=True)

        if not args.skip_val:
            valset = get_segmentation_dataset(args.dataset, split='val', mode='val', **data_kwargs)
            val_sampler = make_data_sampler(valset, False, args.distributed)
            # val_batch_sampler = make_batch_data_sampler(val_sampler, args.batch_size)  # INFO: orig
            val_batch_sampler = make_batch_data_sampler(val_sampler, 4)  # INFO: small batch size only for validation, to avoid cuda overflow TODO: use orig, if needed
            self.val_loader = data.DataLoader(dataset=valset,
                                              batch_sampler=val_batch_sampler,
                                              num_workers=args.workers,
                                              pin_memory=True)

        # create network
        BatchNorm2d = nn.SyncBatchNorm if args.distributed else nn.BatchNorm2d
        self.model = get_segmentation_model(args.model, dataset=args.dataset,
                                            aux=args.aux, norm_layer=BatchNorm2d).to(self.device)

        # resume checkpoint if needed
        if args.resume:
            if os.path.isfile(args.resume):
                name, ext = os.path.splitext(args.resume)
                assert ext == '.pkl' or '.pth', 'Sorry only .pth and .pkl files supported.'
                self.model.load_state_dict(torch.load(args.resume, map_location=lambda storage, loc: storage))
                print('\nLOADED {}\n'.format(args.resume))

        # INFO: mine. Reinitialize the final layer with 4 output channels (only when loading pretrained state)
        # if self.args.dataset == 'elements':
        #     self.model.head.project = nn.Conv2d(128, 4, 1)
        #     self.model.to(self.device)

        # INFO: mine.
        # The whole model consists of 2 large parts, named "pretrained" (MobileNetV3/encoder) and "head" (Segmentation Head/decoder)
        # Therefore, for transfer learning, the encoder should be frozen (having index 0).
        # Turns out, the head has very few trainable parameters -- so maybe TL won't achieve great results
        # and fine tuning should be done very soon.

        # print_trainable_state(self.model)
        # print('******************************************')
        # print('******************************************')

        # This freezing should be applied after every call of model.train().
        # But it's also applied here to initialize the optimizer correctly (maybe not needed, just in case).
        freeze_encoder(self.model)

        # print_trainable_state(self.model)

        # create criterion
        if args.ohem:
            min_kept = int(args.batch_size // args.num_gpus * args.crop_size ** 2 // 16)
            self.criterion = MixSoftmaxCrossEntropyOHEMLoss(args.aux, args.aux_weight, min_kept=min_kept,
                                                            ignore_index=-1).to(self.device)
        else:
            self.criterion = MixSoftmaxCrossEntropyLoss(args.aux, args.aux_weight, ignore_index=-1).to(self.device)

        # optimizer; INFO: parameters are filtered by requires_grad
        self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()),
                                         lr=args.lr,
                                         momentum=args.momentum,
                                         weight_decay=args.weight_decay)
        # lr scheduling
        self.lr_scheduler = WarmupPolyLR(self.optimizer,
                                         max_iters=args.max_iters,
                                         power=0.9,
                                         warmup_factor=args.warmup_factor,
                                         warmup_iters=args.warmup_iters,
                                         warmup_method=args.warmup_method)

        if args.distributed:
            self.model = nn.parallel.DistributedDataParallel(self.model,
                                                             device_ids=[args.local_rank],
                                                             output_device=args.local_rank,
                                                             find_unused_parameters=True)

        # evaluation metrics
        self.metric = SegmentationMetric(trainset.num_class)

        self.best_pred = 0.0

    def train(self):
        # save_to_disk = get_rank() == 0
        # epochs, max_iters = self.args.epochs, self.args.max_iters
        # log_per_iters, val_per_iters = self.args.log_iter, self.args.val_epoch * self.args.iters_per_epoch
        epochs_per_val = self.args.val_epoch
        # save_per_iters = self.args.save_epoch * self.args.iters_per_epoch
        iter_per_epoch = self.args.iters_per_epoch
        epoch = self.args.start_epoch

        # logger.info('Start training, Total Epochs: {:d} = Total Iterations {:d}'.format(epochs, max_iters))
        # logger.info('batch size: {} | workers: {}\n'.format(self.args.batch_size, self.args.workers))

        self.model.train()
        freeze_encoder(self.model)

        start = time.time()
        # duration_all_iterations = 0
        # print(iter_per_epoch)

        for iteration, (images, targets) in enumerate(self.train_loader):
            iteration += 1  # TODO: why iteration is not incremented automatically?

            # start_iter_time = time.time()

            images = images.to(self.device)
            targets = targets.to(self.device)
            outputs = self.model(images)  # INFO: outputs have 4 channels (1 for alpha)
            loss_dict = self.criterion(outputs, targets)
            losses = sum(loss for loss in loss_dict.values())
            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()
            self.metric.update(outputs[0], targets)

            # log per-iteration stats
            # iter_duration = time.time() - start_iter_time
            # iter_pixAcc, iter_mIoU = self.metric.get()
            # logger.info('iteration {} || iter_time: {} | iter_pixAcc: {} | iter_mIoU: {} | iter_combined: {}'
            #             .format(iteration, iter_duration, iter_pixAcc, iter_mIoU, iter_pixAcc + iter_mIoU))
            #
            # duration_all_iterations += iter_duration

            # at the end of the current epoch:
            if iteration % iter_per_epoch == 0:
                epoch_duration = time.time() - start
                training_loss = float(losses)
                pixAcc, mIoU = self.metric.get()
                logger.info('Epoch: {0:03d} | Time: {1:.4f} | Avg. loss: {2:.4f} | pixAcc: {3:.4f} | Mean IoU: {4:.4f} | combined: {5:.4f}'
                            .format(epoch, epoch_duration, training_loss, pixAcc, mIoU, pixAcc + mIoU))

                # validate each n epochs (n = epochs_per_val):
                if epoch % epochs_per_val == 0:
                    self.validation(epoch)
                    self.model.train()
                    freeze_encoder(self.model)

                epoch += 1  # increment epoch counter
                start = time.time()  # reset time
                # torch.cuda.empty_cache()

            self.lr_scheduler.step()

        # save_checkpoint(self.model, self.args, is_best=False)
        # total_training_time = time.time() - start_time
        # total_training_str = str(datetime.timedelta(seconds=total_training_time))
        # logger.info(
        #     "Total training time: {} ({:.4f}s / it)".format(
        #         total_training_str, total_training_time / max_iters))

            img_load_start = time.time()

    def validation(self, epoch):
        # total_inter, total_union, total_correct, total_label = 0, 0, 0, 0
        # is_best = False
        gc.collect()
        torch.cuda.empty_cache()
        self.metric.reset()
        if self.args.distributed:
            model = self.model.module
        else:
            model = self.model

        val_loss = 0.0

        model.eval()
        start = time.time()

        for i, (image, target) in enumerate(self.val_loader):
            image = image.to(self.device)
            target = target.to(self.device)

            with torch.no_grad():
                outputs = model(image)
                self.metric.update(outputs[0], target)
                loss_dict = self.criterion(outputs, target)
                losses = sum(loss for loss in loss_dict.values())

            val_loss += float(losses)

        duration = time.time() - start
        avg_val_loss = val_loss / len(self.val_loader)
        pixAcc, mIoU = self.metric.get()

        logger.info('VALIDATION | Time: {0:.4f} | Avg. loss: {1:.4f} | pixAcc: {2:.4f} | Mean IoU: {3:.4f} | combined: {4:.4f}'
                    .format(duration, avg_val_loss, pixAcc, mIoU, pixAcc + mIoU))

        new_pred = (pixAcc + mIoU) / 2  # TODO: describe that pixel accuracy is also used
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
            save_checkpoint(self.model, self.args, epoch, is_best)
            logger.info('MODEL SAVED')
        else:
            self.no_improvement += 1
        logger.info('')
        torch.cuda.empty_cache()
        if self.no_improvement > self.args.max_nonimprovement:
            logger.info('Max non-improvement {} exceeded. Training stopped!'.format(self.args.max_nonimprovement))
            exit()


def save_checkpoint(model, args, epoch, is_best=True):
    """Save Checkpoint"""
    directory = os.path.expanduser(args.save_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)
    # filename = '{}_{}_epoch_{}.pth'.format(args.model, args.dataset, epoch)
    filename = 'saved_state_{}.pth'.format(epoch)
    filename = os.path.join(directory, filename)

    if args.distributed:
        model = model.module
    torch.save(model.state_dict(), filename)
    if is_best:
        best_filename = 'best_model.pth'
        best_filename = os.path.join(directory, best_filename)
        shutil.copyfile(filename, best_filename)


if __name__ == '__main__':
    args = parse_args()

    torch.cuda.empty_cache()

    # reference maskrcnn-benchmark
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.num_gpus = num_gpus
    args.distributed = num_gpus > 1
    if not args.no_cuda and torch.cuda.is_available():
        cudnn.benchmark = True
        args.device = "cuda"
    else:
        args.distributed = False
        args.device = "cpu"
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()
    # args.lr = args.lr * num_gpus  # INFO: orig

    logger = setup_logger(args.model, args.log_dir, get_rank(), filename='{}_{}_log.txt'.format(
        args.model, args.dataset))
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)
    logger.info('\n')

    trainer = Trainer(args)

    # Examine optimal combination of workers and batch size
    # for batch_size in (2, 4, 8, 16, 32):
    #     args.batch_size = batch_size
    #     for num_workers in range(5):
    #         args.workers = num_workers
    #         try:
    #             trainer.train()
    #         except:
    #             print('ERROR')
    #             continue
    # exit()

    trainer.train()
    torch.cuda.empty_cache()
