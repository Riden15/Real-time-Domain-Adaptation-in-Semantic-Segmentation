#!/usr/bin/python
# -*- encoding: utf-8 -*-
from torchinfo import summary
from torchvision.transforms import v2
from timeit import default_timer as timer
from datasets.gta import Gta
from model.model_stages import BiSeNet
from datasets.cityscapes import CityScapes
import torch
from torch.utils.data import DataLoader
import logging
import argparse
import numpy as np
from tensorboardX import SummaryWriter
import torch.cuda.amp as amp
from utils import poly_lr_scheduler
from utils import reverse_one_hot, compute_global_accuracy, fast_hist, per_class_iu
from tqdm import tqdm

logger = logging.getLogger()


def val(args, model, dataloader):
    print('start val!')
    with torch.no_grad():
        model.eval()
        precision_record = []
        hist = np.zeros((args.num_classes, args.num_classes))
        for i, (data, label) in enumerate(dataloader):
            label = label.type(torch.LongTensor)
            data = data.cuda()
            label = label.long().cuda()

            # get RGB predict image
            predict, _, _ = model(data)
            predict = predict.squeeze(0)
            predict = reverse_one_hot(predict)
            predict = np.array(predict.cpu())

            # get RGB label image
            label = label.squeeze()
            label = np.array(label.cpu())

            # compute per pixel accuracy
            precision = compute_global_accuracy(predict, label)
            hist += fast_hist(label.flatten(), predict.flatten(), args.num_classes)

            # there is no need to transform the one-hot array to visual RGB array
            # predict = colour_code_segmentation(np.array(predict), label_info)
            # label = colour_code_segmentation(np.array(label), label_info)
            precision_record.append(precision)

        precision = np.mean(precision_record)
        miou_list = per_class_iu(hist)
        miou = np.mean(miou_list)
        print('precision per pixel for test: %.3f' % precision)
        print('mIoU for validation: %.3f' % miou)
        print(f'mIoU per class: {miou_list}')

        return precision, miou


def train(args, model, optimizer, dataloader_train, dataloader_val):
    writer = SummaryWriter(logdir=args.tensor_board_path, comment=''.format(args.optimizer))

    scaler = amp.GradScaler()

    # se ho capito bene, il 255 è il valore che rappresenta la classe void e che quindi
    # non deve essere considerato nella loss function.
    loss_func = torch.nn.CrossEntropyLoss(ignore_index=255)
    max_miou = 0
    step = 0
    train_times = []

    for epoch in range(args.num_epochs):
        train_time_start = timer()
        lr = poly_lr_scheduler(optimizer, args.learning_rate, iter=epoch, max_iter=args.num_epochs)
        model.train()
        tq = tqdm(total=len(dataloader_train) * args.batch_size)
        tq.set_description('epoch %d, lr %f' % (epoch, lr))
        loss_record = []
        for i, (data, label) in enumerate(dataloader_train):
            data = data.cuda()
            label = label.long().cuda()
            optimizer.zero_grad()

            with amp.autocast():
                output, out16, out32 = model(data)
                loss1 = loss_func(output, label.squeeze(1))
                loss2 = loss_func(out16, label.squeeze(1))
                loss3 = loss_func(out32, label.squeeze(1))
                loss = loss1 + loss2 + loss3

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            tq.update(args.batch_size)
            tq.set_postfix(loss='%.6f' % loss)
            step += 1
            writer.add_scalar('loss_step', loss, step)
            loss_record.append(loss.item())
        tq.close()
        loss_train_mean = np.mean(loss_record)
        writer.add_scalar('epoch/loss_epoch_train', float(loss_train_mean), epoch)
        print('loss for train : %f' % (loss_train_mean))
        if epoch % args.checkpoint_step == 0 and epoch != 0:
            import os
            if not os.path.isdir(args.save_model_path):
                os.mkdir(args.save_model_path)
            torch.save(model.module.state_dict(), os.path.join(args.save_model_path, 'latest.pth'))

        if epoch % args.validation_step == 0 and epoch != 0:
            precision, miou = val(args, model, dataloader_val)
            if miou > max_miou:
                max_miou = miou
                import os
                os.makedirs(args.save_model_path, exist_ok=True)
                torch.save(model.module.state_dict(), os.path.join(args.save_model_path, 'best.pth'))
            writer.add_scalar('epoch/precision_val', precision, epoch)
            writer.add_scalar('epoch/miou val', miou, epoch)
        train_time_end = timer()
        train_times.append(train_time_end - train_time_start)

    print(f'Average train time per epoch in minutes: {np.mean(train_times) / 60}')


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def parse_args():
    parse = argparse.ArgumentParser()

    parse.add_argument('--mode',
                       dest='mode',
                       type=str,
                       default='train',
                       )
    parse.add_argument('--backbone',
                       dest='backbone',
                       type=str,
                       default='CatmodelSmall',
                       )
    parse.add_argument('--train_dataset',
                       dest='train_dataset',
                       type=str,
                       default='Cityscapes',
                       )
    parse.add_argument('--val_dataset',
                       dest='val_dataset',
                       type=str,
                       default='Cityscapes',
                       )
    parse.add_argument('--pretrain_path',
                       dest='pretrain_path',
                       type=str,
                       default='',
                       )
    parse.add_argument('--save_model_path',
                       type=str,
                       default=None,
                       help='path to save model')
    parse.add_argument('--use_conv_last',
                       dest='use_conv_last',
                       type=str2bool,
                       default=False,
                       )
    parse.add_argument('--num_epochs',
                       type=int,
                       default=300,
                       help='Number of epochs to train for')
    parse.add_argument('--epoch_start_i',
                       type=int,
                       default=0,
                       help='Start counting epochs from this number')
    parse.add_argument('--checkpoint_step',
                       type=int,
                       default=10,
                       help='How often to save checkpoints (epochs)')
    parse.add_argument('--validation_step',
                       type=int,
                       default=1,
                       help='How often to perform validation (epochs)')
    parse.add_argument('--crop_height',
                       type=int,
                       default=512,
                       help='Height of cropped/resized input image to modelwork')
    parse.add_argument('--crop_width',
                       type=int,
                       default=1024,
                       help='Width of cropped/resized input image to modelwork')
    parse.add_argument('--batch_size',
                       type=int,
                       default=2,
                       help='Number of images in each batch')
    parse.add_argument('--learning_rate',
                       type=float,
                       default=0.01,
                       help='learning rate used for train')
    parse.add_argument('--num_workers',
                       type=int,
                       default=4,
                       help='num of workers')
    parse.add_argument('--num_classes',
                       type=int,
                       default=19,
                       help='num of object classes (with void)')
    parse.add_argument('--cuda',
                       type=str,
                       default='0',
                       help='GPU ids used for training')
    parse.add_argument('--use_gpu',
                       type=bool,
                       default=True,
                       help='whether to user gpu for training')
    parse.add_argument('--tensor_board_path',
                       type=str,
                       default='runs',
                       help='path to save graph for TensorBoard')
    parse.add_argument('--optimizer',
                       type=str,
                       default='adam',
                       help='optimizer, support rmsprop, sgd, adam')
    parse.add_argument('--loss',
                       type=str,
                       default='crossentropy',
                       help='loss function')

    return parse.parse_args()


def main():
    args = parse_args()

    n_classes = args.num_classes
    mode = args.mode

    # model
    model = BiSeNet(backbone=args.backbone, n_classes=n_classes, pretrain_model=args.pretrain_path,
                    use_conv_last=args.use_conv_last)

    if mode == 'train':

        # dataset class
        if args.train_dataset == 'Cityscapes' and args.val_dataset == 'Cityscapes':

            train_dataset = CityScapes(mode, transformations=True, args=args)
            val_dataset = CityScapes(mode='val', transformations=True, args=args)

        elif args.train_dataset == 'GTA' and args.val_dataset == 'GTA':

            train_dataset = Gta(transformations=True, args=args)
            val_dataset = Gta(transformations=True, args=args)

        elif args.train_dataset == 'GTA_aug' and args.val_dataset == 'GTA':

            train_dataset = Gta(transformations=True, data_augmentation=True, args=args)
            val_dataset = Gta(transformations=True, args=args)

        elif args.train_dataset == 'Gta_aug' and args.val_dataset == 'Cityscapes':
            train_dataset = Gta(transformations=True, data_augmentation=True, args=args)
            val_dataset = CityScapes(mode='val', transformations=True, args=args)
        else:
            raise ValueError('Dataset not supported')

        # dataloader class
        dataloader_train = DataLoader(train_dataset,
                                      batch_size=args.batch_size,
                                      shuffle=True,
                                      num_workers=args.num_workers,
                                      pin_memory=False,
                                      drop_last=True)

        dataloader_val = DataLoader(val_dataset,
                                    batch_size=1,
                                    shuffle=True,
                                    num_workers=args.num_workers,
                                    drop_last=False)

        # optimizer
        if args.optimizer == 'rmsprop':
            optimizer = torch.optim.RMSprop(model.parameters(), args.learning_rate)
        elif args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=0.9, weight_decay=5e-4)
        elif args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
        else:  # rmsprop
            print('not supported optimizer \n')
            return None

        # load model to gpu
        if torch.cuda.is_available() and args.use_gpu:
            model = torch.nn.DataParallel(model).cuda()

        # train loop
        train(args, model, optimizer, dataloader_train, dataloader_val)

        # final test
        val(args, model, dataloader_val)

    elif mode == 'val':

        if args.val_dataset == 'Cityscapes':
            val_dataset = CityScapes(mode='val', transformations=True, args=args)
        elif args.val_dataset == 'GTA':
            val_dataset = Gta(transformations=True, args=args)
        else:
            raise ValueError('Dataset not supported')

        dataloader_val = DataLoader(val_dataset,
                                    batch_size=1,
                                    shuffle=True,
                                    num_workers=args.num_workers,
                                    drop_last=False)
        if args.save_model_path is not None:
            # Load in the saved state_dict()
            model.load_state_dict(torch.load(f=args.save_model_path))
        else:
            raise ValueError('save_model_path must be specified')

        # load model to gpu
        if torch.cuda.is_available() and args.use_gpu:
            model = torch.nn.DataParallel(model).cuda()

        # final test
        val(args, model, dataloader_val)


if __name__ == "__main__":
    main()
