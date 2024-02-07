import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
import numpy as np
from tensorboardX import SummaryWriter
import torch.cuda.amp as amp
from utils import poly_lr_scheduler
from tqdm import tqdm
from train import val
from timeit import default_timer as timer

lambda_adv = 0.001  # Define the weight of the adversarial loss
lambda_seg = 1  # Define the weight of the segmentation loss

logger = logging.getLogger()


def train(args, G, D, optimizer_G, optimizer_D, dataloader_gta5, dataloader_cityscapes, dataloader_val_cityscapes):
    # torch.autograd.set_detect_anomaly(True)
    writer = SummaryWriter(comment=''.format(args.optimizer))

    scaler = amp.GradScaler()  # Initialize gradient scaler for mixed precision training

    loss_func_seg = torch.nn.CrossEntropyLoss(ignore_index=255)  # Define the loss function for the segmentation model
    loss_func_d = nn.BCEWithLogitsLoss()  # Define the loss function for the discriminator
    loss_func_adv = nn.BCEWithLogitsLoss()  # Define the loss function for the adversarial loss

    max_miou = 0  # Variable to store the maximum mean IoU
    step = 0  # Variable to count training steps
    train_times = []

    for epoch in range(args.num_epochs):
        train_time_start = timer()
        lr_G = poly_lr_scheduler(optimizer_G, args.learning_rate, iter=epoch,
                                 max_iter=args.num_epochs)  # Update learning rate
        lr_D = poly_lr_scheduler(optimizer_D, args.learning_rate, iter=epoch,
                                 max_iter=args.num_epochs)  # Update learning rate
        tq = tqdm(
            total=len(dataloader_gta5) + len(dataloader_cityscapes) * args.batch_size)  # Initialize tqdm progress bar
        tq.set_description('epoch %d, lr_G %f,  lr_D%f' % (epoch, lr_G, lr_D))
        loss_record = []  # List to record loss values

        for i, (data_gta5, data_cityscapes) in enumerate(zip(dataloader_gta5, dataloader_cityscapes)):
            data_gta5, label_gta5 = data_gta5  # Unpack GTA5 datasets
            data_gta5 = data_gta5.cuda()  # Move GTA5 images to GPU
            label_gta5 = label_gta5.cuda()  # Move GTA5 labels to GPU
            data_cityscapes, _ = data_cityscapes  # Unpack Cityscapes datasets
            data_cityscapes = data_cityscapes.cuda()  # Move Cityscapes datasets to GPU
            optimizer_G.zero_grad()  # Zero the gradients

            G.train()  # Set the model to training mode
            # Train the segmentation model with GTA5 datasets
            with amp.autocast():
                output_gta5, out16_gta5, out32_gta5 = G(data_gta5)  # Get predictions from the model at multiple scales
                # Calculate loss at multiple scales
                loss1_gta5 = loss_func_seg(output_gta5, label_gta5.squeeze(1))
                loss2_gta5 = loss_func_seg(out16_gta5, label_gta5.squeeze(1))
                loss3_gta5 = loss_func_seg(out32_gta5, label_gta5.squeeze(1))
                loss_gta5 = loss1_gta5 + loss2_gta5 + loss3_gta5  # Combine losses

                #Get predictions from the segmentation model on Cityscapes datasets
                output_cityscapes, out16_cityscapes, out32_cityscapes = G(data_cityscapes)
                
            scaler.scale(loss_gta5).backward()  # Scale loss and perform backpropagation
            scaler.step(optimizer_G)  # Perform optimizer step


            # Train the discriminator with GTA5 datasets
            for param in D.parameters():
                param.requires_grad = False

            d_label_cityscapes = torch.zeros(d_cityscapes.size(0), 1, d_cityscapes.size(2),
                                             d_cityscapes.size(3)).cuda()  # Labels are 0 for Cityscapes datasets            
            with amp.autocast():
                d_cityscapes = D(output_cityscapes)
                # d16_cityscapes = D(out16_cityscapes)
                # d32_cityscapes = D(out32_cityscapes)
                
                # Calculate the adversarial loss
                loss_adv_cityscapes = loss_func_adv(d_cityscapes, d_label_cityscapes)
                # loss_adv_cityscapes16 = loss_func_adv(d16_cityscapes, d_label_cityscapes)
                # loss_adv_cityscapes32 = loss_func_adv(d32_cityscapes, d_label_cityscapes)

            #loss_adv = loss_adv_cityscapes * lambda_adv + loss_adv_cityscapes16 * lambda_adv + loss_adv_cityscapes32 * lambda_adv
            loss_adv = loss_adv_cityscapes * lambda_adv
            #optimizer_G.zero_grad()  # Zero the gradients
            # backpropagation for adversarial loss to G model and not D model
            scaler.scale(loss_adv).backward()
            scaler.step(optimizer_G)
            scaler.update()
            
            total_loss = loss_gta5 * lambda_seg + loss_adv  # Combine segmentation and adversarial losses
            
            # bring back requires_grad
            for param in D.parameters():
                param.requires_grad = True
                
            with amp.autocast():
                # Forward pass of GTA5 datasets through the discriminator
                d_gta5 = D(output_gta5.detach())
                # d16_gta5 = D(out16_gta5.detach())
                # d32_gta5 = D(out32_gta5.detach())
                # Calculate loss for GTA5 datasets
                d_label_gta5 = torch.ones(d_gta5.size(0), 1, d_gta5.size(2),
                                        d_gta5.size(3)).cuda()  # Labels are 1 for GTA5 datasets
                loss_d_gta5 = loss_func_d(d_gta5, d_label_gta5)
                # loss_d_gta5 += loss_func_d(d16_gta5, d_label_gta5)
                # loss_d_gta5 += loss_func_d(d32_gta5, d_label_gta5)
                
            optimizer_D.zero_grad()  # Zero the gradients
            scaler.scale(loss_d_gta5).backward()  # Scale loss and perform backpropagation
            scaler.step(optimizer_D)  # Perform optimizer step

            with amp.autocast():
                # Train the discriminator with Cityscapes datasets
                # Forward pass of Cityscapes datasets through the discriminator
                d_cityscapes = D(output_cityscapes.detach())
                # d16_cityscapes = D(out16_cityscapes.detach())
                # d32_cityscapes = D(out32_cityscapes.detach())
                # Calculate loss for Cityscapes datasets
                
                loss_d_cityscapes = loss_func_d(d_cityscapes, d_label_cityscapes)
                # loss_d_cityscapes += loss_func_d(d16_cityscapes, d_label_cityscapes)
                # loss_d_cityscapes += loss_func_d(d32_cityscapes, d_label_cityscapes)

            optimizer_D.zero_grad()  # Zero the gradients
            scaler.scale(loss_d_cityscapes).backward()  # Scale loss and perform backpropagation
            scaler.step(optimizer_D)  # Perform optimizer step

            tq.update(args.batch_size)
            tq.set_postfix(loss='%.6f' % total_loss)
            # tq.set_postfix(loss='%.6f' % total_loss)
            step += 1
            writer.add_scalar('seg_loss_step', loss_gta5, step)
            writer.add_scalar('adv_loss_step', loss_adv, step)
            writer.add_scalar('total_loss_step', total_loss, step)
            # writer.add_scalar('loss_step', total_loss, step)
            loss_record.append(total_loss.item())
            # loss_record.append(total_loss.item())

        tq.close()
        loss_train_mean = np.mean(loss_record)
        writer.add_scalar('epoch/loss_epoch_train', float(loss_train_mean), epoch)

        print('loss for train : %f' % (loss_train_mean))
        if epoch % args.checkpoint_step == 0 and epoch != 0:
            import os
            if not os.path.isdir(args.save_model_path):
                os.mkdir(args.save_model_path)
            torch.save(G.module.state_dict(), os.path.join(args.save_model_path, 'G_latest.pth'))
            torch.save(D.module.state_dict(), os.path.join(args.save_model_path, 'D_latest.pth'))

        if epoch % args.validation_step == 0 and epoch != 0:
            precision, miou = val(args, G, dataloader_val_cityscapes)
            if miou > max_miou:
                max_miou = miou
                import os
                os.makedirs(args.save_model_path, exist_ok=True)
                torch.save(G.module.state_dict(), os.path.join(args.save_model_path, 'G_best.pth'))
            writer.add_scalar('epoch/precision_val', precision, epoch)
            writer.add_scalar('epoch/miou val', miou, epoch)
        train_time_end = timer()
        train_times.append(train_time_end - train_time_start)

    print(f'Average train time per epoch in minutes: {np.mean(train_times) / 60}')
