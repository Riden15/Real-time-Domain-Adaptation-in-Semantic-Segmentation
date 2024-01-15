import torch
from torch.utils.data import DataLoader
import logging
import numpy as np
from tensorboardX import SummaryWriter
import torch.cuda.amp as amp
from utils import poly_lr_scheduler
from tqdm import tqdm


logger = logging.getLogger()

def train(args, model, optimizer, dataloader_train):
    writer = SummaryWriter(comment=''.format(args.optimizer))

    scaler = amp.GradScaler()# Initialize gradient scaler for mixed precision training

    loss_func = torch.nn.CrossEntropyLoss(ignore_index=255)  # Define the loss function
    max_miou = 0  # Variable to store the maximum mean IoU
    step = 0  # Variable to count training steps
    for epoch in range(args.num_epochs):
        lr = poly_lr_scheduler(optimizer, args.learning_rate, iter=epoch, max_iter=args.num_epochs)  # Update learning rate
        model.train()  # Set the model to training mode
        tq = tqdm(total=len(dataloader_train) * args.batch_size)  # Initialize tqdm progress bar
        tq.set_description('epoch %d, lr %f' % (epoch, lr))
        loss_record = []  # List to record loss values
        for i, (data, label) in enumerate(dataloader_train):
            data = data.cuda() # Move data to GPU
            label = label.long().cuda() # Move labels to GPU
            optimizer.zero_grad() # Zero the gradients

            with amp.autocast():
                output, out16, out32 = model(data)  # Get predictions from the model at multiple scales
                # Calculate loss at multiple scales
                loss1 = loss_func(output, label.squeeze(1))  
                loss2 = loss_func(out16, label.squeeze(1))
                loss3 = loss_func(out32, label.squeeze(1))
                loss = loss1 + loss2 + loss3  # Combine losses

            scaler.scale(loss).backward()  # Scale loss and perform backpropagation
            scaler.step(optimizer)  # Perform optimizer step
            scaler.update()  # Update the scaler

            tq.update(args.batch_size)  # Update progress bar
            tq.set_postfix(loss='%.6f' % loss)  # Display loss in progress bar
            step += 1
            writer.add_scalar('loss_step', loss, step)  # Write loss to tensorboard
            loss_record.append(loss.item())
            
        loss_train_mean = np.mean(loss_record)  # Calculate mean loss for the epoch
        writer.add_scalar('epoch/loss_epoch_train', float(loss_train_mean), epoch)  # Write mean loss to tensorboard
        print('loss for train : %f' % (loss_train_mean))

        # Save model checkpoint periodically
        if epoch % args.checkpoint_step == 0 and epoch != 0:
            import os
            if not os.path.isdir(args.save_model_path):
                os.mkdir(args.save_model_path)
            torch.save(model.module.state_dict(), os.path.join(args.save_model_path, 'latest.pth'))
