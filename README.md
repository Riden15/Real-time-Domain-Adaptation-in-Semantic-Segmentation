# Real-time Domain Adaptation in Semantic Segmentation [(Course Project)](https://drive.google.com/file/d/1amm6H_718IabI4bn0OIpsIPSyMl-vku4/view?usp=sharing)

This repository provides a starter-code setup for the Real-time Domain Adaptation in Semantic Segmentation project of 
the Machine Learning Course.

## Command Line Arguments

- 2.A: ``--train_dataset Cityscapes --val_dataset Cityscapes --pretrain_path STDCNET_weights/STDCNet813M_73.91.tar --batch_size 8 --num_epochs 50 
         --learning_rate 0.01 --crop_height 512 --crop_width 1024 --tensorboard_path runs/2_A --save_model_path saved_models/2_A --optimizer sgd --loss crossentropy``
- 2.B: ``--train_dataset GTA --val_dataset GTA --pretrain_path STDCNET_weights/STDCNet813M_73.91.tar --batch_size 8 --num_epochs 50 --crop_height 512
        --learning_rate 0.01 --crop_width 1024 --tensorboard_path runs/2_B --save_model_path saved_models/2_B --optimizer sgd --loss crossentropy``
- 2.C.1: ``--mode val --val_dataset Cityscapes --crop_height 512 --crop_width 1024 --save_model_path saved_models/2_B/best.pth``
- 2.C.2: ``--train_dataset GTA_aug --val_dataset Cityscapes --pretrain_path STDCNET_weights/STDCNet813M_73.91.tar --batch_size 8 --learning_rate 0.01
          --num_epochs 50 --crop_height 512 --crop_width 1024 --tensorboard_path runs/2_C_2 --save_model_path saved_models/2_C_2 --optimizer sgd --loss crossentropy``
- 3: ``--mode train_adversarial --pretrain_path STDCNET_weights/STDCNet813M_73.91.tar --batch_size 8 --learning_rate 0.01
      --discriminator_learning_rate 0.001 --num_epochs 50 --crop_height 512 --crop_width 1024 --tensorboard_path runs/3 --save_model_path saved_models/3``
- 4: ``--mode train_adversarial --depthwise_discriminator true --pretrain_path STDCNET_weights/STDCNet813M_73.91.tar --batch_size 8 --learning_rate 0.01
      --discriminator_learning_rate 0.001 --num_epochs 50 --crop_height 512 --crop_width 1024 --tensorboard_path runs/4 --save_model_path saved_models/4``


|      | num sets | density | number of evaluation | best fitness |
|------|----------|---------|----------------------|--------------|
| 100  | 100      | .3      | 200                  | (100, -14)   |
| 100  | 100      | .7      | 200                  | (100, -21)   |
| 1000 | 1000     | .3      | 2.000                | (1000, -183) |
| 1000 | 1000     | .7      | 2.000                | (1000, -192) |
| 5000 | 5000     | .3      | 10.000               | (5000, -929) |
| 5000 | 5000     | .7      | 10.000               | (5000, -)    |