# Real-time Domain Adaptation in Semantic Segmentation [(Course Project)](https://drive.google.com/file/d/1amm6H_718IabI4bn0OIpsIPSyMl-vku4/view?usp=sharing)

This repository provides a starter-code setup for the Real-time Domain Adaptation in Semantic Segmentation project of the Advance Machine Learning Course.

## Package

- [datasets](https://github.com/Riden15/Real-time-Domain-Adaptation-in-Semantic-Segmentation/tree/main/datasets): Contains the dataset classes for the Cityscapes and GTA datasets. The train and validation images for both datasets will also be inserted here.
- [model](https://github.com/Riden15/Real-time-Domain-Adaptation-in-Semantic-Segmentation/tree/main/model): Contains the STDCNet model and the Discriminator model.
- [runs](https://github.com/Riden15/Real-time-Domain-Adaptation-in-Semantic-Segmentation/tree/main/runs): Contains the tensorboard logs for all the project steps.
- [saved_models](https://github.com/Riden15/Real-time-Domain-Adaptation-in-Semantic-Segmentation/tree/main/saved_models): Contains the saved models for all the project steps.
- [STDCNET_weights](https://github.com/Riden15/Real-time-Domain-Adaptation-in-Semantic-Segmentation/tree/main/STDCNET_weights): Contains the pre-trained weights for the STDCNet model.

## Requisites

- Download the pre-trained weight at this [link](https://drive.google.com/file/d/1Q5CL_z1E5y0BMt10WzE6TbZg8U2nyWUK/view?usp=drive_link) at put it in the STDCNET_weights folder.
- Download the Cityscapes dataset and the GTA dataset at this [link](https://drive.google.com/drive/u/0/folders/1iE8wJT7tuDOVjEBZ7A3tOPZmNdroqG1m) and put it in the datasets' folder.

## Steps

- **2.A**: Train the STDCNet model on the Cityscapes dataset and evaluate it on the Cityscapes dataset.
- **2.B**: Train the STDCNet model on the GTA dataset and evaluate it on the GTA dataset.
- **2.C.1**: Evaluate the best model from step 2.B on the Cityscapes dataset.
- **2.C.2**: Train the STDCNet model on the GTA augmented dataset and evaluate it on the Cityscapes dataset.
- **3**: Train the STDCNet model with unsupervised adversarial training domain adaptation with labeled synthetic data (source GTA dataset) and unlabelled real data (target Cityscapes datasets).
- **4**: Train the STDCNet model with unsupervised adversarial training domain adaptation with labeled synthetic data (source GTA dataset) and unlabelled real data (target Cityscapes datasets) using a depthwise discriminator.

## Command Line Arguments

- **2.A**: ``--train_dataset Cityscapes --val_dataset Cityscapes --pretrain_path STDCNET_weights/STDCNet813M_73.91.tar --batch_size 8 --num_epochs 50 
         --learning_rate 0.01 --crop_height 512 --crop_width 1024 --tensorboard_path runs/2_A --save_model_path saved_models/2_A --optimizer sgd --loss crossentropy``
- **2.B**: ``--train_dataset GTA --val_dataset GTA --pretrain_path STDCNET_weights/STDCNet813M_73.91.tar --batch_size 8 --num_epochs 50 --crop_height 512
        --learning_rate 0.01 --crop_width 1024 --tensorboard_path runs/2_B --save_model_path saved_models/2_B --optimizer sgd --loss crossentropy``
- **2.C.1**: ``--mode val --val_dataset Cityscapes --crop_height 512 --crop_width 1024 --save_model_path saved_models/2_B/best.pth``
- **2.C.2**: ``--train_dataset GTA_aug --val_dataset Cityscapes --pretrain_path STDCNET_weights/STDCNet813M_73.91.tar --batch_size 8 --learning_rate 0.01
          --num_epochs 50 --crop_height 512 --crop_width 1024 --tensorboard_path runs/2_C_2 --save_model_path saved_models/2_C_2 --optimizer sgd --loss crossentropy``
- **3**: ``--mode train_adversarial --pretrain_path STDCNET_weights/STDCNet813M_73.91.tar --batch_size 8 --learning_rate 0.01
      --discriminator_learning_rate 0.001 --num_epochs 50 --crop_height 512 --crop_width 1024 --tensorboard_path runs/3 --save_model_path saved_models/3``
- **4**: ``--mode train_adversarial --depthwise_discriminator true --pretrain_path STDCNET_weights/STDCNet813M_73.91.tar --batch_size 8 --learning_rate 0.01
      --discriminator_learning_rate 0.001 --num_epochs 50 --crop_height 512 --crop_width 1024 --tensorboard_path runs/4 --save_model_path saved_models/4``


## Results

| Train Datasets                                                                             | Val Datasets | Accuracy _(%)_ | mIoU _(%)_ | train Time (avg per-epochs) |
|--------------------------------------------------------------------------------------------|--------------|----------------|------------|-----------------------------|
| Cityscapes                                                                                 | Cityscapes   | 81             | 57.8       | 2:33 minutes                |
| GTA                                                                                        | GTA          | 80.8           | 62         | 3:28 minutes                |
| GTA                                                                                        | Cityscapes   | 60.1           | 24.6       | _None_                      |
| GTA augmented                                                                              | Cityscapes   | 70.2           | 30.7       | 5:22 minutes                |
| Single Layer DA <br/> Source=GTA, Target=Cityscapes                                        | Cityscapes   | 74.3           | 33.8       | 4:33 minutes                |
| Single Layer DA <br/> Source=GTA, Target=Cityscapes <br/> Depthwise discriminator function | Cityscapes   | 73.1           | 32.7       | 4:32 minutes                |