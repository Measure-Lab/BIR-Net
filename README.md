# Dual Cross-current Neural Networks (DCNN)

## Install

First, install PyTorch 1.7.0+ and torchvision 0.8.1+ and [pytorch-image-models 0.3.2](https://github.com/rwightman/pytorch-image-models):

```
conda install -c pytorch pytorch torchvision
pip install timm==0.3.2
```

## Data preparation

Download and extract ImageNet train and val images from http://image-net.org/.
The directory structure is the standard layout for the torchvision [`datasets.ImageFolder`](https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder), and the training and validation data is expected to be in the `train/` folder and `val` folder respectively:

```
/path/to/imagenet/
  train/
    class1/
      img1.jpeg
    class2/
      img2.jpeg
  val/
    class1/
      img3.jpeg
    class/2
      img4.jpeg
```

## Training and test
### Training
To train DCNN: All the results were evaluated on a system with a NVIDIA GeForce RTX 4090 GPU with 24 GB of memory, running on Ubuntu20.01 (64-bit).

Firstly, modify the --data-path parameter in run_train_dcnn.sh to the root path of ImageNet1k
```
--data-path /path/to/imagenet/
```
Then run the follow scripts to train DCNN.
```
bash run_train_dcnn.sh
```
The trained models will be saved in the OUTPUT folder, OUTPUT is defined in run_train_dcnn.sh

### Testing
Modify the --data-path parameter in run_test_dcnn.sh to the root path of ImageNet1k
```
--data-path /path/to/imagenet/
```
Then test DCNN on ImageNet on a single gpu run:
```
bash run_test_dcnn.sh
```

### Few-Shot Image Classification
We provide four datasets to finetune the base model, [CUB200](https://www.vision.caltech.edu/datasets/cub_200_2011/), [Food101](https://www.kaggle.com/datasets/dansbecker/food-101/download?datasetVersionNumber=1), [oxfordFlowers](https://www.kaggle.com/datasets/nunenuh/pytorch-challange-flower-dataset/download?datasetVersionNumber=3) and [StanfordDogs](https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset/download?datasetVersionNumber=2), dowdload them and put them in anywhere like /path/to/dataset

Run the following commands separately.
```
bash run_dcnn_CUB200.sh
bash run_dcnn_Food101.sh
bash run_dcnn_oxfordFlowers.sh
bash run_dcnn_StanfordDogs.sh
```
Before running, modify the --data-set parameter to your dataset path.
```
--data-path /path/to/dataset/
```

For training, comment out the code of the Train section.

For validation, comment out the code of the Inference section.



# Proposed model evaluation on ImageNet1k
| Model  | Parameters | MACs   | Top-1 Acc | Link                                                                                                                                                                            |
|--------|------------|--------|-----------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| DCNN   | 52.68 M    | 11.0 G | 82.3 %    | [Google Drive](https://drive.google.com/file/d/1DcYmyYM0enILJjaBT7gEMzPFvEftiQ83/view?usp=drive_link), [Baidu Netdisk](https://pan.baidu.com/s/14VYegT_o4kHkFyskvLhr0w?pwd=51t2) |
| DCNN-B | 88.47 M    | 16.1 G | 84.0 %    | Release later                                                                                                                                                                   |
| DCNN-L | 206.09 M   | 42.0 G | 84.9 %    | Release later                                                                                                                                                                   |

# DCNN Test Results on fine-grained benchmarks
| Datasets          | Top-1 Acc | Top-5 Acc | Link |
|-------------------| ----------| --------- |---- |
| Oxford Flowers102 |   97.3%   |   99.1%   | [Google Drive](https://drive.google.com/file/d/17_nIoVZsRRztMOcRpMvs9A_kenwfzj3D/view?usp=drive_link), [Baidu Netdisk](https://pan.baidu.com/s/1d7hxg3JUGgpGMrhhiU5gCA?pwd=25c0)|
| Food101           |   84.4%   |   96.2%   | [Google Drive](https://drive.google.com/file/d/1ZGy1bHgbqjtEXPmo3DdiaqFo-rjk76Nf/view?usp=drive_link), [Baidu Netdisk](https://pan.baidu.com/s/1d36wszGlBeaqbaDtpeKwTg?pwd=7axm)|
| Stanford Dogs     |   87.4%   |   98.7%   | [Google Drive](https://drive.google.com/file/d/1WDBknPtvIfi7ejKCE3yKYwc1xUhUQUIt/view?usp=drive_link), [Baidu Netdisk](https://pan.baidu.com/s/1xFkFIM9I2uGT7cABANMk_g?pwd=3oqo)|
| CUB200-2011       |   70.5%   |   91.3%   | [Google Drive](https://drive.google.com/file/d/1yvVQV0N87-jrwHLdxuLI1x9IU3yoHGU0/view?usp=drive_link), [Baidu Netdisk](https://pan.baidu.com/s/1KdmRWzP8h7jCMN3DnhJc7A?pwd=hdbc)|

# DCNN Test Results on applied medical datasets
| Datasets        | Accuracy | Link          |
|-----------------|----------|---------------|
| Brain tumor MRI | 90.7%    | Release later |
| Pneumonia CT    | 93.2%    | Release later |

