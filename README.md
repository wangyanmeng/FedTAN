# FedTAN Implementation
This is an implementation of the following paper:
> Yanmeng Wang, Qingjiang Shi, Tsung-Hui Chang.
[Why Batch Normalization Damage Federated Learning on Non-IID Data?](https://arxiv.org/abs/2301.02982)

**Abstract**: As a promising distributed learning paradigm, federated learning (FL) involves training deep neural network (DNN) models at the network edge while protecting the privacy of the edge clients. To train a large-scale DNN model, batch normalization (BN) has been regarded as a simple and effective means to accelerate the training and improve the generalization capability. However, recent findings indicate that BN can significantly impair the performance of FL in the presence of non-i.i.d. data. While several FL algorithms have been proposed to address this issue, their performance still falls significantly when compared to the centralized scheme. Furthermore, none of them have provided a theoretical explanation of how the BN damages the FL convergence. In this paper, we present the first convergence analysis to show that under the non-i.i.d. data, the mismatch between the local and global statistical parameters in BN causes the gradient deviation between the local and global models, which, as a result, slows down and biases the FL convergence. In view of this, we develop a new FL algorithm that is tailored to BN, called FedTAN, which is capable of achieving robust FL performance under a variety of data distributions via iterative layer-wise parameter aggregation. Comprehensive experimental results demonstrate the superiority of the proposed FedTAN over existing baselines for training BN-based DNN models.

## Requirements
The implementation runs on:
- Python 3.8
- PyTorch 1.8.1
- CUDA 11.1
- CIFAR-10 dataset

## Federated Learning Algorithm for Batch Normalization
Currently, this repository supports the following federated learning algorithm:
- FedTAN (Federated learning algorithm tailored for batch normalization): perform iterative layer-wise parameter aggregation.

## Launch Experiments
An example launch script is shown below.
```bash
python FedTAN.py
    --seed 0 \
    --data_distribution 1 \
    --momentum 0 \
    --weight_decay 0 \
```
Explanations of arguments:
- `seed`: random seed
- `data_distribution`: local dataset dirstribution (iid: 1; non-iid: 2)
- `momentum`: momentum parameter used in SGD optimiter
- `weight_decay`: weight decay parameter used in SGD optimiter

## About
This project is still improving, if any problems (bugs), [Issue](https://github.com/wangyanmeng/FedTAN/issues) please.
