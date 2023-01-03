# FedTAN Implementation
This is an implementation of the following paper:
> Yanmeng Wang, Qingjiang Shi, Tsung-Hui Chang.
**Why Batch Normalization Damage Federated Learning on Non-IID Data?**

**Abstract**: As a promising distributed learning paradigm, federated learning (FL) involves training deep neural network (DNN) models at the network edge while protecting the privacy of the edge clients. To train a large-scale DNN model, batch normalization (BN) has been regarded as a simple and effective means to accelerate the training and improve the generalization capability. However, recent findings indicate that BN can significantly impair the performance of FL in the presence of non-i.i.d. data. While several FL algorithms have been proposed to overcome this problem, their performance still falls significantly when compared to the centralized scheme. Furthermore, none of them have provided a theoretical explanation of how the BN affects the FL convergence. In this paper, we present the first convergence analysis to show that under the non-i.i.d. data, the mismatch between the local and global statistical parameters in BN causes the gradient deviation between the local and global models, which, as a result, slows down and biases the FL convergence. As a result, we develop a new FL algorithm that is tailored to BN, called FedTAN, which is capable of achieving robust FL performance under a variety of data distributions via iterative layer-wise parameter aggregation. Comprehensive experimental results demonstrate the superiority of the proposed FedTAN over existing baselines for training BN-based DNN models.

## Requirements
The implementation runs on:
- Python 3.8
- PyTorch 1.8.1
- CUDA 11.1

## Federated Learning Algorithm for Batch Normalization
Currently, this repository supports the following federated learning algorithm:
- FedTAN (Federated learning algorithm tailored for batch normalization): perform iterative layer-wise parameter aggregation
