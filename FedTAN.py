import time
import torch.nn as nn
import torch.utils.data
import copy
import numpy as np
import os
import random
import pandas as pd
from tqdm import tqdm

from args import args_parser
from utils.evaluation import Inference_loss, Inference_accuracy # Performance measurement
from utils.random_seed import setup_seed # random seed
from utils.fedavg import average_weights # federated averaging algorithm
from utils.training import train  # training on whole network (forward and backward propagations)
from utils.datasets import dataset_preprocess, DatasetSplit
from libs.initial_models import initial_global_local_models
from libs.first_local_update import first_local_updating
from libs.global_aggregation import global_aggregation_FedSGD
from libs.transformation import subnet_2_whole, whole_2_subnet

# parameters
global args
args = args_parser()

node_number = 5 # the number of node
local_update_step_number_total = args.local_update_step_number
interval = 100 # plot interval

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # define paths
gpu_id = "3,2"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

weight_set = np.ones(node_number) / node_number # weight of local device, balanced
criterion = nn.CrossEntropyLoss().to(device)
selected_node_number = node_number # number of selected nodes in each epoch

setup_seed(args.seed)

train_dataset, train_all_loader, test_all_loader, node_groups = dataset_preprocess(node_number, weight_set, args.data_distribution, args.batch_size, args.workers)

# ---------------- initialize the global model in server and the local models in clients
global_average, global_average_weight, \
local_list_0_1, local_list_0_2, local_list_0_Mean, local_list_0_Var, \
local_list_11_1, local_list_11_2, local_list_12_2, local_list_11_Mean, local_list_11_Var, local_list_12_Mean, local_list_12_Var, \
local_list_13_1, local_list_13_2, local_list_14_2, local_list_13_Mean, local_list_13_Var, local_list_14_Mean, local_list_14_Var, \
local_list_15_1, local_list_15_2, local_list_16_2, local_list_15_Mean, local_list_15_Var, local_list_16_Mean, local_list_16_Var, \
local_list_21_1, local_list_21_2, local_list_22_2, local_list_21_Mean, local_list_21_Var, local_list_22_Mean, local_list_22_Var, \
local_list_23_1, local_list_23_2, local_list_24_2, local_list_23_Mean, local_list_23_Var, local_list_24_Mean, local_list_24_Var, \
local_list_25_1, local_list_25_2, local_list_26_2, local_list_25_Mean, local_list_25_Var, local_list_26_Mean, local_list_26_Var, \
local_list_31_1, local_list_31_2, local_list_32_2, local_list_31_Mean, local_list_31_Var, local_list_32_Mean, local_list_32_Var, \
local_list_33_1, local_list_33_2, local_list_34_2, local_list_33_Mean, local_list_33_Var, local_list_34_Mean, local_list_34_Var, \
local_list_35_1, local_list_35_2, local_list_36_2, local_list_35_Mean, local_list_35_Var, local_list_36_Mean, local_list_36_Var, \
local_list_4 \
= initial_global_local_models(device, args.half, node_number)

# initial global model performance
train_loss_DGD, test_accuracy_DGD = [], []
loss0 = Inference_loss(device,train_all_loader,criterion,global_average, args.half)
acc0 = Inference_accuracy(device,test_all_loader,global_average, args.half)
train_loss_DGD.append(loss0)
test_accuracy_DGD.append(acc0)


# select clients for each iteration
selected_local_device_index_set = np.zeros((args.epochs,selected_node_number))
for epoch in range(args.epochs):
    selected_local_device_index_set[epoch] =  np.arange(selected_node_number).astype(int) # np.random.choice(node_number, selected_node_number, replace=True, p = weight_set).astype(int)                    

# distributed training
for epoch in tqdm(range(args.epochs)):

    if epoch >= 6000:
        args.lr = 0.05
    if epoch >= 8000:
        if args.momentum > 0:
            args.lr = 0.005

    selected_local_device_index = selected_local_device_index_set[epoch].astype(int)

    # training sample index
    idxs_all = []
    for idx in selected_local_device_index:
        idxs = list(node_groups[idx])
        random.shuffle(idxs)
        idxs_all.append(idxs)

    local_list_0_1, local_list_0_2, \
    local_list_11_1, local_list_11_2, local_list_12_2, \
    local_list_13_1, local_list_13_2, local_list_14_2, \
    local_list_15_1, local_list_15_2, local_list_16_2, \
    local_list_21_1, local_list_21_2, local_list_22_2, \
    local_list_23_1, local_list_23_2, local_list_24_2, \
    local_list_25_1, local_list_25_2, local_list_26_2, \
    local_list_31_1, local_list_31_2, local_list_32_2, \
    local_list_33_1, local_list_33_2, local_list_34_2, \
    local_list_35_1, local_list_35_2, local_list_36_2, \
    local_list_4\
    = first_local_updating(local_list_0_1, local_list_0_2, local_list_0_Mean, local_list_0_Var,
                           local_list_11_1, local_list_11_2, local_list_12_2, local_list_11_Mean, local_list_11_Var, local_list_12_Mean, local_list_12_Var,
                           local_list_13_1, local_list_13_2, local_list_14_2, local_list_13_Mean, local_list_13_Var, local_list_14_Mean, local_list_14_Var,
                           local_list_15_1, local_list_15_2, local_list_16_2, local_list_15_Mean, local_list_15_Var, local_list_16_Mean, local_list_16_Var,
                           local_list_21_1, local_list_21_2, local_list_22_2, local_list_21_Mean, local_list_21_Var, local_list_22_Mean, local_list_22_Var,
                           local_list_23_1, local_list_23_2, local_list_24_2, local_list_23_Mean, local_list_23_Var, local_list_24_Mean, local_list_24_Var,
                           local_list_25_1, local_list_25_2, local_list_26_2, local_list_25_Mean, local_list_25_Var, local_list_26_Mean, local_list_26_Var,
                           local_list_31_1, local_list_31_2, local_list_32_2, local_list_31_Mean, local_list_31_Var, local_list_32_Mean, local_list_32_Var,
                           local_list_33_1, local_list_33_2, local_list_34_2, local_list_33_Mean, local_list_33_Var, local_list_34_Mean, local_list_34_Var,
                           local_list_35_1, local_list_35_2, local_list_36_2, local_list_35_Mean, local_list_35_Var, local_list_36_Mean, local_list_36_Var,
                           local_list_4,
                           selected_node_number, selected_local_device_index, idxs_all, train_dataset, criterion, args, device)


    # ---- block 0
    del local_list_0_Mean, local_list_0_Var
    # ---- block 1
    del local_list_11_Mean, local_list_11_Var, local_list_12_Mean, local_list_12_Var
    del local_list_13_Mean, local_list_13_Var, local_list_14_Mean, local_list_14_Var
    del local_list_15_Mean, local_list_15_Var, local_list_16_Mean, local_list_16_Var
    # ---- block 2
    del local_list_21_Mean, local_list_21_Var, local_list_22_Mean, local_list_22_Var
    del local_list_23_Mean, local_list_23_Var, local_list_24_Mean, local_list_24_Var
    del local_list_25_Mean, local_list_25_Var, local_list_26_Mean, local_list_26_Var
    # ---- block 3
    del local_list_31_Mean, local_list_31_Var, local_list_32_Mean, local_list_32_Var
    del local_list_33_Mean, local_list_33_Var, local_list_34_Mean, local_list_34_Var
    del local_list_35_Mean, local_list_35_Var, local_list_36_Mean, local_list_36_Var


    # ========================== only 1 local updating steps ========================== #
    if local_update_step_number_total == 1:

        # global aggregation
        global_average, global_average_weight, \
        local_list_0_1, local_list_0_2, local_list_0_Mean, local_list_0_Var, \
        local_list_11_1, local_list_11_2, local_list_12_2, local_list_11_Mean, local_list_11_Var, local_list_12_Mean, local_list_12_Var, \
        local_list_13_1, local_list_13_2, local_list_14_2, local_list_13_Mean, local_list_13_Var, local_list_14_Mean, local_list_14_Var, \
        local_list_15_1, local_list_15_2, local_list_16_2, local_list_15_Mean, local_list_15_Var, local_list_16_Mean, local_list_16_Var, \
        local_list_21_1, local_list_21_2, local_list_22_2, local_list_21_Mean, local_list_21_Var, local_list_22_Mean, local_list_22_Var, \
        local_list_23_1, local_list_23_2, local_list_24_2, local_list_23_Mean, local_list_23_Var, local_list_24_Mean, local_list_24_Var, \
        local_list_25_1, local_list_25_2, local_list_26_2, local_list_25_Mean, local_list_25_Var, local_list_26_Mean, local_list_26_Var, \
        local_list_31_1, local_list_31_2, local_list_32_2, local_list_31_Mean, local_list_31_Var, local_list_32_Mean, local_list_32_Var, \
        local_list_33_1, local_list_33_2, local_list_34_2, local_list_33_Mean, local_list_33_Var, local_list_34_Mean, local_list_34_Var, \
        local_list_35_1, local_list_35_2, local_list_36_2, local_list_35_Mean, local_list_35_Var, local_list_36_Mean, local_list_36_Var, \
        local_list_4\
        = global_aggregation_FedSGD(epoch, node_number, selected_local_device_index, device, args.half,
                                    global_average, global_average_weight,
                                    local_list_0_1, local_list_0_2,
                                    local_list_11_1, local_list_11_2, local_list_12_2,
                                    local_list_13_1, local_list_13_2, local_list_14_2,
                                    local_list_15_1, local_list_15_2, local_list_16_2,
                                    local_list_21_1, local_list_21_2, local_list_22_2,
                                    local_list_23_1, local_list_23_2, local_list_24_2,
                                    local_list_25_1, local_list_25_2, local_list_26_2,
                                    local_list_31_1, local_list_31_2, local_list_32_2,
                                    local_list_33_1, local_list_33_2, local_list_34_2,
                                    local_list_35_1, local_list_35_2, local_list_36_2,
                                    local_list_4)


    # ========================== more than 1 local updating steps ========================== #
    # average mean and variance
    if local_update_step_number_total > 1:

        # transform the representation of local models (transform subnet form to whole network form)
        local_list = subnet_2_whole(selected_local_device_index, epoch, node_number, device, args.half,
                                   local_list_0_1, local_list_0_2,
                                   local_list_11_1, local_list_11_2, local_list_12_2,
                                   local_list_13_1, local_list_13_2, local_list_14_2,
                                   local_list_15_1, local_list_15_2, local_list_16_2,
                                   local_list_21_1, local_list_21_2, local_list_22_2,
                                   local_list_23_1, local_list_23_2, local_list_24_2,
                                   local_list_25_1, local_list_25_2, local_list_26_2,
                                   local_list_31_1, local_list_31_2, local_list_32_2,
                                   local_list_33_1, local_list_33_2, local_list_34_2,
                                   local_list_35_1, local_list_35_2, local_list_36_2,
                                   local_list_4)

        # ---- block 0
        del local_list_0_1, local_list_0_2
        # ---- block 1
        del local_list_11_1, local_list_11_2, local_list_12_2
        del local_list_13_1, local_list_13_2, local_list_14_2
        del local_list_15_1, local_list_15_2, local_list_16_2
        # ---- block 2
        del local_list_21_1, local_list_21_2, local_list_22_2
        del local_list_23_1, local_list_23_2, local_list_24_2
        del local_list_25_1, local_list_25_2, local_list_26_2
        # ---- block 3
        del local_list_31_1, local_list_31_2, local_list_32_2
        del local_list_33_1, local_list_33_2, local_list_34_2
        del local_list_35_1, local_list_35_2, local_list_36_2
        # ---- Final
        del local_list_4

        # local updating
        local_weights_set = []
        for idx in selected_local_device_index:

            model_train = copy.deepcopy(local_list[idx])
            idxs = idxs_all[idx]

            train_dataset_each_client = DatasetSplit(train_dataset, idxs)
            optimizer = torch.optim.SGD(model_train.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
            train_loader = torch.utils.data.DataLoader(train_dataset_each_client,
                                       batch_size=args.batch_size, shuffle=False,
                                       sampler = torch.utils.data.sampler.RandomSampler(range(len(train_dataset_each_client)),
                                                                                        replacement=True,
                                                                                        num_samples=args.batch_size * (local_update_step_number_total - 1)),
                                       num_workers=args.workers, pin_memory=True)
            train(train_loader, model_train, criterion, optimizer, device, args.half)
            local_weights_set.append(copy.deepcopy(model_train.state_dict()))

        del local_list
        global_average_weight = average_weights(local_weights_set)
        global_average.load_state_dict(global_average_weight)

        # transform whole network form to subnet form
        local_list_0_1, local_list_0_2, local_list_0_Mean, local_list_0_Var, \
        local_list_11_1, local_list_11_2, local_list_12_2, local_list_11_Mean, local_list_11_Var, local_list_12_Mean, local_list_12_Var, \
        local_list_13_1, local_list_13_2, local_list_14_2, local_list_13_Mean, local_list_13_Var, local_list_14_Mean, local_list_14_Var, \
        local_list_15_1, local_list_15_2, local_list_16_2, local_list_15_Mean, local_list_15_Var, local_list_16_Mean, local_list_16_Var, \
        local_list_21_1, local_list_21_2, local_list_22_2, local_list_21_Mean, local_list_21_Var, local_list_22_Mean, local_list_22_Var, \
        local_list_23_1, local_list_23_2, local_list_24_2, local_list_23_Mean, local_list_23_Var, local_list_24_Mean, local_list_24_Var, \
        local_list_25_1, local_list_25_2, local_list_26_2, local_list_25_Mean, local_list_25_Var, local_list_26_Mean, local_list_26_Var, \
        local_list_31_1, local_list_31_2, local_list_32_2, local_list_31_Mean, local_list_31_Var, local_list_32_Mean, local_list_32_Var, \
        local_list_33_1, local_list_33_2, local_list_34_2, local_list_33_Mean, local_list_33_Var, local_list_34_Mean, local_list_34_Var, \
        local_list_35_1, local_list_35_2, local_list_36_2, local_list_35_Mean, local_list_35_Var, local_list_36_Mean, local_list_36_Var, \
        local_list_4 \
        = whole_2_subnet(node_number, global_average_weight, device, args.half)


    # evaluation
    if (epoch+1) % interval == 0 :                                                            
        loss_avg_DGD = Inference_loss(device,train_all_loader,criterion,global_average, args.half)
        train_loss_DGD.append(loss_avg_DGD) 
        accuracy_DGD = Inference_accuracy(device,test_all_loader,global_average, args.half)
        test_accuracy_DGD.append(accuracy_DGD)            
        print(f'\n | Global Training Round : {epoch+1}',"Objective DGD: ", loss_avg_DGD,
                "Test Accuracy DGD: {:.2f}%".format(accuracy_DGD))

        del loss_avg_DGD, accuracy_DGD

# save file
run_name = "{}_balanced_K5_E{}_ResNet20_32bits_FedTAN_AvgModel_10000_lr05_data_{}_mom_{}_wd_{}_seed_{}.xlsx".format(time.strftime("%Y_%m_%d", time.localtime()),
                                                                                                                    args.local_update_step_number,
                                                                                                                    args.data_distribution,
                                                                                                                    args.momentum, args.weight_decay, args.seed)
writer = pd.ExcelWriter(run_name)
# Training loss
df1_21 = pd.DataFrame(data = train_loss_DGD)
df1_21.to_excel(writer, 'train_loss_model_K10')
# testing accuracy
df2_21 = pd.DataFrame(data = test_accuracy_DGD)
df2_21.to_excel(writer, 'test_accuracy_model_K10')
writer.save()