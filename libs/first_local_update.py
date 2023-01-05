import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
from utils.training import train_forward_1, train_forward_var, train_forward_2, train_forward_3_woRes, train_forward_3_Res, train_forward_fc # training on subnet (forward propagation)
from utils.training import train_backward_3_ResOut, train_backward_3_ResInOut, train_backward_3_ResIn, train_backward_2, train_backward_1_Input, train_backward_1_woInput # training on subnet (backward propagation)
from utils.datasets import DatasetSplit

def first_local_updating(local_list_0_1, local_list_0_2, local_list_0_Mean, local_list_0_Var,
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
                         selected_node_number, selected_local_device_index, idxs_all, train_dataset, criterion, args, device):

    net_input_list_set, target_list_set, input_list_0_set = [], [], []
    iidx = 0
    for idx in selected_local_device_index:

        idxs = idxs_all[iidx]

        train_dataset_each_client = DatasetSplit(train_dataset, idxs)
        train_loader = torch.utils.data.DataLoader(train_dataset_each_client,
                                                   batch_size=args.batch_size, shuffle=False,
                                                   sampler=torch.utils.data.sampler.RandomSampler(
                                                       range(len(train_dataset_each_client)),
                                                       replacement=True,
                                                       num_samples=args.batch_size),
                                                   num_workers=args.workers, pin_memory=True)

        net_input_list, target_list = [], []
        for i, (input_data, target) in enumerate(train_loader):
            if args.half:
                net_input_list.append(input_data.to(device).half())
            else:
                net_input_list.append(input_data.to(device))
            target_list.append(target.to(device))

        net_input_list_set.append(net_input_list)
        target_list_set.append(target_list)

        # block 0
        # ---- train_forward_0_1
        (input_list_0, mean_list_0,
         batchsize, channels_0, height_0, width_0) = train_forward_1(local_list_0_1[idx], local_list_0_Mean[idx], net_input_list)

        input_list_0_set.append(input_list_0)
        # ==== average mean
        if iidx == 0:
            mean_list_0_set_average = [torch.zeros_like(mean_list_0[0])]
        mean_list_0_set_average[0] += mean_list_0[0] / selected_node_number
        iidx += 1
    del net_input_list, target_list, input_list_0, mean_list_0

    # ---- train_forward_0_var
    iidx = 0
    for idx in selected_local_device_index:
        input_list_0 = input_list_0_set[iidx]
        mean_list_0 = mean_list_0_set_average  # mean_list_0_set[iidx]

        bias_var_list_0 = train_forward_var(local_list_0_Var[idx], input_list_0, mean_list_0)
        # ==== average variance
        if iidx == 0:
            bias_var_list_0_set_average = [torch.zeros_like(bias_var_list_0[0])]
        bias_var_list_0_set_average[0] += bias_var_list_0[0] / selected_node_number
        iidx += 1
    del input_list_0, mean_list_0, bias_var_list_0

    out0_list_set, input_list_11_set = [], []
    iidx = 0
    for idx in selected_local_device_index:
        input_list_0 = input_list_0_set[iidx]
        mean_list_0 = mean_list_0_set_average  # mean_list_0_set[iidx]
        bias_var_list_0 = bias_var_list_0_set_average  # bias_var_list_0_set[iidx]

        # ---- train_forward_0_2
        out0_list = train_forward_3_woRes(local_list_0_2[idx], input_list_0, mean_list_0, bias_var_list_0,
                                          batchsize, channels_0, height_0, width_0)
        out0_list_set.append(out0_list)

        # block 1
        # ---- train_forward_11_1
        (input_list_11, mean_list_11,
         batchsize, channels_11, height_11, width_11) = train_forward_1(local_list_11_1[idx], local_list_11_Mean[idx], out0_list)

        input_list_11_set.append(input_list_11)
        # ==== average mean
        if iidx == 0:
            mean_list_11_set_average = [torch.zeros_like(mean_list_11[0])]
        mean_list_11_set_average[0] += mean_list_11[0] / selected_node_number
        iidx += 1
    del input_list_0, mean_list_0, bias_var_list_0, out0_list, input_list_11, mean_list_11

    # ---- train_forward_11_var
    iidx = 0
    for idx in selected_local_device_index:
        input_list_11 = input_list_11_set[iidx]
        mean_list_11 = mean_list_11_set_average  # mean_list_11_set[iidx]

        bias_var_list_11 = train_forward_var(local_list_11_Var[idx], input_list_11, mean_list_11)
        # ==== average variance
        if iidx == 0:
            bias_var_list_11_set_average = [torch.zeros_like(bias_var_list_11[0])]
        bias_var_list_11_set_average[0] += bias_var_list_11[0] / selected_node_number
        iidx += 1
    del input_list_11, mean_list_11, bias_var_list_11

    # ---- train_forward_11_2
    input_list_12_set = []
    iidx = 0
    for idx in selected_local_device_index:
        input_list_11 = input_list_11_set[iidx]
        mean_list_11 = mean_list_11_set_average  # mean_list_11_set[iidx]
        bias_var_list_11 = bias_var_list_11_set_average  # bias_var_list_11_set[iidx]

        (input_list_12, mean_list_12,
         batchsize, channels_12, height_12, width_12) = train_forward_2(local_list_11_2[idx], local_list_12_Mean[idx],
                                                                        input_list_11, mean_list_11, bias_var_list_11,
                                                                        batchsize, channels_11, height_11, width_11)
        input_list_12_set.append(input_list_12)
        # ==== average mean
        if iidx == 0:
            mean_list_12_set_average = [torch.zeros_like(mean_list_12[0])]
        mean_list_12_set_average[0] += mean_list_12[0] / selected_node_number
        iidx += 1
    del input_list_11, mean_list_11, bias_var_list_11, input_list_12, mean_list_12

    # ---- train_forward_12_var
    iidx = 0
    for idx in selected_local_device_index:
        input_list_12 = input_list_12_set[iidx]
        mean_list_12 = mean_list_12_set_average  # mean_list_12_set[iidx]

        bias_var_list_12 = train_forward_var(local_list_12_Var[idx], input_list_12, mean_list_12)
        # ==== average variance
        if iidx == 0:
            bias_var_list_12_set_average = [torch.zeros_like(bias_var_list_12[0])]
        bias_var_list_12_set_average[0] += bias_var_list_12[0] / selected_node_number
        iidx += 1
    del input_list_12, mean_list_12, bias_var_list_12

    out12_list_set, input_list_13_set = [], []
    iidx = 0
    for idx in selected_local_device_index:
        out0_list = out0_list_set[iidx]
        input_list_12 = input_list_12_set[iidx]
        mean_list_12 = mean_list_12_set_average  # mean_list_12_set[iidx]
        bias_var_list_12 = bias_var_list_12_set_average  # bias_var_list_12_set[iidx]

        # ---- train_forward_12_2
        out12_list = train_forward_3_Res(local_list_12_2[idx], out0_list,
                                         input_list_12, mean_list_12, bias_var_list_12,
                                         batchsize, channels_12, height_12, width_12)
        out12_list_set.append(out12_list)

        # ---- train_forward_13_1
        (input_list_13, mean_list_13,
         batchsize, channels_13, height_13, width_13) = train_forward_1(local_list_13_1[idx], local_list_13_Mean[idx], out12_list)
        input_list_13_set.append(input_list_13)
        # ==== average mean
        if iidx == 0:
            mean_list_13_set_average = [torch.zeros_like(mean_list_13[0])]
        mean_list_13_set_average[0] += mean_list_13[0] / selected_node_number
        iidx += 1
    del out0_list, input_list_12, mean_list_12, bias_var_list_12, out12_list, input_list_13, mean_list_13

    # ---- train_forward_13_var
    iidx = 0
    for idx in selected_local_device_index:
        input_list_13 = input_list_13_set[iidx]
        mean_list_13 = mean_list_13_set_average  # mean_list_13_set[iidx]

        bias_var_list_13 = train_forward_var(local_list_13_Var[idx], input_list_13, mean_list_13)
        # ==== average variance
        if iidx == 0:
            bias_var_list_13_set_average = [torch.zeros_like(bias_var_list_13[0])]
        bias_var_list_13_set_average[0] += bias_var_list_13[0] / selected_node_number
        iidx += 1
    del input_list_13, mean_list_13, bias_var_list_13

    # ---- train_forward_13_2
    input_list_14_set = []
    iidx = 0
    for idx in selected_local_device_index:
        input_list_13 = input_list_13_set[iidx]
        mean_list_13 = mean_list_13_set_average  # mean_list_13_set[iidx]
        bias_var_list_13 = bias_var_list_13_set_average  # bias_var_list_13_set[iidx]

        (input_list_14, mean_list_14,
         batchsize, channels_14, height_14, width_14) = train_forward_2(local_list_13_2[idx], local_list_14_Mean[idx],
                                                                        input_list_13, mean_list_13, bias_var_list_13,
                                                                        batchsize, channels_13, height_13, width_13)
        input_list_14_set.append(input_list_14)
        # ==== average mean
        if iidx == 0:
            mean_list_14_set_average = [torch.zeros_like(mean_list_14[0])]
        mean_list_14_set_average[0] += mean_list_14[0] / selected_node_number
        iidx += 1
    del input_list_13, mean_list_13, bias_var_list_13, input_list_14, mean_list_14

    # ---- train_forward_14_var
    iidx = 0
    for idx in selected_local_device_index:
        input_list_14 = input_list_14_set[iidx]
        mean_list_14 = mean_list_14_set_average  # mean_list_14_set[iidx]

        bias_var_list_14 = train_forward_var(local_list_14_Var[idx], input_list_14, mean_list_14)
        # ==== average variance
        if iidx == 0:
            bias_var_list_14_set_average = [torch.zeros_like(bias_var_list_14[0])]
        bias_var_list_14_set_average[0] += bias_var_list_14[0] / selected_node_number
        iidx += 1
    del input_list_14, mean_list_14, bias_var_list_14

    out14_list_set, input_list_15_set = [], []
    iidx = 0
    for idx in selected_local_device_index:
        out12_list = out12_list_set[iidx]
        input_list_14 = input_list_14_set[iidx]
        mean_list_14 = mean_list_14_set_average  # mean_list_14_set[iidx]
        bias_var_list_14 = bias_var_list_14_set_average  # bias_var_list_14_set[iidx]

        # ---- train_forward_14_2
        out14_list = train_forward_3_Res(local_list_14_2[idx], out12_list,
                                         input_list_14, mean_list_14, bias_var_list_14,
                                         batchsize, channels_14, height_14, width_14)
        out14_list_set.append(out14_list)

        # ---- train_forward_15_1
        (input_list_15, mean_list_15,
         batchsize, channels_15, height_15, width_15) = train_forward_1(local_list_15_1[idx], local_list_15_Mean[idx], out14_list)
        input_list_15_set.append(input_list_15)
        # ==== average mean
        if iidx == 0:
            mean_list_15_set_average = [torch.zeros_like(mean_list_15[0])]
        mean_list_15_set_average[0] += mean_list_15[0] / selected_node_number
        iidx += 1
    del out12_list, input_list_14, mean_list_14, bias_var_list_14, out14_list, input_list_15, mean_list_15

    # ---- train_forward_15_var
    iidx = 0
    for idx in selected_local_device_index:
        input_list_15 = input_list_15_set[iidx]
        mean_list_15 = mean_list_15_set_average  # mean_list_11_set[iidx]

        bias_var_list_15 = train_forward_var(local_list_15_Var[idx], input_list_15, mean_list_15)
        # ==== average variance
        if iidx == 0:
            bias_var_list_15_set_average = [torch.zeros_like(bias_var_list_15[0])]
        bias_var_list_15_set_average[0] += bias_var_list_15[0] / selected_node_number
        iidx += 1
    del input_list_15, mean_list_15, bias_var_list_15

    input_list_16_set = []
    iidx = 0
    for idx in selected_local_device_index:
        input_list_15 = input_list_15_set[iidx]
        mean_list_15 = mean_list_15_set_average  # mean_list_15_set[iidx]
        bias_var_list_15 = bias_var_list_15_set_average  # bias_var_list_15_set[iidx]

        # ---- train_forward_15_2
        (input_list_16, mean_list_16,
         batchsize, channels_16, height_16, width_16) = train_forward_2(local_list_15_2[idx], local_list_16_Mean[idx],
                                                                        input_list_15, mean_list_15, bias_var_list_15,
                                                                        batchsize, channels_15, height_15, width_15)
        input_list_16_set.append(input_list_16)
        # ==== average mean
        if iidx == 0:
            mean_list_16_set_average = [torch.zeros_like(mean_list_16[0])]
        mean_list_16_set_average[0] += mean_list_16[0] / selected_node_number
        iidx += 1
    del input_list_15, mean_list_15, bias_var_list_15, input_list_16, mean_list_16

    # ---- train_forward_16_var
    iidx = 0
    for idx in selected_local_device_index:
        input_list_16 = input_list_16_set[iidx]
        mean_list_16 = mean_list_16_set_average  # mean_list_16_set[iidx]

        bias_var_list_16 = train_forward_var(local_list_16_Var[idx], input_list_16, mean_list_16)
        # ==== average variance
        if iidx == 0:
            bias_var_list_16_set_average = [torch.zeros_like(bias_var_list_16[0])]
        bias_var_list_16_set_average[0] += bias_var_list_16[0] / selected_node_number
        iidx += 1
    del input_list_16, mean_list_16, bias_var_list_16

    out16_list_set, input_list_21_set = [], []
    iidx = 0
    for idx in selected_local_device_index:
        out14_list = out14_list_set[iidx]
        input_list_16 = input_list_16_set[iidx]
        mean_list_16 = mean_list_16_set_average  # mean_list_16_set[iidx]
        bias_var_list_16 = bias_var_list_16_set_average  # bias_var_list_16_set[iidx]

        # ---- train_forward_16_2
        out16_list = train_forward_3_Res(local_list_16_2[idx], out14_list,
                                         input_list_16, mean_list_16, bias_var_list_16,
                                         batchsize, channels_16, height_16, width_16)
        out16_list_set.append(out16_list)

        # block 2
        # ---- train_forward_21_1
        (input_list_21, mean_list_21,
         batchsize, channels_21, height_21, width_21) = train_forward_1(local_list_21_1[idx], local_list_21_Mean[idx], out16_list)
        input_list_21_set.append(input_list_21)
        # ==== average mean
        if iidx == 0:
            mean_list_21_set_average = [torch.zeros_like(mean_list_21[0])]
        mean_list_21_set_average[0] += mean_list_21[0] / selected_node_number
        iidx += 1
    del out14_list, input_list_16, mean_list_16, bias_var_list_16, out16_list, input_list_21, mean_list_21

    # ---- train_forward_21_var
    iidx = 0
    for idx in selected_local_device_index:
        input_list_21 = input_list_21_set[iidx]
        mean_list_21 = mean_list_21_set_average  # mean_list_21_set[iidx]

        bias_var_list_21 = train_forward_var(local_list_21_Var[idx], input_list_21, mean_list_21)
        # ==== average variance
        if iidx == 0:
            bias_var_list_21_set_average = [torch.zeros_like(bias_var_list_21[0])]
        bias_var_list_21_set_average[0] += bias_var_list_21[0] / selected_node_number
        iidx += 1
    del input_list_21, mean_list_21, bias_var_list_21

    # ---- train_forward_21_2
    input_list_22_set = []
    iidx = 0
    for idx in selected_local_device_index:
        input_list_21 = input_list_21_set[iidx]
        mean_list_21 = mean_list_21_set_average  # mean_list_21_set[iidx]
        bias_var_list_21 = bias_var_list_21_set_average  # bias_var_list_21_set[iidx]

        (input_list_22, mean_list_22,
         batchsize, channels_22, height_22, width_22) = train_forward_2(local_list_21_2[idx], local_list_22_Mean[idx],
                                                                        input_list_21, mean_list_21, bias_var_list_21,
                                                                        batchsize, channels_21, height_21, width_21)
        input_list_22_set.append(input_list_22)
        # ==== average mean
        if iidx == 0:
            mean_list_22_set_average = [torch.zeros_like(mean_list_22[0])]
        mean_list_22_set_average[0] += mean_list_22[0] / selected_node_number
        iidx += 1
    del input_list_21, mean_list_21, bias_var_list_21, input_list_22, mean_list_22

    # ---- train_forward_22_var
    iidx = 0
    for idx in selected_local_device_index:
        input_list_22 = input_list_22_set[iidx]
        mean_list_22 = mean_list_22_set_average  # mean_list_22_set[iidx]

        bias_var_list_22 = train_forward_var(local_list_22_Var[idx], input_list_22, mean_list_22)
        # ==== average variance
        if iidx == 0:
            bias_var_list_22_set_average = [torch.zeros_like(bias_var_list_22[0])]
        bias_var_list_22_set_average[0] += bias_var_list_22[0] / selected_node_number
        iidx += 1
    del input_list_22, mean_list_22, bias_var_list_22

    out22_list_set, input_list_23_set = [], []
    iidx = 0
    for idx in selected_local_device_index:
        out16_list = out16_list_set[iidx]
        input_list_22 = input_list_22_set[iidx]
        mean_list_22 = mean_list_22_set_average  # mean_list_22_set[iidx]
        bias_var_list_22 = bias_var_list_22_set_average  # bias_var_list_22_set[iidx]

        # ---- train_forward_22_2
        out22_list = train_forward_3_Res(local_list_22_2[idx], out16_list,
                                         input_list_22, mean_list_22, bias_var_list_22,
                                         batchsize, channels_22, height_22, width_22)
        out22_list_set.append(out22_list)

        # ---- train_forward_23_1
        (input_list_23, mean_list_23,
         batchsize, channels_23, height_23, width_23) = train_forward_1(local_list_23_1[idx], local_list_23_Mean[idx], out22_list)
        input_list_23_set.append(input_list_23)
        # ==== average mean
        if iidx == 0:
            mean_list_23_set_average = [torch.zeros_like(mean_list_23[0])]
        mean_list_23_set_average[0] += mean_list_23[0] / selected_node_number
        iidx += 1
    del out16_list, input_list_22, mean_list_22, bias_var_list_22, out22_list, input_list_23, mean_list_23

    # ---- train_forward_23_var
    iidx = 0
    for idx in selected_local_device_index:
        input_list_23 = input_list_23_set[iidx]
        mean_list_23 = mean_list_23_set_average  # mean_list_23_set[iidx]

        bias_var_list_23 = train_forward_var(local_list_23_Var[idx], input_list_23, mean_list_23)
        # ==== average variance
        if iidx == 0:
            bias_var_list_23_set_average = [torch.zeros_like(bias_var_list_23[0])]
        bias_var_list_23_set_average[0] += bias_var_list_23[0] / selected_node_number
        iidx += 1
    del input_list_23, mean_list_23, bias_var_list_23

    # ---- train_forward_23_2
    input_list_24_set = []
    iidx = 0
    for idx in selected_local_device_index:
        input_list_23 = input_list_23_set[iidx]
        mean_list_23 = mean_list_23_set_average  # mean_list_21_set[iidx]
        bias_var_list_23 = bias_var_list_23_set_average  # bias_var_list_21_set[iidx]

        (input_list_24, mean_list_24,
         batchsize, channels_24, height_24, width_24) = train_forward_2(local_list_23_2[idx], local_list_24_Mean[idx],
                                                                        input_list_23, mean_list_23, bias_var_list_23,
                                                                        batchsize, channels_23, height_23, width_23)
        input_list_24_set.append(input_list_24)
        # ==== average mean
        if iidx == 0:
            mean_list_24_set_average = [torch.zeros_like(mean_list_24[0])]
        mean_list_24_set_average[0] += mean_list_24[0] / selected_node_number
        iidx += 1
    del input_list_23, mean_list_23, bias_var_list_23, input_list_24, mean_list_24

    # ---- train_forward_24_var
    iidx = 0
    for idx in selected_local_device_index:
        input_list_24 = input_list_24_set[iidx]
        mean_list_24 = mean_list_24_set_average  # mean_list_24_set[iidx]

        bias_var_list_24 = train_forward_var(local_list_24_Var[idx], input_list_24, mean_list_24)
        # ==== average variance
        if iidx == 0:
            bias_var_list_24_set_average = [torch.zeros_like(bias_var_list_24[0])]
        bias_var_list_24_set_average[0] += bias_var_list_24[0] / selected_node_number
        iidx += 1
    del input_list_24, mean_list_24, bias_var_list_24

    out24_list_set, input_list_25_set = [], []
    iidx = 0
    for idx in selected_local_device_index:
        out22_list = out22_list_set[iidx]
        input_list_24 = input_list_24_set[iidx]
        mean_list_24 = mean_list_24_set_average  # mean_list_24_set[iidx]
        bias_var_list_24 = bias_var_list_24_set_average  # bias_var_list_24_set[iidx]

        # ---- train_forward_24_2
        out24_list = train_forward_3_Res(local_list_24_2[idx], out22_list,
                                         input_list_24, mean_list_24, bias_var_list_24,
                                         batchsize, channels_24, height_24, width_24)
        out24_list_set.append(out24_list)

        # ---- train_forward_25_1
        (input_list_25, mean_list_25,
         batchsize, channels_25, height_25, width_25) = train_forward_1(local_list_25_1[idx], local_list_25_Mean[idx], out24_list)
        input_list_25_set.append(input_list_25)
        # ==== average mean
        if iidx == 0:
            mean_list_25_set_average = [torch.zeros_like(mean_list_25[0])]
        mean_list_25_set_average[0] += mean_list_25[0] / selected_node_number
        iidx += 1
    del out22_list, input_list_24, mean_list_24, bias_var_list_24, out24_list, input_list_25, mean_list_25

    # ---- train_forward_25_var
    iidx = 0
    for idx in selected_local_device_index:
        input_list_25 = input_list_25_set[iidx]
        mean_list_25 = mean_list_25_set_average  # mean_list_25_set[iidx]

        bias_var_list_25 = train_forward_var(local_list_25_Var[idx], input_list_25, mean_list_25)
        # ==== average variance
        if iidx == 0:
            bias_var_list_25_set_average = [torch.zeros_like(bias_var_list_25[0])]
        bias_var_list_25_set_average[0] += bias_var_list_25[0] / selected_node_number
        iidx += 1
    del input_list_25, mean_list_25, bias_var_list_25

    # ---- train_forward_25_2
    input_list_26_set = []
    iidx = 0
    for idx in selected_local_device_index:
        input_list_25 = input_list_25_set[iidx]
        mean_list_25 = mean_list_25_set_average  # mean_list_25_set[iidx]
        bias_var_list_25 = bias_var_list_25_set_average  # bias_var_list_25_set[iidx]

        (input_list_26, mean_list_26,
         batchsize, channels_26, height_26, width_26) = train_forward_2(local_list_25_2[idx], local_list_26_Mean[idx],
                                                                        input_list_25, mean_list_25, bias_var_list_25,
                                                                        batchsize, channels_25, height_25, width_25)
        input_list_26_set.append(input_list_26)
        # ==== average mean
        if iidx == 0:
            mean_list_26_set_average = [torch.zeros_like(mean_list_26[0])]
        mean_list_26_set_average[0] += mean_list_26[0] / selected_node_number
        iidx += 1
    del input_list_25, mean_list_25, bias_var_list_25, input_list_26, mean_list_26

    # ---- train_forward_26_var
    iidx = 0
    for idx in selected_local_device_index:
        input_list_26 = input_list_26_set[iidx]
        mean_list_26 = mean_list_26_set_average  # mean_list_26_set[iidx]

        bias_var_list_26 = train_forward_var(local_list_26_Var[idx], input_list_26, mean_list_26)
        # ==== average variance
        if iidx == 0:
            bias_var_list_26_set_average = [torch.zeros_like(bias_var_list_26[0])]
        bias_var_list_26_set_average[0] += bias_var_list_26[0] / selected_node_number
        iidx += 1
    del input_list_26, mean_list_26, bias_var_list_26

    out26_list_set, input_list_31_set = [], []
    iidx = 0
    for idx in selected_local_device_index:
        out24_list = out24_list_set[iidx]
        input_list_26 = input_list_26_set[iidx]
        mean_list_26 = mean_list_26_set_average  # mean_list_26_set[iidx]
        bias_var_list_26 = bias_var_list_26_set_average  # bias_var_list_26_set[iidx]

        # ---- train_forward_26_2
        out26_list = train_forward_3_Res(local_list_26_2[idx], out24_list,
                                         input_list_26, mean_list_26, bias_var_list_26,
                                         batchsize, channels_26, height_26, width_26)
        out26_list_set.append(out26_list)

        # block 3
        # ---- train_forward_31_1
        (input_list_31, mean_list_31,
         batchsize, channels_31, height_31, width_31) = train_forward_1(local_list_31_1[idx], local_list_31_Mean[idx], out26_list)
        input_list_31_set.append(input_list_31)
        # ==== average mean
        if iidx == 0:
            mean_list_31_set_average = [torch.zeros_like(mean_list_31[0])]
        mean_list_31_set_average[0] += mean_list_31[0] / selected_node_number
        iidx += 1
    del out24_list, input_list_26, mean_list_26, bias_var_list_26, out26_list, input_list_31, mean_list_31

    # ---- train_forward_31_var
    iidx = 0
    for idx in selected_local_device_index:
        input_list_31 = input_list_31_set[iidx]
        mean_list_31 = mean_list_31_set_average  # mean_list_31_set[iidx]

        bias_var_list_31 = train_forward_var(local_list_31_Var[idx], input_list_31, mean_list_31)
        # ==== average variance
        if iidx == 0:
            bias_var_list_31_set_average = [torch.zeros_like(bias_var_list_31[0])]
        bias_var_list_31_set_average[0] += bias_var_list_31[0] / selected_node_number
        iidx += 1
    del input_list_31, mean_list_31, bias_var_list_31

    # ---- train_forward_31_2
    input_list_32_set = []
    iidx = 0
    for idx in selected_local_device_index:
        input_list_31 = input_list_31_set[iidx]
        mean_list_31 = mean_list_31_set_average  # mean_list_31_set[iidx]
        bias_var_list_31 = bias_var_list_31_set_average  # bias_var_list_31_set[iidx]

        (input_list_32, mean_list_32,
         batchsize, channels_32, height_32, width_32) = train_forward_2(local_list_31_2[idx], local_list_32_Mean[idx],
                                                                        input_list_31, mean_list_31, bias_var_list_31,
                                                                        batchsize, channels_31, height_31, width_31)
        input_list_32_set.append(input_list_32)
        # ==== average mean
        if iidx == 0:
            mean_list_32_set_average = [torch.zeros_like(mean_list_32[0])]
        mean_list_32_set_average[0] += mean_list_32[0] / selected_node_number
        iidx += 1
    del input_list_31, mean_list_31, bias_var_list_31, input_list_32, mean_list_32

    # ---- train_forward_32_var
    iidx = 0
    for idx in selected_local_device_index:
        input_list_32 = input_list_32_set[iidx]
        mean_list_32 = mean_list_32_set_average  # mean_list_32_set[iidx]

        bias_var_list_32 = train_forward_var(local_list_32_Var[idx], input_list_32, mean_list_32)
        # ==== average variance
        if iidx == 0:
            bias_var_list_32_set_average = [torch.zeros_like(bias_var_list_32[0])]
        bias_var_list_32_set_average[0] += bias_var_list_32[0] / selected_node_number
        iidx += 1
    del input_list_32, mean_list_32, bias_var_list_32

    out32_list_set, input_list_33_set = [], []
    iidx = 0
    for idx in selected_local_device_index:
        out26_list = out26_list_set[iidx]
        input_list_32 = input_list_32_set[iidx]
        mean_list_32 = mean_list_32_set_average  # mean_list_32_set[iidx]
        bias_var_list_32 = bias_var_list_32_set_average  # bias_var_list_32_set[iidx]

        # ---- train_forward_32_2
        out32_list = train_forward_3_Res(local_list_32_2[idx], out26_list,
                                         input_list_32, mean_list_32, bias_var_list_32,
                                         batchsize, channels_32, height_32, width_32)
        out32_list_set.append(out32_list)

        # ---- train_forward_33_1
        (input_list_33, mean_list_33,
         batchsize, channels_33, height_33, width_33) = train_forward_1(local_list_33_1[idx], local_list_33_Mean[idx], out32_list)
        input_list_33_set.append(input_list_33)
        # ==== average mean
        if iidx == 0:
            mean_list_33_set_average = [torch.zeros_like(mean_list_33[0])]
        mean_list_33_set_average[0] += mean_list_33[0] / selected_node_number
        iidx += 1
    del out26_list, input_list_32, mean_list_32, bias_var_list_32, out32_list, input_list_33, mean_list_33

    # ---- train_forward_33_var
    iidx = 0
    for idx in selected_local_device_index:
        input_list_33 = input_list_33_set[iidx]
        mean_list_33 = mean_list_33_set_average  # mean_list_33_set[iidx]

        bias_var_list_33 = train_forward_var(local_list_33_Var[idx], input_list_33, mean_list_33)
        # ==== average variance
        if iidx == 0:
            bias_var_list_33_set_average = [torch.zeros_like(bias_var_list_33[0])]
        bias_var_list_33_set_average[0] += bias_var_list_33[0] / selected_node_number
        iidx += 1
    del input_list_33, mean_list_33, bias_var_list_33

    # ---- train_forward_33_2
    input_list_34_set = []
    iidx = 0
    for idx in selected_local_device_index:
        input_list_33 = input_list_33_set[iidx]
        mean_list_33 = mean_list_33_set_average  # mean_list_33_set[iidx]
        bias_var_list_33 = bias_var_list_33_set_average  # bias_var_list_33_set[iidx]

        (input_list_34, mean_list_34,
         batchsize, channels_34, height_34, width_34) = train_forward_2(local_list_33_2[idx], local_list_34_Mean[idx],
                                                                        input_list_33, mean_list_33, bias_var_list_33,
                                                                        batchsize, channels_33, height_33, width_33)
        input_list_34_set.append(input_list_34)
        # ==== average mean
        if iidx == 0:
            mean_list_34_set_average = [torch.zeros_like(mean_list_34[0])]
        mean_list_34_set_average[0] += mean_list_34[0] / selected_node_number
        iidx += 1
    del input_list_33, mean_list_33, bias_var_list_33, input_list_34, mean_list_34

    # ---- train_forward_34_var
    iidx = 0
    for idx in selected_local_device_index:
        input_list_34 = input_list_34_set[iidx]
        mean_list_34 = mean_list_34_set_average  # mean_list_34_set[iidx]

        bias_var_list_34 = train_forward_var(local_list_34_Var[idx], input_list_34, mean_list_34)
        # ==== average variance
        if iidx == 0:
            bias_var_list_34_set_average = [torch.zeros_like(bias_var_list_34[0])]
        bias_var_list_34_set_average[0] += bias_var_list_34[0] / selected_node_number
        iidx += 1
    del input_list_34, mean_list_34, bias_var_list_34

    out34_list_set, input_list_35_set = [], []
    iidx = 0
    for idx in selected_local_device_index:
        out32_list = out32_list_set[iidx]
        input_list_34 = input_list_34_set[iidx]
        mean_list_34 = mean_list_34_set_average  # mean_list_34_set[iidx]
        bias_var_list_34 = bias_var_list_34_set_average  # bias_var_list_34_set[iidx]

        # ---- train_forward_34_2
        out34_list = train_forward_3_Res(local_list_34_2[idx], out32_list,
                                         input_list_34, mean_list_34, bias_var_list_34,
                                         batchsize, channels_34, height_34, width_34)
        out34_list_set.append(out34_list)

        # ---- train_forward_35_1
        (input_list_35, mean_list_35,
         batchsize, channels_35, height_35, width_35) = train_forward_1(local_list_35_1[idx], local_list_35_Mean[idx], out34_list)
        input_list_35_set.append(input_list_35)
        # ==== average mean
        if iidx == 0:
            mean_list_35_set_average = [torch.zeros_like(mean_list_35[0])]
        mean_list_35_set_average[0] += mean_list_35[0] / selected_node_number
        iidx += 1
    del out32_list, input_list_34, mean_list_34, bias_var_list_34, out34_list, input_list_35, mean_list_35

    # ---- train_forward_35_var
    iidx = 0
    for idx in selected_local_device_index:
        input_list_35 = input_list_35_set[iidx]
        mean_list_35 = mean_list_35_set_average  # mean_list_35_set[iidx]

        bias_var_list_35 = train_forward_var(local_list_35_Var[idx], input_list_35, mean_list_35)
        # ==== average variance
        if iidx == 0:
            bias_var_list_35_set_average = [torch.zeros_like(bias_var_list_35[0])]
        bias_var_list_35_set_average[0] += bias_var_list_35[0] / selected_node_number
        iidx += 1
    del input_list_35, mean_list_35, bias_var_list_35

    # ---- train_forward_35_2
    input_list_36_set = []
    iidx = 0
    for idx in selected_local_device_index:
        input_list_35 = input_list_35_set[iidx]
        mean_list_35 = mean_list_35_set_average  # mean_list_35_set[iidx]
        bias_var_list_35 = bias_var_list_35_set_average  # bias_var_list_35_set[iidx]

        (input_list_36, mean_list_36,
         batchsize, channels_36, height_36, width_36) = train_forward_2(local_list_35_2[idx], local_list_36_Mean[idx],
                                                                        input_list_35, mean_list_35, bias_var_list_35,
                                                                        batchsize, channels_35, height_35, width_35)
        input_list_36_set.append(input_list_36)
        # ==== average mean
        if iidx == 0:
            mean_list_36_set_average = [torch.zeros_like(mean_list_36[0])]
        mean_list_36_set_average[0] += mean_list_36[0] / selected_node_number
        iidx += 1
    del input_list_35, mean_list_35, bias_var_list_35, input_list_36, mean_list_36

    # ---- train_forward_36_var
    iidx = 0
    for idx in selected_local_device_index:
        input_list_36 = input_list_36_set[iidx]
        mean_list_36 = mean_list_36_set_average  # mean_list_36_set[iidx]

        bias_var_list_36 = train_forward_var(local_list_36_Var[idx], input_list_36, mean_list_36)
        # ==== average variance
        if iidx == 0:
            bias_var_list_36_set_average = [torch.zeros_like(bias_var_list_36[0])]
        bias_var_list_36_set_average[0] += bias_var_list_36[0] / selected_node_number
        iidx += 1

    del input_list_36, mean_list_36, bias_var_list_36

    # ---- train_forward_36_2
    out36_list_set = []
    iidx = 0
    for idx in selected_local_device_index:
        out34_list = out34_list_set[iidx]
        input_list_36 = input_list_36_set[iidx]
        mean_list_36 = mean_list_36_set_average  # mean_list_36_set[iidx]
        bias_var_list_36 = bias_var_list_36_set_average  # bias_var_list_36_set[iidx]
        iidx += 1

        out36_list = train_forward_3_Res(local_list_36_2[idx], out34_list,
                                         input_list_36, mean_list_36, bias_var_list_36,
                                         batchsize, channels_36, height_36, width_36)
        out36_list_set.append(out36_list)
    del out34_list, input_list_36, mean_list_36, bias_var_list_36, out36_list

    (grad_model_36_2_shortcut_input_0_list_set,
     grad_bn36_11_input_0_list_set) = [], []
    iidx = 0
    for idx in selected_local_device_index:
        out36_list = out36_list_set[iidx]
        target_list = target_list_set[iidx]

        out34_list = out34_list_set[iidx]
        input_list_36 = input_list_36_set[iidx]
        mean_list_36 = mean_list_36_set_average  # mean_list_36_set[iidx]
        bias_var_list_36 = bias_var_list_36_set_average  # bias_var_list_36_set[iidx]

        # final
        # ---- train_forward_4
        grad_model_fc_origin_input_0 = torch.rand([batchsize, channels_36, height_36, width_36])

        optimizer_4 = torch.optim.SGD([{'params': local_list_4[idx].parameters()}],
                                      args.lr,
                                      momentum=args.momentum,
                                      weight_decay=args.weight_decay)

        grad_model_4_origin_input_0_list = train_forward_fc(local_list_4[idx], out36_list, target_list, optimizer_4, criterion)

        # block 3
        # ---- train_backward_36_2
        grad_model_3_shortcut_input_0 = torch.rand([batchsize, channels_34, height_34, width_34])
        grad_bn_11_input_0 = torch.rand([channels_36, batchsize * height_36 * width_36])
        grad_bn_11_input_1 = torch.rand(channels_36)
        grad_bn_12_input_0 = torch.rand(channels_36)

        optimizer_36_2 = torch.optim.SGD([{'params': local_list_36_2[idx].parameters()}], args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

        (grad_model_36_2_shortcut_input_0_list,
         grad_bn36_11_input_0_list,
         grad_bn36_11_input_1_list,
         grad_bn36_12_input_0_list) = train_backward_3_ResIn(local_list_36_2[idx], optimizer_36_2, out34_list,
                                                             input_list_36, mean_list_36, bias_var_list_36,
                                                             batchsize, channels_36, height_36, width_36,
                                                             grad_model_4_origin_input_0_list, device)

        grad_model_36_2_shortcut_input_0_list_set.append(grad_model_36_2_shortcut_input_0_list)
        grad_bn36_11_input_0_list_set.append(grad_bn36_11_input_0_list)
        # ==== average mean gradient
        if iidx == 0:
            grad_bn36_11_input_1_list_set_average = [torch.zeros_like(grad_bn36_11_input_1_list[0])]
        grad_bn36_11_input_1_list_set_average[0] += grad_bn36_11_input_1_list[0] / selected_node_number
        # ==== average variance gradient
        if iidx == 0:
            grad_bn36_12_input_0_list_set_average = [torch.zeros_like(grad_bn36_12_input_0_list[0])]
        grad_bn36_12_input_0_list_set_average[0] += grad_bn36_12_input_0_list[0] / selected_node_number

        iidx += 1

    del out36_list_set, target_list_set
    del out36_list, target_list, out34_list, input_list_36, mean_list_36, bias_var_list_36
    del grad_model_4_origin_input_0_list
    del grad_model_36_2_shortcut_input_0_list, grad_bn36_11_input_0_list, grad_bn36_11_input_1_list, grad_bn36_12_input_0_list

    # ---- train_backward_35_2
    grad_bn35_11_input_0_list_set = []
    iidx = 0
    for idx in selected_local_device_index:
        input_list_35 = input_list_35_set[iidx]
        mean_list_35 = mean_list_35_set_average  # mean_list_35_set[iidx]
        bias_var_list_35 = bias_var_list_35_set_average  # bias_var_list_35_set[iidx]
        input_list_36 = input_list_36_set[iidx]
        mean_list_36 = mean_list_36_set_average  # mean_list_36_set[iidx]
        grad_bn36_11_input_0_list = grad_bn36_11_input_0_list_set[iidx]
        grad_bn36_11_input_1_list = grad_bn36_11_input_1_list_set_average  # grad_bn36_11_input_1_list_set[iidx]
        grad_bn36_12_input_0_list = grad_bn36_12_input_0_list_set_average  # grad_bn36_12_input_0_list_set[iidx]

        grad_bn_11_input_0 = torch.rand([channels_35, batchsize * height_35 * width_35])
        grad_bn_11_input_1 = torch.rand(channels_35)
        grad_bn_12_input_0 = torch.rand(channels_35)

        optimizer_35_2 = torch.optim.SGD([{'params': local_list_35_2[idx].parameters()}], args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

        (grad_bn35_11_input_0_list,
         grad_bn35_11_input_1_list,
         grad_bn35_12_input_0_list) = train_backward_2(local_list_35_2[idx], optimizer_35_2,
                                                       input_list_35, mean_list_35, bias_var_list_35,
                                                       batchsize, channels_35, height_35, width_35,
                                                       grad_bn36_11_input_0_list, grad_bn36_11_input_1_list,
                                                       grad_bn36_12_input_0_list,
                                                       input_list_36, mean_list_36, device)

        grad_bn35_11_input_0_list_set.append(grad_bn35_11_input_0_list)
        # ==== average mean gradient
        if iidx == 0:
            grad_bn35_11_input_1_list_set_average = [torch.zeros_like(grad_bn35_11_input_1_list[0])]
        grad_bn35_11_input_1_list_set_average[0] += grad_bn35_11_input_1_list[0] / selected_node_number
        # ==== average variance gradient
        if iidx == 0:
            grad_bn35_12_input_0_list_set_average = [torch.zeros_like(grad_bn35_12_input_0_list[0])]
        grad_bn35_12_input_0_list_set_average[0] += grad_bn35_12_input_0_list[0] / selected_node_number
        iidx += 1

    del input_list_36_set, grad_bn36_11_input_0_list_set
    del input_list_35, mean_list_35, bias_var_list_35, input_list_36, mean_list_36
    del grad_bn36_11_input_0_list, grad_bn36_11_input_1_list, grad_bn36_12_input_0_list
    del grad_bn35_11_input_0_list, grad_bn35_11_input_1_list, grad_bn35_12_input_0_list

    (grad_model_34_2_shortcut_input_0_list_set,
     grad_bn34_11_input_0_list_set) = [], []
    iidx = 0
    for idx in selected_local_device_index:
        out34_list = out34_list_set[iidx]
        input_list_35 = input_list_35_set[iidx]
        mean_list_35 = mean_list_35_set_average  # mean_list_35_set[iidx]
        grad_bn35_11_input_0_list = grad_bn35_11_input_0_list_set[iidx]
        grad_bn35_11_input_1_list = grad_bn35_11_input_1_list_set_average  # grad_bn35_11_input_1_list_set[iidx]
        grad_bn35_12_input_0_list = grad_bn35_12_input_0_list_set_average  # grad_bn35_12_input_0_list_set[iidx]

        out32_list = out32_list_set[iidx]
        input_list_34 = input_list_34_set[iidx]
        mean_list_34 = mean_list_34_set_average  # mean_list_34_set[iidx]
        bias_var_list_34 = bias_var_list_34_set_average  # bias_var_list_34_set[iidx]
        grad_model_36_2_shortcut_input_0_list = grad_model_36_2_shortcut_input_0_list_set[iidx]

        # ---- train_backward_35_1
        grad_model_1_conv_input_0 = torch.rand([batchsize, channels_34, height_34, width_34])

        optimizer_35_1 = torch.optim.SGD([{'params': local_list_35_1[idx].parameters()}], args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

        grad_model_35_1_conv_input_0_list = train_backward_1_Input(local_list_35_1[idx], optimizer_35_1, out34_list,
                                                                   grad_bn35_11_input_0_list, grad_bn35_11_input_1_list,
                                                                   grad_bn35_12_input_0_list,
                                                                   input_list_35, mean_list_35)

        # ---- train_backward_34_2
        grad_model_3_shortcut_input_0 = torch.rand([batchsize, channels_32, height_32, width_32])
        grad_bn_11_input_0 = torch.rand([channels_34, batchsize * height_34 * width_34])
        grad_bn_11_input_1 = torch.rand(channels_34)
        grad_bn_12_input_0 = torch.rand(channels_34)

        optimizer_34_2 = torch.optim.SGD([{'params': local_list_34_2[idx].parameters()}], args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

        (grad_model_34_2_shortcut_input_0_list,
         grad_bn34_11_input_0_list,
         grad_bn34_11_input_1_list,
         grad_bn34_12_input_0_list) = train_backward_3_ResInOut(local_list_34_2[idx], optimizer_34_2, out32_list,
                                                                input_list_34, mean_list_34, bias_var_list_34,
                                                                batchsize, channels_34, height_34, width_34,
                                                                grad_model_35_1_conv_input_0_list,
                                                                grad_model_36_2_shortcut_input_0_list, device)

        grad_model_34_2_shortcut_input_0_list_set.append(grad_model_34_2_shortcut_input_0_list)
        grad_bn34_11_input_0_list_set.append(grad_bn34_11_input_0_list)
        # ==== average mean gradient
        if iidx == 0:
            grad_bn34_11_input_1_list_set_average = [torch.zeros_like(grad_bn34_11_input_1_list[0])]
        grad_bn34_11_input_1_list_set_average[0] += grad_bn34_11_input_1_list[0] / selected_node_number
        # ==== average variance gradient
        if iidx == 0:
            grad_bn34_12_input_0_list_set_average = [torch.zeros_like(grad_bn34_12_input_0_list[0])]
        grad_bn34_12_input_0_list_set_average[0] += grad_bn34_12_input_0_list[0] / selected_node_number
        iidx += 1

    del out34_list_set, input_list_35_set, grad_bn35_11_input_0_list_set, grad_model_36_2_shortcut_input_0_list_set
    del out34_list, input_list_35, mean_list_35
    del grad_bn35_11_input_0_list, grad_bn35_11_input_1_list, grad_bn35_12_input_0_list
    del out32_list, input_list_34, mean_list_34, bias_var_list_34
    del grad_model_36_2_shortcut_input_0_list
    del grad_model_35_1_conv_input_0_list
    del grad_model_34_2_shortcut_input_0_list, grad_bn34_11_input_0_list, grad_bn34_11_input_1_list, grad_bn34_12_input_0_list

    # ---- train_backward_33_2
    grad_bn33_11_input_0_list_set = []
    iidx = 0
    for idx in selected_local_device_index:
        input_list_33 = input_list_33_set[iidx]
        mean_list_33 = mean_list_33_set_average  # mean_list_33_set[iidx]
        bias_var_list_33 = bias_var_list_33_set_average  # bias_var_list_33_set[iidx]
        input_list_34 = input_list_34_set[iidx]
        mean_list_34 = mean_list_34_set_average  # mean_list_34_set[iidx]
        grad_bn34_11_input_0_list = grad_bn34_11_input_0_list_set[iidx]
        grad_bn34_11_input_1_list = grad_bn34_11_input_1_list_set_average  # grad_bn34_11_input_1_list_set[iidx]
        grad_bn34_12_input_0_list = grad_bn34_12_input_0_list_set_average  # grad_bn34_12_input_0_list_set[iidx]

        grad_bn_11_input_0 = torch.rand([channels_33, batchsize * height_33 * width_33])
        grad_bn_11_input_1 = torch.rand(channels_33)
        grad_bn_12_input_0 = torch.rand(channels_33)

        optimizer_33_2 = torch.optim.SGD([{'params': local_list_33_2[idx].parameters()}], args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

        (grad_bn33_11_input_0_list,
         grad_bn33_11_input_1_list,
         grad_bn33_12_input_0_list) = train_backward_2(local_list_33_2[idx], optimizer_33_2,
                                                       input_list_33, mean_list_33, bias_var_list_33,
                                                       batchsize, channels_33, height_33, width_33,
                                                       grad_bn34_11_input_0_list, grad_bn34_11_input_1_list,
                                                       grad_bn34_12_input_0_list,
                                                       input_list_34, mean_list_34, device)

        grad_bn33_11_input_0_list_set.append(grad_bn33_11_input_0_list)
        # ==== average mean gradient
        if iidx == 0:
            grad_bn33_11_input_1_list_set_average = [torch.zeros_like(grad_bn33_11_input_1_list[0])]
        grad_bn33_11_input_1_list_set_average[0] += grad_bn33_11_input_1_list[0] / selected_node_number
        # ==== average variance gradient
        if iidx == 0:
            grad_bn33_12_input_0_list_set_average = [torch.zeros_like(grad_bn33_12_input_0_list[0])]
        grad_bn33_12_input_0_list_set_average[0] += grad_bn33_12_input_0_list[0] / selected_node_number
        iidx += 1

    del input_list_34_set, grad_bn34_11_input_0_list_set
    del input_list_33, mean_list_33, bias_var_list_33, input_list_34, mean_list_34
    del grad_bn34_11_input_0_list, grad_bn34_11_input_1_list, grad_bn34_12_input_0_list
    del grad_bn33_11_input_0_list, grad_bn33_11_input_1_list, grad_bn33_12_input_0_list

    (grad_model_32_2_shortcut_input_0_list_set,
     grad_bn32_11_input_0_list_set) = [], []
    iidx = 0
    for idx in selected_local_device_index:
        out32_list = out32_list_set[iidx]
        input_list_33 = input_list_33_set[iidx]
        mean_list_33 = mean_list_33_set_average  # mean_list_33_set[iidx]
        grad_bn33_11_input_0_list = grad_bn33_11_input_0_list_set[iidx]
        grad_bn33_11_input_1_list = grad_bn33_11_input_1_list_set_average  # grad_bn33_11_input_1_list_set[iidx]
        grad_bn33_12_input_0_list = grad_bn33_12_input_0_list_set_average  # grad_bn33_12_input_0_list_set[iidx]

        out26_list = out26_list_set[iidx]
        input_list_32 = input_list_32_set[iidx]
        mean_list_32 = mean_list_32_set_average  # mean_list_32_set[iidx]
        bias_var_list_32 = bias_var_list_32_set_average  # bias_var_list_32_set[iidx]
        grad_model_34_2_shortcut_input_0_list = grad_model_34_2_shortcut_input_0_list_set[iidx]

        # ---- train_backward_33_1
        grad_model_1_conv_input_0 = torch.rand([batchsize, channels_32, height_32, width_32])

        optimizer_33_1 = torch.optim.SGD([{'params': local_list_33_1[idx].parameters()}], args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

        grad_model_33_1_conv_input_0_list = train_backward_1_Input(local_list_33_1[idx], optimizer_33_1, out32_list,
                                                                   grad_bn33_11_input_0_list, grad_bn33_11_input_1_list,
                                                                   grad_bn33_12_input_0_list,
                                                                   input_list_33, mean_list_33)

        # ---- train_backward_32_2
        grad_model_3_shortcut_input_0 = torch.rand([batchsize, channels_26, height_26, width_26])
        grad_bn_11_input_0 = torch.rand([channels_32, batchsize * height_32 * width_32])
        grad_bn_11_input_1 = torch.rand(channels_32)
        grad_bn_12_input_0 = torch.rand(channels_32)

        optimizer_32_2 = torch.optim.SGD([{'params': local_list_32_2[idx].parameters()}], args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

        (grad_model_32_2_shortcut_input_0_list,
         grad_bn32_11_input_0_list,
         grad_bn32_11_input_1_list,
         grad_bn32_12_input_0_list) = train_backward_3_ResInOut(local_list_32_2[idx], optimizer_32_2, out26_list,
                                                                input_list_32, mean_list_32, bias_var_list_32,
                                                                batchsize, channels_32, height_32, width_32,
                                                                grad_model_33_1_conv_input_0_list,
                                                                grad_model_34_2_shortcut_input_0_list, device)

        grad_model_32_2_shortcut_input_0_list_set.append(grad_model_32_2_shortcut_input_0_list)
        grad_bn32_11_input_0_list_set.append(grad_bn32_11_input_0_list)
        # ==== average mean gradient
        if iidx == 0:
            grad_bn32_11_input_1_list_set_average = [torch.zeros_like(grad_bn32_11_input_1_list[0])]
        grad_bn32_11_input_1_list_set_average[0] += grad_bn32_11_input_1_list[0] / selected_node_number
        # ==== average variance gradient
        if iidx == 0:
            grad_bn32_12_input_0_list_set_average = [torch.zeros_like(grad_bn32_12_input_0_list[0])]
        grad_bn32_12_input_0_list_set_average[0] += grad_bn32_12_input_0_list[0] / selected_node_number
        iidx += 1

    del out32_list_set, input_list_33_set, grad_bn33_11_input_0_list_set, grad_model_34_2_shortcut_input_0_list_set
    del out32_list, input_list_33, mean_list_33
    del grad_bn33_11_input_0_list, grad_bn33_11_input_1_list, grad_bn33_12_input_0_list
    del out26_list, input_list_32, mean_list_32, bias_var_list_32
    del grad_model_34_2_shortcut_input_0_list, grad_model_33_1_conv_input_0_list
    del grad_model_32_2_shortcut_input_0_list, grad_bn32_11_input_0_list, grad_bn32_11_input_1_list, grad_bn32_12_input_0_list

    # ---- train_backward_31_2
    grad_bn31_11_input_0_list_set = []
    iidx = 0
    for idx in selected_local_device_index:
        input_list_31 = input_list_31_set[iidx]
        mean_list_31 = mean_list_31_set_average  # mean_list_31_set[iidx]
        bias_var_list_31 = bias_var_list_31_set_average  # bias_var_list_31_set[iidx]
        input_list_32 = input_list_32_set[iidx]
        mean_list_32 = mean_list_32_set_average  # mean_list_32_set[iidx]
        grad_bn32_11_input_0_list = grad_bn32_11_input_0_list_set[iidx]
        grad_bn32_11_input_1_list = grad_bn32_11_input_1_list_set_average  # grad_bn32_11_input_1_list_set[iidx]
        grad_bn32_12_input_0_list = grad_bn32_12_input_0_list_set_average  # grad_bn32_12_input_0_list_set[iidx]

        grad_bn_11_input_0 = torch.rand([channels_31, batchsize * height_31 * width_31])
        grad_bn_11_input_1 = torch.rand(channels_31)
        grad_bn_12_input_0 = torch.rand(channels_31)

        optimizer_31_2 = torch.optim.SGD([{'params': local_list_31_2[idx].parameters()}], args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

        (grad_bn31_11_input_0_list,
         grad_bn31_11_input_1_list,
         grad_bn31_12_input_0_list) = train_backward_2(local_list_31_2[idx], optimizer_31_2,
                                                       input_list_31, mean_list_31, bias_var_list_31,
                                                       batchsize, channels_31, height_31, width_31,
                                                       grad_bn32_11_input_0_list, grad_bn32_11_input_1_list,
                                                       grad_bn32_12_input_0_list,
                                                       input_list_32, mean_list_32, device)

        grad_bn31_11_input_0_list_set.append(grad_bn31_11_input_0_list)
        # ==== average mean gradient
        if iidx == 0:
            grad_bn31_11_input_1_list_set_average = [torch.zeros_like(grad_bn31_11_input_1_list[0])]
        grad_bn31_11_input_1_list_set_average[0] += grad_bn31_11_input_1_list[0] / selected_node_number
        # ==== average variance gradient
        if iidx == 0:
            grad_bn31_12_input_0_list_set_average = [torch.zeros_like(grad_bn31_12_input_0_list[0])]
        grad_bn31_12_input_0_list_set_average[0] += grad_bn31_12_input_0_list[0] / selected_node_number
        iidx += 1

    del input_list_32_set, grad_bn32_11_input_0_list_set
    del input_list_31, mean_list_31, bias_var_list_31, input_list_32, mean_list_32
    del grad_bn32_11_input_0_list, grad_bn32_11_input_1_list, grad_bn32_12_input_0_list
    del grad_bn31_11_input_0_list, grad_bn31_11_input_1_list, grad_bn31_12_input_0_list

    (grad_model_26_2_shortcut_input_0_list_set,
     grad_bn26_11_input_0_list_set) = [], []
    iidx = 0
    for idx in selected_local_device_index:
        out26_list = out26_list_set[iidx]
        input_list_31 = input_list_31_set[iidx]
        mean_list_31 = mean_list_31_set_average  # mean_list_31_set[iidx]
        grad_bn31_11_input_0_list = grad_bn31_11_input_0_list_set[iidx]
        grad_bn31_11_input_1_list = grad_bn31_11_input_1_list_set_average  # grad_bn31_11_input_1_list_set[iidx]
        grad_bn31_12_input_0_list = grad_bn31_12_input_0_list_set_average  # grad_bn31_12_input_0_list_set[iidx]

        out24_list = out24_list_set[iidx]
        input_list_26 = input_list_26_set[iidx]
        mean_list_26 = mean_list_26_set_average  # mean_list_26_set[iidx]
        bias_var_list_26 = bias_var_list_26_set_average  # bias_var_list_26_set[iidx]
        grad_model_32_2_shortcut_input_0_list = grad_model_32_2_shortcut_input_0_list_set[iidx]

        # ---- train_backward_31_1
        grad_model_1_conv_input_0 = torch.rand([batchsize, channels_26, height_26, width_26])
        optimizer_31_1 = torch.optim.SGD([{'params': local_list_31_1[idx].parameters()}], args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        grad_model_31_1_conv_input_0_list = train_backward_1_Input(local_list_31_1[idx], optimizer_31_1, out26_list,
                                                                   grad_bn31_11_input_0_list, grad_bn31_11_input_1_list,
                                                                   grad_bn31_12_input_0_list,
                                                                   input_list_31, mean_list_31)

        # block 2
        # ---- train_backward_26_2
        grad_model_3_shortcut_input_0 = torch.rand([batchsize, channels_24, height_24, width_24])
        grad_bn_11_input_0 = torch.rand([channels_26, batchsize * height_26 * width_26])
        grad_bn_11_input_1 = torch.rand(channels_26)
        grad_bn_12_input_0 = torch.rand(channels_26)
        optimizer_26_2 = torch.optim.SGD([{'params': local_list_26_2[idx].parameters()}], args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        (grad_model_26_2_shortcut_input_0_list,
         grad_bn26_11_input_0_list,
         grad_bn26_11_input_1_list,
         grad_bn26_12_input_0_list) = train_backward_3_ResInOut(local_list_26_2[idx],
                                                                optimizer_26_2,
                                                                out24_list,
                                                                input_list_26, mean_list_26, bias_var_list_26,
                                                                batchsize, channels_26, height_26, width_26,
                                                                grad_model_31_1_conv_input_0_list,
                                                                grad_model_32_2_shortcut_input_0_list, device)

        grad_model_26_2_shortcut_input_0_list_set.append(grad_model_26_2_shortcut_input_0_list)
        grad_bn26_11_input_0_list_set.append(grad_bn26_11_input_0_list)
        # ==== average mean gradient
        if iidx == 0:
            grad_bn26_11_input_1_list_set_average = [torch.zeros_like(grad_bn26_11_input_1_list[0])]
        grad_bn26_11_input_1_list_set_average[0] += grad_bn26_11_input_1_list[0] / selected_node_number
        # ==== average variance gradient
        if iidx == 0:
            grad_bn26_12_input_0_list_set_average = [torch.zeros_like(grad_bn26_12_input_0_list[0])]
        grad_bn26_12_input_0_list_set_average[0] += grad_bn26_12_input_0_list[0] / selected_node_number
        iidx += 1

    del out26_list_set, input_list_31_set, grad_bn31_11_input_0_list_set, grad_model_32_2_shortcut_input_0_list_set
    del out26_list, input_list_31, mean_list_31
    del grad_bn31_11_input_0_list, grad_bn31_11_input_1_list, grad_bn31_12_input_0_list
    del out24_list, input_list_26, mean_list_26, bias_var_list_26
    del grad_model_32_2_shortcut_input_0_list, grad_model_31_1_conv_input_0_list
    del grad_model_26_2_shortcut_input_0_list, grad_bn26_11_input_0_list, grad_bn26_11_input_1_list, grad_bn26_12_input_0_list

    # ---- train_backward_25_2
    grad_bn25_11_input_0_list_set = []
    iidx = 0
    for idx in selected_local_device_index:
        input_list_25 = input_list_25_set[iidx]
        mean_list_25 = mean_list_25_set_average  # mean_list_25_set[iidx]
        bias_var_list_25 = bias_var_list_25_set_average  # bias_var_list_25_set[iidx]
        input_list_26 = input_list_26_set[iidx]
        mean_list_26 = mean_list_26_set_average  # mean_list_26_set[iidx]
        grad_bn26_11_input_0_list = grad_bn26_11_input_0_list_set[iidx]
        grad_bn26_11_input_1_list = grad_bn26_11_input_1_list_set_average  # grad_bn26_11_input_1_list_set[iidx]
        grad_bn26_12_input_0_list = grad_bn26_12_input_0_list_set_average  # grad_bn26_12_input_0_list_set[iidx]

        grad_bn_11_input_0 = torch.rand([channels_25, batchsize * height_25 * width_25])
        grad_bn_11_input_1 = torch.rand(channels_25)
        grad_bn_12_input_0 = torch.rand(channels_25)

        optimizer_25_2 = torch.optim.SGD([{'params': local_list_25_2[idx].parameters()}], args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        (grad_bn25_11_input_0_list,
         grad_bn25_11_input_1_list,
         grad_bn25_12_input_0_list) = train_backward_2(local_list_25_2[idx], optimizer_25_2,
                                                       input_list_25, mean_list_25, bias_var_list_25,
                                                       batchsize, channels_25, height_25, width_25,
                                                       grad_bn26_11_input_0_list, grad_bn26_11_input_1_list,
                                                       grad_bn26_12_input_0_list,
                                                       input_list_26, mean_list_26, device)
        grad_bn25_11_input_0_list_set.append(grad_bn25_11_input_0_list)
        # ==== average mean gradient
        if iidx == 0:
            grad_bn25_11_input_1_list_set_average = [torch.zeros_like(grad_bn25_11_input_1_list[0])]
        grad_bn25_11_input_1_list_set_average[0] += grad_bn25_11_input_1_list[0] / selected_node_number
        # ==== average variance gradient
        if iidx == 0:
            grad_bn25_12_input_0_list_set_average = [torch.zeros_like(grad_bn25_12_input_0_list[0])]
        grad_bn25_12_input_0_list_set_average[0] += grad_bn25_12_input_0_list[0] / selected_node_number
        iidx += 1

    del input_list_26_set, grad_bn26_11_input_0_list_set
    del input_list_25, mean_list_25, bias_var_list_25, input_list_26, mean_list_26
    del grad_bn26_11_input_0_list, grad_bn26_11_input_1_list, grad_bn26_12_input_0_list
    del grad_bn25_11_input_0_list, grad_bn25_11_input_1_list, grad_bn25_12_input_0_list

    (grad_model_24_2_shortcut_input_0_list_set,
     grad_bn24_11_input_0_list_set) = [], []
    iidx = 0
    for idx in selected_local_device_index:
        out24_list = out24_list_set[iidx]
        input_list_25 = input_list_25_set[iidx]
        mean_list_25 = mean_list_25_set_average  # mean_list_25_set[iidx]
        grad_bn25_11_input_0_list = grad_bn25_11_input_0_list_set[iidx]
        grad_bn25_11_input_1_list = grad_bn25_11_input_1_list_set_average  # grad_bn25_11_input_1_list_set[iidx]
        grad_bn25_12_input_0_list = grad_bn25_12_input_0_list_set_average  # grad_bn25_12_input_0_list_set[iidx]

        out22_list = out22_list_set[iidx]
        input_list_24 = input_list_24_set[iidx]
        mean_list_24 = mean_list_24_set_average  # mean_list_24_set[iidx]
        bias_var_list_24 = bias_var_list_24_set_average  # bias_var_list_24_set[iidx]
        grad_model_26_2_shortcut_input_0_list = grad_model_26_2_shortcut_input_0_list_set[iidx]

        # ---- train_backward_25_1
        grad_model_1_conv_input_0 = torch.rand([batchsize, channels_24, height_24, width_24])
        optimizer_25_1 = torch.optim.SGD([{'params': local_list_25_1[idx].parameters()}], args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        grad_model_25_1_conv_input_0_list = train_backward_1_Input(local_list_25_1[idx], optimizer_25_1, out24_list,
                                                                   grad_bn25_11_input_0_list, grad_bn25_11_input_1_list,
                                                                   grad_bn25_12_input_0_list,
                                                                   input_list_25, mean_list_25)

        # ---- train_backward_24_2
        grad_model_3_shortcut_input_0 = torch.rand([batchsize, channels_22, height_22, width_22])
        grad_bn_11_input_0 = torch.rand([channels_24, batchsize * height_24 * width_24])
        grad_bn_11_input_1 = torch.rand(channels_24)
        grad_bn_12_input_0 = torch.rand(channels_24)

        optimizer_24_2 = torch.optim.SGD([{'params': local_list_24_2[idx].parameters()}], args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        (grad_model_24_2_shortcut_input_0_list,
         grad_bn24_11_input_0_list,
         grad_bn24_11_input_1_list,
         grad_bn24_12_input_0_list) = train_backward_3_ResInOut(local_list_24_2[idx], optimizer_24_2, out22_list,
                                                                input_list_24, mean_list_24, bias_var_list_24,
                                                                batchsize, channels_24, height_24, width_24,
                                                                grad_model_25_1_conv_input_0_list,
                                                                grad_model_26_2_shortcut_input_0_list, device)

        grad_model_24_2_shortcut_input_0_list_set.append(grad_model_24_2_shortcut_input_0_list)
        grad_bn24_11_input_0_list_set.append(grad_bn24_11_input_0_list)
        # ==== average mean gradient
        if iidx == 0:
            grad_bn24_11_input_1_list_set_average = [torch.zeros_like(grad_bn24_11_input_1_list[0])]
        grad_bn24_11_input_1_list_set_average[0] += grad_bn24_11_input_1_list[0] / selected_node_number
        # ==== average variance gradient
        if iidx == 0:
            grad_bn24_12_input_0_list_set_average = [torch.zeros_like(grad_bn24_12_input_0_list[0])]
        grad_bn24_12_input_0_list_set_average[0] += grad_bn24_12_input_0_list[0] / selected_node_number
        iidx += 1

    del out24_list_set, input_list_25_set, grad_bn25_11_input_0_list_set, grad_model_26_2_shortcut_input_0_list_set
    del out24_list, input_list_25, mean_list_25
    del grad_bn25_11_input_0_list, grad_bn25_11_input_1_list, grad_bn25_12_input_0_list
    del out22_list, input_list_24, mean_list_24, bias_var_list_24
    del grad_model_26_2_shortcut_input_0_list, grad_model_25_1_conv_input_0_list
    del grad_model_24_2_shortcut_input_0_list, grad_bn24_11_input_0_list, grad_bn24_11_input_1_list, grad_bn24_12_input_0_list

    # ---- train_backward_23_2
    grad_bn23_11_input_0_list_set = []
    iidx = 0
    for idx in selected_local_device_index:
        input_list_23 = input_list_23_set[iidx]
        mean_list_23 = mean_list_23_set_average  # mean_list_23_set[iidx]
        bias_var_list_23 = bias_var_list_23_set_average  # bias_var_list_23_set[iidx]
        input_list_24 = input_list_24_set[iidx]
        mean_list_24 = mean_list_24_set_average  # mean_list_24_set[iidx]
        grad_bn24_11_input_0_list = grad_bn24_11_input_0_list_set[iidx]
        grad_bn24_11_input_1_list = grad_bn24_11_input_1_list_set_average  # grad_bn24_11_input_1_list_set[iidx]
        grad_bn24_12_input_0_list = grad_bn24_12_input_0_list_set_average  # grad_bn24_12_input_0_list_set[iidx]

        grad_bn_11_input_0 = torch.rand([channels_23, batchsize * height_23 * width_23])
        grad_bn_11_input_1 = torch.rand(channels_23)
        grad_bn_12_input_0 = torch.rand(channels_23)
        optimizer_23_2 = torch.optim.SGD([{'params': local_list_23_2[idx].parameters()}], args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        (grad_bn23_11_input_0_list,
         grad_bn23_11_input_1_list,
         grad_bn23_12_input_0_list) = train_backward_2(local_list_23_2[idx], optimizer_23_2,
                                                       input_list_23, mean_list_23, bias_var_list_23,
                                                       batchsize, channels_23, height_23, width_23,
                                                       grad_bn24_11_input_0_list, grad_bn24_11_input_1_list,
                                                       grad_bn24_12_input_0_list,
                                                       input_list_24, mean_list_24, device)

        grad_bn23_11_input_0_list_set.append(grad_bn23_11_input_0_list)
        # ==== average mean gradient
        if iidx == 0:
            grad_bn23_11_input_1_list_set_average = [torch.zeros_like(grad_bn23_11_input_1_list[0])]
        grad_bn23_11_input_1_list_set_average[0] += grad_bn23_11_input_1_list[0] / selected_node_number
        # ==== average variance gradient
        if iidx == 0:
            grad_bn23_12_input_0_list_set_average = [torch.zeros_like(grad_bn23_12_input_0_list[0])]
        grad_bn23_12_input_0_list_set_average[0] += grad_bn23_12_input_0_list[0] / selected_node_number
        iidx += 1

    del input_list_24_set, grad_bn24_11_input_0_list_set
    del input_list_23, mean_list_23, bias_var_list_23, input_list_24, mean_list_24
    del grad_bn24_11_input_0_list, grad_bn24_11_input_1_list, grad_bn24_12_input_0_list
    del grad_bn23_11_input_0_list, grad_bn23_11_input_1_list, grad_bn23_12_input_0_list

    # ---- train_backward_23_1
    (grad_model_22_2_shortcut_input_0_list_set,
     grad_bn22_11_input_0_list_set) = [], []
    iidx = 0
    for idx in selected_local_device_index:
        out22_list = out22_list_set[iidx]
        input_list_23 = input_list_23_set[iidx]
        mean_list_23 = mean_list_23_set_average  # mean_list_23_set[iidx]
        grad_bn23_11_input_0_list = grad_bn23_11_input_0_list_set[iidx]
        grad_bn23_11_input_1_list = grad_bn23_11_input_1_list_set_average  # grad_bn23_11_input_1_list_set[iidx]
        grad_bn23_12_input_0_list = grad_bn23_12_input_0_list_set_average  # grad_bn23_12_input_0_list_set[iidx]

        out16_list = out16_list_set[iidx]
        input_list_22 = input_list_22_set[iidx]
        mean_list_22 = mean_list_22_set_average  # mean_list_22_set[iidx]
        bias_var_list_22 = bias_var_list_22_set_average  # bias_var_list_22_set[iidx]
        grad_model_24_2_shortcut_input_0_list = grad_model_24_2_shortcut_input_0_list_set[iidx]

        grad_model_1_conv_input_0 = torch.rand([batchsize, channels_22, height_22, width_22])

        optimizer_23_1 = torch.optim.SGD([{'params': local_list_23_1[idx].parameters()}], args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

        grad_model_23_1_conv_input_0_list = train_backward_1_Input(local_list_23_1[idx], optimizer_23_1, out22_list,
                                                                   grad_bn23_11_input_0_list, grad_bn23_11_input_1_list,
                                                                   grad_bn23_12_input_0_list,
                                                                   input_list_23, mean_list_23)

        # ---- train_backward_22_2
        grad_model_3_shortcut_input_0 = torch.rand([batchsize, channels_16, height_16, width_16])
        grad_bn_11_input_0 = torch.rand([channels_22, batchsize * height_22 * width_22])
        grad_bn_11_input_1 = torch.rand(channels_22)
        grad_bn_12_input_0 = torch.rand(channels_22)
        optimizer_22_2 = torch.optim.SGD([{'params': local_list_22_2[idx].parameters()}], args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        (grad_model_22_2_shortcut_input_0_list,
         grad_bn22_11_input_0_list,
         grad_bn22_11_input_1_list,
         grad_bn22_12_input_0_list) = train_backward_3_ResInOut(local_list_22_2[idx], optimizer_22_2, out16_list,
                                                                input_list_22, mean_list_22, bias_var_list_22,
                                                                batchsize, channels_22, height_22, width_22,
                                                                grad_model_23_1_conv_input_0_list,
                                                                grad_model_24_2_shortcut_input_0_list, device)

        grad_model_22_2_shortcut_input_0_list_set.append(grad_model_22_2_shortcut_input_0_list)
        grad_bn22_11_input_0_list_set.append(grad_bn22_11_input_0_list)
        # ==== average mean gradient
        if iidx == 0:
            grad_bn22_11_input_1_list_set_average = [torch.zeros_like(grad_bn22_11_input_1_list[0])]
        grad_bn22_11_input_1_list_set_average[0] += grad_bn22_11_input_1_list[0] / selected_node_number
        # ==== average variance gradient
        if iidx == 0:
            grad_bn22_12_input_0_list_set_average = [torch.zeros_like(grad_bn22_12_input_0_list[0])]
        grad_bn22_12_input_0_list_set_average[0] += grad_bn22_12_input_0_list[0] / selected_node_number
        iidx += 1

    del out22_list_set, input_list_23_set, grad_bn23_11_input_0_list_set, grad_model_24_2_shortcut_input_0_list_set
    del out22_list, input_list_23, mean_list_23
    del grad_bn23_11_input_0_list, grad_bn23_11_input_1_list, grad_bn23_12_input_0_list
    del out16_list, input_list_22, mean_list_22, bias_var_list_22, bias_var_list_22_set_average
    del grad_model_24_2_shortcut_input_0_list, grad_model_23_1_conv_input_0_list
    del grad_model_22_2_shortcut_input_0_list, grad_bn22_11_input_0_list, grad_bn22_11_input_1_list, grad_bn22_12_input_0_list

    # ---- train_backward_21_2
    grad_bn21_11_input_0_list_set = []
    iidx = 0
    for idx in selected_local_device_index:
        input_list_21 = input_list_21_set[iidx]
        mean_list_21 = mean_list_21_set_average  # mean_list_21_set[iidx]
        bias_var_list_21 = bias_var_list_21_set_average  # bias_var_list_21_set[iidx]
        input_list_22 = input_list_22_set[iidx]
        mean_list_22 = mean_list_22_set_average  # mean_list_22_set[iidx]
        grad_bn22_11_input_0_list = grad_bn22_11_input_0_list_set[iidx]
        grad_bn22_11_input_1_list = grad_bn22_11_input_1_list_set_average  # grad_bn22_11_input_1_list_set[iidx]
        grad_bn22_12_input_0_list = grad_bn22_12_input_0_list_set_average  # grad_bn22_12_input_0_list_set[iidx]

        grad_bn_11_input_0 = torch.rand([channels_21, batchsize * height_21 * width_21])
        grad_bn_11_input_1 = torch.rand(channels_21)
        grad_bn_12_input_0 = torch.rand(channels_21)
        optimizer_21_2 = torch.optim.SGD([{'params': local_list_21_2[idx].parameters()}], args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        (grad_bn21_11_input_0_list,
         grad_bn21_11_input_1_list,
         grad_bn21_12_input_0_list) = train_backward_2(local_list_21_2[idx], optimizer_21_2,
                                                       input_list_21, mean_list_21, bias_var_list_21,
                                                       batchsize, channels_21, height_21, width_21,
                                                       grad_bn22_11_input_0_list, grad_bn22_11_input_1_list,
                                                       grad_bn22_12_input_0_list,
                                                       input_list_22, mean_list_22, device)

        grad_bn21_11_input_0_list_set.append(grad_bn21_11_input_0_list)
        # ==== average mean gradient
        if iidx == 0:
            grad_bn21_11_input_1_list_set_average = [torch.zeros_like(grad_bn21_11_input_1_list[0])]
        grad_bn21_11_input_1_list_set_average[0] += grad_bn21_11_input_1_list[0] / selected_node_number
        # ==== average variance gradient
        if iidx == 0:
            grad_bn21_12_input_0_list_set_average = [torch.zeros_like(grad_bn21_12_input_0_list[0])]
        grad_bn21_12_input_0_list_set_average[0] += grad_bn21_12_input_0_list[0] / selected_node_number
        iidx += 1

    del input_list_22_set, grad_bn22_11_input_0_list_set
    del input_list_21, mean_list_21, bias_var_list_21, input_list_22, mean_list_22
    del grad_bn22_11_input_0_list, grad_bn22_11_input_1_list, grad_bn22_12_input_0_list
    del grad_bn21_11_input_0_list, grad_bn21_11_input_1_list, grad_bn21_12_input_0_list

    (grad_model_16_2_shortcut_input_0_list_set,
     grad_bn16_11_input_0_list_set) = [], []
    iidx = 0
    for idx in selected_local_device_index:
        out16_list = out16_list_set[iidx]
        input_list_21 = input_list_21_set[iidx]
        mean_list_21 = mean_list_21_set_average  # mean_list_21_set[iidx]
        grad_bn21_11_input_0_list = grad_bn21_11_input_0_list_set[iidx]
        grad_bn21_11_input_1_list = grad_bn21_11_input_1_list_set_average  # grad_bn21_11_input_1_list_set[iidx]
        grad_bn21_12_input_0_list = grad_bn21_12_input_0_list_set_average  # grad_bn21_12_input_0_list_set[iidx]

        out14_list = out16_list_set[iidx]
        input_list_16 = input_list_16_set[iidx]
        mean_list_16 = mean_list_16_set_average  # mean_list_16_set[iidx]
        bias_var_list_16 = bias_var_list_16_set_average  # bias_var_list_16_set[iidx]
        grad_model_22_2_shortcut_input_0_list = grad_model_22_2_shortcut_input_0_list_set[iidx]

        # ---- train_backward_21_1
        grad_model_1_conv_input_0 = torch.rand([batchsize, channels_16, height_16, width_16])
        optimizer_21_1 = torch.optim.SGD([{'params': local_list_21_1[idx].parameters()}], args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        grad_model_21_1_conv_input_0_list = train_backward_1_Input(local_list_21_1[idx], optimizer_21_1, out16_list,
                                                                   grad_bn21_11_input_0_list, grad_bn21_11_input_1_list,
                                                                   grad_bn21_12_input_0_list,
                                                                   input_list_21, mean_list_21)

        # block 1
        # ---- train_backward_16_2
        grad_model_3_shortcut_input_0 = torch.rand([batchsize, channels_14, height_14, width_14])
        grad_bn_11_input_0 = torch.rand([channels_16, batchsize * height_16 * width_16])
        grad_bn_11_input_1 = torch.rand(channels_16)
        grad_bn_12_input_0 = torch.rand(channels_16)
        optimizer_16_2 = torch.optim.SGD([{'params': local_list_16_2[idx].parameters()}], args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        (grad_model_16_2_shortcut_input_0_list,
         grad_bn16_11_input_0_list,
         grad_bn16_11_input_1_list,
         grad_bn16_12_input_0_list) = train_backward_3_ResInOut(local_list_16_2[idx], optimizer_16_2, out14_list,
                                                                input_list_16, mean_list_16, bias_var_list_16,
                                                                batchsize, channels_16, height_16, width_16,
                                                                grad_model_21_1_conv_input_0_list,
                                                                grad_model_22_2_shortcut_input_0_list, device)

        grad_model_16_2_shortcut_input_0_list_set.append(grad_model_16_2_shortcut_input_0_list)
        grad_bn16_11_input_0_list_set.append(grad_bn16_11_input_0_list)
        # ==== average mean gradient
        if iidx == 0:
            grad_bn16_11_input_1_list_set_average = [torch.zeros_like(grad_bn16_11_input_1_list[0])]
        grad_bn16_11_input_1_list_set_average[0] += grad_bn16_11_input_1_list[0] / selected_node_number
        # ==== average variance gradient
        if iidx == 0:
            grad_bn16_12_input_0_list_set_average = [torch.zeros_like(grad_bn16_12_input_0_list[0])]
        grad_bn16_12_input_0_list_set_average[0] += grad_bn16_12_input_0_list[0] / selected_node_number
        iidx += 1

    del input_list_21_set, grad_bn21_11_input_0_list_set, out16_list_set, grad_model_22_2_shortcut_input_0_list_set
    del out16_list, input_list_21, mean_list_21,
    del grad_bn21_11_input_0_list, grad_bn21_11_input_1_list, grad_bn21_12_input_0_list
    del out14_list, input_list_16, mean_list_16, bias_var_list_16
    del grad_model_22_2_shortcut_input_0_list, grad_model_21_1_conv_input_0_list
    del grad_model_16_2_shortcut_input_0_list, grad_bn16_11_input_0_list, grad_bn16_11_input_1_list, grad_bn16_12_input_0_list

    grad_bn15_11_input_0_list_set = []
    iidx = 0
    for idx in selected_local_device_index:
        input_list_15 = input_list_15_set[iidx]
        mean_list_15 = mean_list_15_set_average  # mean_list_15_set[iidx]
        bias_var_list_15 = bias_var_list_15_set_average  # bias_var_list_15_set[iidx]
        input_list_16 = input_list_16_set[iidx]
        mean_list_16 = mean_list_16_set_average  # mean_list_16_set[iidx]
        grad_bn16_11_input_0_list = grad_bn16_11_input_0_list_set[iidx]
        grad_bn16_11_input_1_list = grad_bn16_11_input_1_list_set_average  # grad_bn16_11_input_1_list_set[iidx]
        grad_bn16_12_input_0_list = grad_bn16_12_input_0_list_set_average  # grad_bn16_12_input_0_list_set[iidx]

        # ---- train_backward_15_2
        grad_bn_11_input_0 = torch.rand([channels_15, batchsize * height_15 * width_15])
        grad_bn_11_input_1 = torch.rand(channels_15)
        grad_bn_12_input_0 = torch.rand(channels_15)

        optimizer_15_2 = torch.optim.SGD([{'params': local_list_15_2[idx].parameters()}], args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

        (grad_bn15_11_input_0_list,
         grad_bn15_11_input_1_list,
         grad_bn15_12_input_0_list) = train_backward_2(local_list_15_2[idx], optimizer_15_2,
                                                       input_list_15, mean_list_15, bias_var_list_15,
                                                       batchsize, channels_15, height_15, width_15,
                                                       grad_bn16_11_input_0_list, grad_bn16_11_input_1_list,
                                                       grad_bn16_12_input_0_list,
                                                       input_list_16, mean_list_16, device)

        grad_bn15_11_input_0_list_set.append(grad_bn15_11_input_0_list)
        # ==== average mean gradient
        if iidx == 0:
            grad_bn15_11_input_1_list_set_average = [torch.zeros_like(grad_bn15_11_input_1_list[0])]
        grad_bn15_11_input_1_list_set_average[0] += grad_bn15_11_input_1_list[0] / selected_node_number
        # ==== average variance gradient
        if iidx == 0:
            grad_bn15_12_input_0_list_set_average = [torch.zeros_like(grad_bn15_12_input_0_list[0])]
        grad_bn15_12_input_0_list_set_average[0] += grad_bn15_12_input_0_list[0] / selected_node_number
        iidx += 1

    del input_list_16_set, grad_bn16_11_input_0_list_set
    del input_list_15, mean_list_15, bias_var_list_15, input_list_16, mean_list_16
    del grad_bn16_11_input_0_list, grad_bn16_11_input_1_list, grad_bn16_12_input_0_list
    del grad_bn15_11_input_0_list, grad_bn15_11_input_1_list, grad_bn15_12_input_0_list

    (grad_model_14_2_shortcut_input_0_list_set,
     grad_bn14_11_input_0_list_set) = [], []
    iidx = 0
    for idx in selected_local_device_index:
        out14_list = out14_list_set[iidx]
        input_list_15 = input_list_15_set[iidx]
        mean_list_15 = mean_list_15_set_average  # mean_list_15_set[iidx]
        grad_bn15_11_input_0_list = grad_bn15_11_input_0_list_set[iidx]
        grad_bn15_11_input_1_list = grad_bn15_11_input_1_list_set_average  # grad_bn15_11_input_1_list_set[iidx]
        grad_bn15_12_input_0_list = grad_bn15_12_input_0_list_set_average  # grad_bn15_12_input_0_list_set[iidx]

        out12_list = out12_list_set[iidx]
        input_list_14 = input_list_14_set[iidx]
        mean_list_14 = mean_list_14_set_average  # mean_list_14_set[iidx]
        bias_var_list_14 = bias_var_list_14_set_average  # bias_var_list_14_set[iidx]
        grad_model_16_2_shortcut_input_0_list = grad_model_16_2_shortcut_input_0_list_set[iidx]

        # ---- train_backward_15_1
        grad_model_1_conv_input_0 = torch.rand([batchsize, channels_14, height_14, width_14])
        optimizer_15_1 = torch.optim.SGD([{'params': local_list_15_1[idx].parameters()}], args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        grad_model_15_1_conv_input_0_list = train_backward_1_Input(local_list_15_1[idx], optimizer_15_1, out14_list,
                                                                   grad_bn15_11_input_0_list, grad_bn15_11_input_1_list,
                                                                   grad_bn15_12_input_0_list,
                                                                   input_list_15, mean_list_15)

        # ---- train_backward_14_2
        grad_model_3_shortcut_input_0 = torch.rand([batchsize, channels_12, height_12, width_12])
        grad_bn_11_input_0 = torch.rand([channels_14, batchsize * height_14 * width_14])
        grad_bn_11_input_1 = torch.rand(channels_14)
        grad_bn_12_input_0 = torch.rand(channels_14)
        optimizer_14_2 = torch.optim.SGD([{'params': local_list_14_2[idx].parameters()}], args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        (grad_model_14_2_shortcut_input_0_list,
         grad_bn14_11_input_0_list,
         grad_bn14_11_input_1_list,
         grad_bn14_12_input_0_list) = train_backward_3_ResInOut(local_list_14_2[idx], optimizer_14_2, out12_list,
                                                                input_list_14, mean_list_14, bias_var_list_14,
                                                                batchsize, channels_14, height_14, width_14,
                                                                grad_model_15_1_conv_input_0_list,
                                                                grad_model_16_2_shortcut_input_0_list, device)

        grad_model_14_2_shortcut_input_0_list_set.append(grad_model_14_2_shortcut_input_0_list)
        grad_bn14_11_input_0_list_set.append(grad_bn14_11_input_0_list)
        # ==== average mean gradient
        if iidx == 0:
            grad_bn14_11_input_1_list_set_average = [torch.zeros_like(grad_bn14_11_input_1_list[0])]
        grad_bn14_11_input_1_list_set_average[0] += grad_bn14_11_input_1_list[0] / selected_node_number
        # ==== average variance gradient
        if iidx == 0:
            grad_bn14_12_input_0_list_set_average = [torch.zeros_like(grad_bn14_12_input_0_list[0])]
        grad_bn14_12_input_0_list_set_average[0] += grad_bn14_12_input_0_list[0] / selected_node_number
        iidx += 1

    del out14_list_set, input_list_15_set, grad_bn15_11_input_0_list_set, grad_model_16_2_shortcut_input_0_list_set
    del out14_list, input_list_15, mean_list_15
    del grad_bn15_11_input_0_list, grad_bn15_11_input_1_list, grad_bn15_12_input_0_list
    del out12_list, input_list_14, mean_list_14, bias_var_list_14
    del grad_model_16_2_shortcut_input_0_list, grad_model_15_1_conv_input_0_list
    del grad_model_14_2_shortcut_input_0_list, grad_bn14_11_input_0_list, grad_bn14_11_input_1_list, grad_bn14_12_input_0_list

    # ---- train_backward_13_2
    grad_bn13_11_input_0_list_set = []
    iidx = 0
    for idx in selected_local_device_index:
        input_list_13 = input_list_13_set[iidx]
        mean_list_13 = mean_list_13_set_average  # mean_list_13_set[iidx]
        bias_var_list_13 = bias_var_list_13_set_average  # bias_var_list_13_set[iidx]
        input_list_14 = input_list_14_set[iidx]
        mean_list_14 = mean_list_14_set_average  # mean_list_14_set[iidx]
        grad_bn14_11_input_0_list = grad_bn14_11_input_0_list_set[iidx]
        grad_bn14_11_input_1_list = grad_bn14_11_input_1_list_set_average  # grad_bn14_11_input_1_list_set[iidx]
        grad_bn14_12_input_0_list = grad_bn14_12_input_0_list_set_average  # grad_bn14_12_input_0_list_set[iidx]

        grad_bn_11_input_0 = torch.rand([channels_13, batchsize * height_13 * width_13])
        grad_bn_11_input_1 = torch.rand(channels_13)
        grad_bn_12_input_0 = torch.rand(channels_13)

        optimizer_13_2 = torch.optim.SGD([{'params': local_list_13_2[idx].parameters()}], args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        (grad_bn13_11_input_0_list,
         grad_bn13_11_input_1_list,
         grad_bn13_12_input_0_list) = train_backward_2(local_list_13_2[idx], optimizer_13_2,
                                                       input_list_13, mean_list_13, bias_var_list_13,
                                                       batchsize, channels_13, height_13, width_13,
                                                       grad_bn14_11_input_0_list, grad_bn14_11_input_1_list,
                                                       grad_bn14_12_input_0_list,
                                                       input_list_14, mean_list_14, device)
        grad_bn13_11_input_0_list_set.append(grad_bn13_11_input_0_list)
        # ==== average mean gradient
        if iidx == 0:
            grad_bn13_11_input_1_list_set_average = [torch.zeros_like(grad_bn13_11_input_1_list[0])]
        grad_bn13_11_input_1_list_set_average[0] += grad_bn13_11_input_1_list[0] / selected_node_number
        # ==== average variance gradient
        if iidx == 0:
            grad_bn13_12_input_0_list_set_average = [torch.zeros_like(grad_bn13_12_input_0_list[0])]
        grad_bn13_12_input_0_list_set_average[0] += grad_bn13_12_input_0_list[0] / selected_node_number
        iidx += 1

    del input_list_14_set, grad_bn14_11_input_0_list_set
    del input_list_13, mean_list_13, bias_var_list_13, input_list_14, mean_list_14
    del grad_bn14_11_input_0_list, grad_bn14_11_input_1_list, grad_bn14_12_input_0_list
    del grad_bn13_11_input_0_list, grad_bn13_11_input_1_list, grad_bn13_12_input_0_list

    (grad_model_12_2_shortcut_input_0_list_set,
     grad_bn12_11_input_0_list_set) = [], []
    iidx = 0
    for idx in selected_local_device_index:
        out12_list = out12_list_set[iidx]
        input_list_13 = input_list_13_set[iidx]
        mean_list_13 = mean_list_13_set_average  # mean_list_13_set[iidx]
        grad_bn13_11_input_0_list = grad_bn13_11_input_0_list_set[iidx]
        grad_bn13_11_input_1_list = grad_bn13_11_input_1_list_set_average  # grad_bn13_11_input_1_list_set[iidx]
        grad_bn13_12_input_0_list = grad_bn13_12_input_0_list_set_average  # grad_bn13_12_input_0_list_set[iidx]

        out0_list = out0_list_set[iidx]
        input_list_12 = input_list_12_set[iidx]
        mean_list_12 = mean_list_12_set_average  # mean_list_12_set[iidx]
        bias_var_list_12 = bias_var_list_12_set_average  # bias_var_list_12_set[iidx]
        grad_model_14_2_shortcut_input_0_list = grad_model_14_2_shortcut_input_0_list_set[iidx]

        # ---- train_backward_13_1
        grad_model_1_conv_input_0 = torch.rand([batchsize, channels_12, height_12, width_12])
        optimizer_13_1 = torch.optim.SGD([{'params': local_list_13_1[idx].parameters()}], args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        grad_model_13_1_conv_input_0_list = train_backward_1_Input(local_list_13_1[idx], optimizer_13_1, out12_list,
                                                                   grad_bn13_11_input_0_list, grad_bn13_11_input_1_list,
                                                                   grad_bn13_12_input_0_list,
                                                                   input_list_13, mean_list_13)

        # ---- train_backward_12_2
        grad_model_3_shortcut_input_0 = torch.rand([batchsize, channels_0, height_0, width_0])
        grad_bn_11_input_0 = torch.rand([channels_12, batchsize * height_12 * width_12])
        grad_bn_11_input_1 = torch.rand(channels_12)
        grad_bn_12_input_0 = torch.rand(channels_12)
        optimizer_12_2 = torch.optim.SGD([{'params': local_list_12_2[idx].parameters()}], args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        (grad_model_12_2_shortcut_input_0_list,
         grad_bn12_11_input_0_list,
         grad_bn12_11_input_1_list,
         grad_bn12_12_input_0_list) = train_backward_3_ResInOut(local_list_12_2[idx], optimizer_12_2, out0_list,
                                                                input_list_12, mean_list_12, bias_var_list_12,
                                                                batchsize, channels_12, height_12, width_12,
                                                                grad_model_13_1_conv_input_0_list,
                                                                grad_model_14_2_shortcut_input_0_list, device)

        grad_model_12_2_shortcut_input_0_list_set.append(grad_model_12_2_shortcut_input_0_list)
        grad_bn12_11_input_0_list_set.append(grad_bn12_11_input_0_list)
        # ==== average mean gradient
        if iidx == 0:
            grad_bn12_11_input_1_list_set_average = [torch.zeros_like(grad_bn12_11_input_1_list[0])]
        grad_bn12_11_input_1_list_set_average[0] += grad_bn12_11_input_1_list[0] / selected_node_number
        # ==== average variance gradient
        if iidx == 0:
            grad_bn12_12_input_0_list_set_average = [torch.zeros_like(grad_bn12_12_input_0_list[0])]
        grad_bn12_12_input_0_list_set_average[0] += grad_bn12_12_input_0_list[0] / selected_node_number
        iidx += 1

    del out12_list_set, input_list_13_set, grad_bn13_11_input_0_list_set
    del out12_list, input_list_13, mean_list_13
    del grad_bn13_11_input_0_list, grad_bn13_11_input_1_list, grad_bn13_12_input_0_list
    del out0_list, input_list_12, mean_list_12, bias_var_list_12
    del grad_model_14_2_shortcut_input_0_list, grad_model_13_1_conv_input_0_list
    del grad_model_12_2_shortcut_input_0_list, grad_bn12_11_input_0_list, grad_bn12_11_input_1_list, grad_bn12_12_input_0_list

    # ---- train_backward_11_2
    grad_bn11_11_input_0_list_set = []
    iidx = 0
    for idx in selected_local_device_index:
        input_list_11 = input_list_11_set[iidx]
        mean_list_11 = mean_list_11_set_average  # mean_list_11_set[iidx]
        bias_var_list_11 = bias_var_list_11_set_average  # bias_var_list_11_set[iidx]
        input_list_12 = input_list_12_set[iidx]
        mean_list_12 = mean_list_12_set_average  # mean_list_12_set[iidx]
        grad_bn12_11_input_0_list = grad_bn12_11_input_0_list_set[iidx]
        grad_bn12_11_input_1_list = grad_bn12_11_input_1_list_set_average  # grad_bn12_11_input_1_list_set[iidx]
        grad_bn12_12_input_0_list = grad_bn12_12_input_0_list_set_average  # grad_bn12_12_input_0_list_set[iidx]

        grad_bn_11_input_0 = torch.rand([channels_11, batchsize * height_11 * width_11])
        grad_bn_11_input_1 = torch.rand(channels_11)
        grad_bn_12_input_0 = torch.rand(channels_11)
        optimizer_11_2 = torch.optim.SGD([{'params': local_list_11_2[idx].parameters()}], args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        (grad_bn11_11_input_0_list,
         grad_bn11_11_input_1_list,
         grad_bn11_12_input_0_list) = train_backward_2(local_list_11_2[idx], optimizer_11_2,
                                                       input_list_11, mean_list_11, bias_var_list_11,
                                                       batchsize, channels_11, height_11, width_11,
                                                       grad_bn12_11_input_0_list, grad_bn12_11_input_1_list,
                                                       grad_bn12_12_input_0_list,
                                                       input_list_12, mean_list_12, device)

        grad_bn11_11_input_0_list_set.append(grad_bn11_11_input_0_list)
        # ==== average mean gradient
        if iidx == 0:
            grad_bn11_11_input_1_list_set_average = [torch.zeros_like(grad_bn11_11_input_1_list[0])]
        grad_bn11_11_input_1_list_set_average[0] += grad_bn11_11_input_1_list[0] / selected_node_number
        # ==== average variance gradient
        if iidx == 0:
            grad_bn11_12_input_0_list_set_average = [torch.zeros_like(grad_bn11_12_input_0_list[0])]
        grad_bn11_12_input_0_list_set_average[0] += grad_bn11_12_input_0_list[0] / selected_node_number
        iidx += 1

    del input_list_12_set, grad_bn12_11_input_0_list_set
    del input_list_11, mean_list_11, bias_var_list_11, input_list_12, mean_list_12
    del grad_bn12_11_input_0_list, grad_bn12_11_input_1_list, grad_bn12_12_input_0_list
    del grad_bn11_11_input_0_list, grad_bn11_11_input_1_list, grad_bn11_12_input_0_list

    grad_bn0_11_input_0_list_set = []
    iidx = 0
    for idx in selected_local_device_index:
        out0_list = out0_list_set[iidx]
        input_list_11 = input_list_11_set[iidx]
        mean_list_11 = mean_list_11_set_average  # mean_list_11_set[iidx]
        grad_bn11_11_input_0_list = grad_bn11_11_input_0_list_set[iidx]
        grad_bn11_11_input_1_list = grad_bn11_11_input_1_list_set_average  # grad_bn11_11_input_1_list_set[iidx]
        grad_bn11_12_input_0_list = grad_bn11_12_input_0_list_set_average  # grad_bn11_12_input_0_list_set[iidx]

        input_list_0 = input_list_0_set[iidx]
        mean_list_0 = mean_list_0_set_average  # mean_list_0_set[iidx]
        bias_var_list_0 = bias_var_list_0_set_average  # bias_var_list_0_set[iidx]
        grad_model_12_2_shortcut_input_0_list = grad_model_12_2_shortcut_input_0_list_set[iidx]

        # ---- train_backward_11_1
        grad_model_1_conv_input_0 = torch.rand([batchsize, channels_0, height_0, width_0])
        optimizer_11_1 = torch.optim.SGD([{'params': local_list_11_1[idx].parameters()}], args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        grad_model_11_1_conv_input_0_list = train_backward_1_Input(local_list_11_1[idx], optimizer_11_1, out0_list,
                                                                   grad_bn11_11_input_0_list, grad_bn11_11_input_1_list,
                                                                   grad_bn11_12_input_0_list,
                                                                   input_list_11, mean_list_11)

        # block 0
        # ---- train_backward_0_2
        grad_bn_11_input_0 = torch.rand([channels_0, batchsize * height_0 * width_0])
        grad_bn_11_input_1 = torch.rand(channels_0)
        grad_bn_12_input_0 = torch.rand(channels_0)
        optimizer_0_2 = torch.optim.SGD([{'params': local_list_0_2[idx].parameters()}], args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        (grad_bn0_11_input_0_list,
         grad_bn0_11_input_1_list,
         grad_bn0_12_input_0_list) = train_backward_3_ResOut(local_list_0_2[idx], optimizer_0_2,
                                                             input_list_0, mean_list_0, bias_var_list_0,
                                                             batchsize, channels_0, height_0, width_0,
                                                             grad_model_11_1_conv_input_0_list,
                                                             grad_model_12_2_shortcut_input_0_list, device)
        grad_bn0_11_input_0_list_set.append(grad_bn0_11_input_0_list)
        # ==== average mean gradient
        if iidx == 0:
            grad_bn0_11_input_1_list_set_average = [torch.zeros_like(grad_bn0_11_input_1_list[0])]
        grad_bn0_11_input_1_list_set_average[0] += grad_bn0_11_input_1_list[0] / selected_node_number
        # ==== average variance gradient
        if iidx == 0:
            grad_bn0_12_input_0_list_set_average = [torch.zeros_like(grad_bn0_12_input_0_list[0])]
        grad_bn0_12_input_0_list_set_average[0] += grad_bn0_12_input_0_list[0] / selected_node_number
        iidx += 1

    del out0_list_set, input_list_11_set, grad_bn11_11_input_0_list_set, grad_model_12_2_shortcut_input_0_list_set
    del out0_list, input_list_11, mean_list_11
    del grad_bn11_11_input_0_list, grad_bn11_11_input_1_list, grad_bn11_12_input_0_list
    del input_list_0, mean_list_0, bias_var_list_0,
    del grad_model_12_2_shortcut_input_0_list, grad_model_11_1_conv_input_0_list
    del grad_bn0_11_input_0_list, grad_bn0_11_input_1_list, grad_bn0_12_input_0_list

    iidx = 0
    for idx in selected_local_device_index:
        net_input_list = net_input_list_set[iidx]
        input_list_0 = input_list_0_set[iidx]
        mean_list_0 = mean_list_0_set_average  # mean_list_0_set[iidx]
        grad_bn0_11_input_0_list = grad_bn0_11_input_0_list_set[iidx]
        grad_bn0_11_input_1_list = grad_bn0_11_input_1_list_set_average  # grad_bn0_11_input_1_list_set[iidx]
        grad_bn0_12_input_0_list = grad_bn0_12_input_0_list_set_average  # grad_bn0_12_input_0_list_set[iidx]
        iidx += 1

        # ---- train_backward_0_1
        optimizer_0_1 = torch.optim.SGD([{'params': local_list_0_1[idx].parameters()}], args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        train_backward_1_woInput(local_list_0_1[idx],
                                 optimizer_0_1,
                                 net_input_list,
                                 grad_bn0_11_input_0_list, grad_bn0_11_input_1_list, grad_bn0_12_input_0_list,
                                 input_list_0, mean_list_0)

    del net_input_list_set, input_list_0_set, grad_bn0_11_input_0_list_set
    del net_input_list, input_list_0, mean_list_0
    del grad_bn0_11_input_0_list, grad_bn0_11_input_1_list, grad_bn0_12_input_0_list

    return local_list_0_1, local_list_0_2,\
           local_list_11_1, local_list_11_2, local_list_12_2,\
           local_list_13_1, local_list_13_2, local_list_14_2,\
           local_list_15_1, local_list_15_2, local_list_16_2,\
           local_list_21_1, local_list_21_2, local_list_22_2,\
           local_list_23_1, local_list_23_2, local_list_24_2,\
           local_list_25_1, local_list_25_2, local_list_26_2,\
           local_list_31_1, local_list_31_2, local_list_32_2,\
           local_list_33_1, local_list_33_2, local_list_34_2,\
           local_list_35_1, local_list_35_2, local_list_36_2,\
           local_list_4