import torch.utils.data
import copy
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.fedavg import average_weights # federated averaging algorithm
from libs.initial_models import initial_subnet_model

# global aggregation of local models (if local_update_step_number_total == 1)
def global_aggregation_FedSGD(epoch, node_number, selected_local_device_index, device, half_flag,
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
                              local_list_4):

    # initialize subnet models
    model_0_1, model_0_2, model_0_Mean, model_0_Var, \
    model_11_1, model_11_2, model_12_2, model_11_Mean, model_11_Var, model_12_Mean, model_12_Var, \
    model_13_1, model_13_2, model_14_2, model_13_Mean, model_13_Var, model_14_Mean, model_14_Var, \
    model_15_1, model_15_2, model_16_2, model_15_Mean, model_15_Var, model_16_Mean, model_16_Var, \
    model_21_1, model_21_2, model_22_2, model_21_Mean, model_21_Var, model_22_Mean, model_22_Var, \
    model_23_1, model_23_2, model_24_2, model_23_Mean, model_23_Var, model_24_Mean, model_24_Var, \
    model_25_1, model_25_2, model_26_2, model_25_Mean, model_25_Var, model_26_Mean, model_26_Var, \
    model_31_1, model_31_2, model_32_2, model_31_Mean, model_31_Var, model_32_Mean, model_32_Var, \
    model_33_1, model_33_2, model_34_2, model_33_Mean, model_33_Var, model_34_Mean, model_34_Var, \
    model_35_1, model_35_2, model_36_2, model_35_Mean, model_35_Var, model_36_Mean, model_36_Var, \
    model_4 = initial_subnet_model(device, half_flag)

    # ---- block 0
    local_weights_DGD_0_1, local_weights_DGD_0_2 = [], []
    # ---- block 1
    local_weights_DGD_11_1, local_weights_DGD_11_2, local_weights_DGD_12_2 = [], [], []
    local_weights_DGD_13_1, local_weights_DGD_13_2, local_weights_DGD_14_2 = [], [], []
    local_weights_DGD_15_1, local_weights_DGD_15_2, local_weights_DGD_16_2 = [], [], []
    # ---- block 2
    local_weights_DGD_21_1, local_weights_DGD_21_2, local_weights_DGD_22_2 = [], [], []
    local_weights_DGD_23_1, local_weights_DGD_23_2, local_weights_DGD_24_2 = [], [], []
    local_weights_DGD_25_1, local_weights_DGD_25_2, local_weights_DGD_26_2 = [], [], []
    # ---- block 3
    local_weights_DGD_31_1, local_weights_DGD_31_2, local_weights_DGD_32_2 = [], [], []
    local_weights_DGD_33_1, local_weights_DGD_33_2, local_weights_DGD_34_2 = [], [], []
    local_weights_DGD_35_1, local_weights_DGD_35_2, local_weights_DGD_36_2 = [], [], []
    # ---- Final
    local_weights_DGD_4 = []

    for idx in selected_local_device_index:
        # ---- block 0
        local_weights_DGD_0_1.append(copy.deepcopy(local_list_0_1[idx].state_dict()))
        local_weights_DGD_0_2.append(copy.deepcopy(local_list_0_2[idx].state_dict()))
        # ---- block 1
        local_weights_DGD_11_1.append(copy.deepcopy(local_list_11_1[idx].state_dict()))
        local_weights_DGD_11_2.append(copy.deepcopy(local_list_11_2[idx].state_dict()))
        local_weights_DGD_12_2.append(copy.deepcopy(local_list_12_2[idx].state_dict()))
        local_weights_DGD_13_1.append(copy.deepcopy(local_list_13_1[idx].state_dict()))
        local_weights_DGD_13_2.append(copy.deepcopy(local_list_13_2[idx].state_dict()))
        local_weights_DGD_14_2.append(copy.deepcopy(local_list_14_2[idx].state_dict()))
        local_weights_DGD_15_1.append(copy.deepcopy(local_list_15_1[idx].state_dict()))
        local_weights_DGD_15_2.append(copy.deepcopy(local_list_15_2[idx].state_dict()))
        local_weights_DGD_16_2.append(copy.deepcopy(local_list_16_2[idx].state_dict()))
        # ---- block 2
        local_weights_DGD_21_1.append(copy.deepcopy(local_list_21_1[idx].state_dict()))
        local_weights_DGD_21_2.append(copy.deepcopy(local_list_21_2[idx].state_dict()))
        local_weights_DGD_22_2.append(copy.deepcopy(local_list_22_2[idx].state_dict()))
        local_weights_DGD_23_1.append(copy.deepcopy(local_list_23_1[idx].state_dict()))
        local_weights_DGD_23_2.append(copy.deepcopy(local_list_23_2[idx].state_dict()))
        local_weights_DGD_24_2.append(copy.deepcopy(local_list_24_2[idx].state_dict()))
        local_weights_DGD_25_1.append(copy.deepcopy(local_list_25_1[idx].state_dict()))
        local_weights_DGD_25_2.append(copy.deepcopy(local_list_25_2[idx].state_dict()))
        local_weights_DGD_26_2.append(copy.deepcopy(local_list_26_2[idx].state_dict()))
        # ---- block 3
        local_weights_DGD_31_1.append(copy.deepcopy(local_list_31_1[idx].state_dict()))
        local_weights_DGD_31_2.append(copy.deepcopy(local_list_31_2[idx].state_dict()))
        local_weights_DGD_32_2.append(copy.deepcopy(local_list_32_2[idx].state_dict()))
        local_weights_DGD_33_1.append(copy.deepcopy(local_list_33_1[idx].state_dict()))
        local_weights_DGD_33_2.append(copy.deepcopy(local_list_33_2[idx].state_dict()))
        local_weights_DGD_34_2.append(copy.deepcopy(local_list_34_2[idx].state_dict()))
        local_weights_DGD_35_1.append(copy.deepcopy(local_list_35_1[idx].state_dict()))
        local_weights_DGD_35_2.append(copy.deepcopy(local_list_35_2[idx].state_dict()))
        local_weights_DGD_36_2.append(copy.deepcopy(local_list_36_2[idx].state_dict()))
        # ---- Final
        local_weights_DGD_4.append(copy.deepcopy(local_list_4[idx].state_dict()))

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

    # Average
    # ---- block 0
    global_average_weight_0_1 = average_weights(local_weights_DGD_0_1)
    global_average_weight_0_2 = average_weights(local_weights_DGD_0_2)
    # ---- block 1
    global_average_weight_11_1 = average_weights(local_weights_DGD_11_1)
    global_average_weight_11_2 = average_weights(local_weights_DGD_11_2)
    global_average_weight_12_2 = average_weights(local_weights_DGD_12_2)
    global_average_weight_13_1 = average_weights(local_weights_DGD_13_1)
    global_average_weight_13_2 = average_weights(local_weights_DGD_13_2)
    global_average_weight_14_2 = average_weights(local_weights_DGD_14_2)
    global_average_weight_15_1 = average_weights(local_weights_DGD_15_1)
    global_average_weight_15_2 = average_weights(local_weights_DGD_15_2)
    global_average_weight_16_2 = average_weights(local_weights_DGD_16_2)
    # ---- block 2
    global_average_weight_21_1 = average_weights(local_weights_DGD_21_1)
    global_average_weight_21_2 = average_weights(local_weights_DGD_21_2)
    global_average_weight_22_2 = average_weights(local_weights_DGD_22_2)
    global_average_weight_23_1 = average_weights(local_weights_DGD_23_1)
    global_average_weight_23_2 = average_weights(local_weights_DGD_23_2)
    global_average_weight_24_2 = average_weights(local_weights_DGD_24_2)
    global_average_weight_25_1 = average_weights(local_weights_DGD_25_1)
    global_average_weight_25_2 = average_weights(local_weights_DGD_25_2)
    global_average_weight_26_2 = average_weights(local_weights_DGD_26_2)
    # ---- block 3
    global_average_weight_31_1 = average_weights(local_weights_DGD_31_1)
    global_average_weight_31_2 = average_weights(local_weights_DGD_31_2)
    global_average_weight_32_2 = average_weights(local_weights_DGD_32_2)
    global_average_weight_33_1 = average_weights(local_weights_DGD_33_1)
    global_average_weight_33_2 = average_weights(local_weights_DGD_33_2)
    global_average_weight_34_2 = average_weights(local_weights_DGD_34_2)
    global_average_weight_35_1 = average_weights(local_weights_DGD_35_1)
    global_average_weight_35_2 = average_weights(local_weights_DGD_35_2)
    global_average_weight_36_2 = average_weights(local_weights_DGD_36_2)
    # ---- Final
    global_average_weight_4 = average_weights(local_weights_DGD_4)

    # ---- Initial
    del local_weights_DGD_0_1, local_weights_DGD_0_2
    # ---- block 1
    del local_weights_DGD_11_1, local_weights_DGD_11_2, local_weights_DGD_12_2
    del local_weights_DGD_13_1, local_weights_DGD_13_2, local_weights_DGD_14_2
    del local_weights_DGD_15_1, local_weights_DGD_15_2, local_weights_DGD_16_2
    # ---- block 2
    del local_weights_DGD_21_1, local_weights_DGD_21_2, local_weights_DGD_22_2
    del local_weights_DGD_23_1, local_weights_DGD_23_2, local_weights_DGD_24_2
    del local_weights_DGD_25_1, local_weights_DGD_25_2, local_weights_DGD_26_2
    # ---- block 3
    del local_weights_DGD_31_1, local_weights_DGD_31_2, local_weights_DGD_32_2
    del local_weights_DGD_33_1, local_weights_DGD_33_2, local_weights_DGD_34_2
    del local_weights_DGD_35_1, local_weights_DGD_35_2, local_weights_DGD_36_2
    # ---- Final
    del local_weights_DGD_4

    # updating local model parameters
    # ---- block 0
    local_list_0_1, local_list_0_2 = [[]] * node_number, [[]] * node_number
    local_list_0_Mean, local_list_0_Var = [[]] * node_number, [[]] * node_number
    # ---- block 1
    local_list_11_1, local_list_11_2, local_list_12_2 = [[]] * node_number, [[]] * node_number, [[]] * node_number
    local_list_11_Mean, local_list_11_Var = [[]] * node_number, [[]] * node_number
    local_list_12_Mean, local_list_12_Var = [[]] * node_number, [[]] * node_number
    local_list_13_1, local_list_13_2, local_list_14_2 = [[]] * node_number, [[]] * node_number, [[]] * node_number
    local_list_13_Mean, local_list_13_Var = [[]] * node_number, [[]] * node_number
    local_list_14_Mean, local_list_14_Var = [[]] * node_number, [[]] * node_number
    local_list_15_1, local_list_15_2, local_list_16_2 = [[]] * node_number, [[]] * node_number, [[]] * node_number
    local_list_15_Mean, local_list_15_Var = [[]] * node_number, [[]] * node_number
    local_list_16_Mean, local_list_16_Var = [[]] * node_number, [[]] * node_number
    # ---- block 2
    local_list_21_1, local_list_21_2, local_list_22_2 = [[]] * node_number, [[]] * node_number, [[]] * node_number
    local_list_21_Mean, local_list_21_Var = [[]] * node_number, [[]] * node_number
    local_list_22_Mean, local_list_22_Var = [[]] * node_number, [[]] * node_number
    local_list_23_1, local_list_23_2, local_list_24_2 = [[]] * node_number, [[]] * node_number, [[]] * node_number
    local_list_23_Mean, local_list_23_Var = [[]] * node_number, [[]] * node_number
    local_list_24_Mean, local_list_24_Var = [[]] * node_number, [[]] * node_number
    local_list_25_1, local_list_25_2, local_list_26_2 = [[]] * node_number, [[]] * node_number, [[]] * node_number
    local_list_25_Mean, local_list_25_Var = [[]] * node_number, [[]] * node_number
    local_list_26_Mean, local_list_26_Var = [[]] * node_number, [[]] * node_number
    # ---- block 3
    local_list_31_1, local_list_31_2, local_list_32_2 = [[]] * node_number, [[]] * node_number, [[]] * node_number
    local_list_31_Mean, local_list_31_Var = [[]] * node_number, [[]] * node_number
    local_list_32_Mean, local_list_32_Var = [[]] * node_number, [[]] * node_number
    local_list_33_1, local_list_33_2, local_list_34_2 = [[]] * node_number, [[]] * node_number, [[]] * node_number
    local_list_33_Mean, local_list_33_Var = [[]] * node_number, [[]] * node_number
    local_list_34_Mean, local_list_34_Var = [[]] * node_number, [[]] * node_number
    local_list_35_1, local_list_35_2, local_list_36_2 = [[]] * node_number, [[]] * node_number, [[]] * node_number
    local_list_35_Mean, local_list_35_Var = [[]] * node_number, [[]] * node_number
    local_list_36_Mean, local_list_36_Var = [[]] * node_number, [[]] * node_number
    # ---- Final
    local_list_4 = [[]] * node_number

    for i in range(node_number):
        # ---- Block 0
        local_list_0_1[i], local_list_0_2[i] = copy.deepcopy(model_0_1), copy.deepcopy(model_0_2)
        local_list_0_Mean[i], local_list_0_Var[i] = copy.deepcopy(model_0_Mean), copy.deepcopy(model_0_Var)
        # ---- block 1
        local_list_11_1[i], local_list_11_2[i] = copy.deepcopy(model_11_1), copy.deepcopy(model_11_2)
        local_list_12_2[i] = copy.deepcopy(model_12_2)
        local_list_11_Mean[i], local_list_11_Var[i] = copy.deepcopy(model_11_Mean), copy.deepcopy(model_11_Var)
        local_list_12_Mean[i], local_list_12_Var[i] = copy.deepcopy(model_12_Mean), copy.deepcopy(model_12_Var)

        local_list_13_1[i], local_list_13_2[i] = copy.deepcopy(model_13_1), copy.deepcopy(model_13_2)
        local_list_14_2[i] = copy.deepcopy(model_14_2)
        local_list_13_Mean[i], local_list_13_Var[i] = copy.deepcopy(model_13_Mean), copy.deepcopy(model_13_Var)
        local_list_14_Mean[i], local_list_14_Var[i] = copy.deepcopy(model_14_Mean), copy.deepcopy(model_14_Var)

        local_list_15_1[i], local_list_15_2[i] = copy.deepcopy(model_15_1), copy.deepcopy(model_15_2)
        local_list_16_2[i] = copy.deepcopy(model_16_2)
        local_list_15_Mean[i], local_list_15_Var[i] = copy.deepcopy(model_15_Mean), copy.deepcopy(model_15_Var)
        local_list_16_Mean[i], local_list_16_Var[i] = copy.deepcopy(model_16_Mean), copy.deepcopy(model_16_Var)
        # ---- block 2
        local_list_21_1[i], local_list_21_2[i] = copy.deepcopy(model_21_1), copy.deepcopy(model_21_2)
        local_list_22_2[i] = copy.deepcopy(model_22_2)
        local_list_21_Mean[i], local_list_21_Var[i] = copy.deepcopy(model_21_Mean), copy.deepcopy(model_21_Var)
        local_list_22_Mean[i], local_list_22_Var[i] = copy.deepcopy(model_22_Mean), copy.deepcopy(model_22_Var)

        local_list_23_1[i], local_list_23_2[i] = copy.deepcopy(model_23_1), copy.deepcopy(model_23_2)
        local_list_24_2[i] = copy.deepcopy(model_24_2)
        local_list_23_Mean[i], local_list_23_Var[i] = copy.deepcopy(model_23_Mean), copy.deepcopy(model_23_Var)
        local_list_24_Mean[i], local_list_24_Var[i] = copy.deepcopy(model_24_Mean), copy.deepcopy(model_24_Var)

        local_list_25_1[i], local_list_25_2[i] = copy.deepcopy(model_25_1), copy.deepcopy(model_25_2)
        local_list_26_2[i] = copy.deepcopy(model_26_2)
        local_list_25_Mean[i], local_list_25_Var[i] = copy.deepcopy(model_25_Mean), copy.deepcopy(model_25_Var)
        local_list_26_Mean[i], local_list_26_Var[i] = copy.deepcopy(model_26_Mean), copy.deepcopy(model_26_Var)
        # ---- block 3
        local_list_31_1[i], local_list_31_2[i] = copy.deepcopy(model_31_1), copy.deepcopy(model_31_2)
        local_list_32_2[i] = copy.deepcopy(model_32_2)
        local_list_31_Mean[i], local_list_31_Var[i] = copy.deepcopy(model_31_Mean), copy.deepcopy(model_31_Var)
        local_list_32_Mean[i], local_list_32_Var[i] = copy.deepcopy(model_32_Mean), copy.deepcopy(model_32_Var)

        local_list_33_1[i], local_list_33_2[i] = copy.deepcopy(model_33_1), copy.deepcopy(model_33_2)
        local_list_34_2[i] = copy.deepcopy(model_34_2)
        local_list_33_Mean[i], local_list_33_Var[i] = copy.deepcopy(model_33_Mean), copy.deepcopy(model_33_Var)
        local_list_34_Mean[i], local_list_34_Var[i] = copy.deepcopy(model_34_Mean), copy.deepcopy(model_34_Var)

        local_list_35_1[i], local_list_35_2[i] = copy.deepcopy(model_35_1), copy.deepcopy(model_35_2)
        local_list_36_2[i] = copy.deepcopy(model_36_2)
        local_list_35_Mean[i], local_list_35_Var[i] = copy.deepcopy(model_35_Mean), copy.deepcopy(model_35_Var)
        local_list_36_Mean[i], local_list_36_Var[i] = copy.deepcopy(model_36_Mean), copy.deepcopy(model_36_Var)
        # ---- Final
        local_list_4[i] = copy.deepcopy(model_4)

        # ---- Initial
        local_list_0_1[i].load_state_dict(global_average_weight_0_1)
        local_list_0_2[i].load_state_dict(global_average_weight_0_2)
        # ---- block 1
        local_list_11_1[i].load_state_dict(global_average_weight_11_1)
        local_list_11_2[i].load_state_dict(global_average_weight_11_2)
        local_list_12_2[i].load_state_dict(global_average_weight_12_2)

        local_list_13_1[i].load_state_dict(global_average_weight_13_1)
        local_list_13_2[i].load_state_dict(global_average_weight_13_2)
        local_list_14_2[i].load_state_dict(global_average_weight_14_2)

        local_list_15_1[i].load_state_dict(global_average_weight_15_1)
        local_list_15_2[i].load_state_dict(global_average_weight_15_2)
        local_list_16_2[i].load_state_dict(global_average_weight_16_2)
        # ---- block 2
        local_list_21_1[i].load_state_dict(global_average_weight_21_1)
        local_list_21_2[i].load_state_dict(global_average_weight_21_2)
        local_list_22_2[i].load_state_dict(global_average_weight_22_2)

        local_list_23_1[i].load_state_dict(global_average_weight_23_1)
        local_list_23_2[i].load_state_dict(global_average_weight_23_2)
        local_list_24_2[i].load_state_dict(global_average_weight_24_2)

        local_list_25_1[i].load_state_dict(global_average_weight_25_1)
        local_list_25_2[i].load_state_dict(global_average_weight_25_2)
        local_list_26_2[i].load_state_dict(global_average_weight_26_2)
        # ---- block 3
        local_list_31_1[i].load_state_dict(global_average_weight_31_1)
        local_list_31_2[i].load_state_dict(global_average_weight_31_2)
        local_list_32_2[i].load_state_dict(global_average_weight_32_2)

        local_list_33_1[i].load_state_dict(global_average_weight_33_1)
        local_list_33_2[i].load_state_dict(global_average_weight_33_2)
        local_list_34_2[i].load_state_dict(global_average_weight_34_2)

        local_list_35_1[i].load_state_dict(global_average_weight_35_1)
        local_list_35_2[i].load_state_dict(global_average_weight_35_2)
        local_list_36_2[i].load_state_dict(global_average_weight_36_2)
        # ---- Final
        local_list_4[i].load_state_dict(global_average_weight_4)

    # ---- Block 0
    global_average_weight['conv0.weight'] = global_average_weight_0_1['conv.weight']
    global_average_weight['bn0.weight'] = global_average_weight_0_2['bn_2.weight']
    global_average_weight['bn0.bias'] = global_average_weight_0_2['bn_2.bias']
    global_average_weight['bn0.running_mean'] = global_average_weight_0_2['bn_11.running_mean']
    global_average_weight['bn0.running_var'] = global_average_weight_0_2['bn_12.running_var']
    global_average_weight['bn0.num_batches_tracked'] = torch.tensor(epoch + 1)

    # ---- Block 1
    global_average_weight['conv11.weight'] = global_average_weight_11_1['conv.weight']
    global_average_weight['bn11.weight'] = global_average_weight_11_2['bn_2.weight']
    global_average_weight['bn11.bias'] = global_average_weight_11_2['bn_2.bias']
    global_average_weight['bn11.running_mean'] = global_average_weight_11_2['bn_11.running_mean']
    global_average_weight['bn11.running_var'] = global_average_weight_11_2['bn_12.running_var']
    global_average_weight['bn11.num_batches_tracked'] = torch.tensor(epoch + 1)
    global_average_weight['conv12.weight'] = global_average_weight_11_2['conv.weight']
    global_average_weight['bn12.weight'] = global_average_weight_12_2['bn_2.weight']
    global_average_weight['bn12.bias'] = global_average_weight_12_2['bn_2.bias']
    global_average_weight['bn12.running_mean'] = global_average_weight_12_2['bn_11.running_mean']
    global_average_weight['bn12.running_var'] = global_average_weight_12_2['bn_12.running_var']
    global_average_weight['bn12.num_batches_tracked'] = torch.tensor(epoch + 1)

    global_average_weight['conv13.weight'] = global_average_weight_13_1['conv.weight']
    global_average_weight['bn13.weight'] = global_average_weight_13_2['bn_2.weight']
    global_average_weight['bn13.bias'] = global_average_weight_13_2['bn_2.bias']
    global_average_weight['bn13.running_mean'] = global_average_weight_13_2['bn_11.running_mean']
    global_average_weight['bn13.running_var'] = global_average_weight_13_2['bn_12.running_var']
    global_average_weight['bn13.num_batches_tracked'] = torch.tensor(epoch + 1)
    global_average_weight['conv14.weight'] = global_average_weight_13_2['conv.weight']
    global_average_weight['bn14.weight'] = global_average_weight_14_2['bn_2.weight']
    global_average_weight['bn14.bias'] = global_average_weight_14_2['bn_2.bias']
    global_average_weight['bn14.running_mean'] = global_average_weight_14_2['bn_11.running_mean']
    global_average_weight['bn14.running_var'] = global_average_weight_14_2['bn_12.running_var']
    global_average_weight['bn14.num_batches_tracked'] = torch.tensor(epoch + 1)

    global_average_weight['conv15.weight'] = global_average_weight_15_1['conv.weight']
    global_average_weight['bn15.weight'] = global_average_weight_15_2['bn_2.weight']
    global_average_weight['bn15.bias'] = global_average_weight_15_2['bn_2.bias']
    global_average_weight['bn15.running_mean'] = global_average_weight_15_2['bn_11.running_mean']
    global_average_weight['bn15.running_var'] = global_average_weight_15_2['bn_12.running_var']
    global_average_weight['bn15.num_batches_tracked'] = torch.tensor(epoch + 1)
    global_average_weight['conv16.weight'] = global_average_weight_15_2['conv.weight']
    global_average_weight['bn16.weight'] = global_average_weight_16_2['bn_2.weight']
    global_average_weight['bn16.bias'] = global_average_weight_16_2['bn_2.bias']
    global_average_weight['bn16.running_mean'] = global_average_weight_16_2['bn_11.running_mean']
    global_average_weight['bn16.running_var'] = global_average_weight_16_2['bn_12.running_var']
    global_average_weight['bn16.num_batches_tracked'] = torch.tensor(epoch + 1)

    # ---- Block 2
    global_average_weight['conv21.weight'] = global_average_weight_21_1['conv.weight']
    global_average_weight['bn21.weight'] = global_average_weight_21_2['bn_2.weight']
    global_average_weight['bn21.bias'] = global_average_weight_21_2['bn_2.bias']
    global_average_weight['bn21.running_mean'] = global_average_weight_21_2['bn_11.running_mean']
    global_average_weight['bn21.running_var'] = global_average_weight_21_2['bn_12.running_var']
    global_average_weight['bn21.num_batches_tracked'] = torch.tensor(epoch + 1)
    global_average_weight['conv22.weight'] = global_average_weight_21_2['conv.weight']
    global_average_weight['bn22.weight'] = global_average_weight_22_2['bn_2.weight']
    global_average_weight['bn22.bias'] = global_average_weight_22_2['bn_2.bias']
    global_average_weight['bn22.running_mean'] = global_average_weight_22_2['bn_11.running_mean']
    global_average_weight['bn22.running_var'] = global_average_weight_22_2['bn_12.running_var']
    global_average_weight['bn22.num_batches_tracked'] = torch.tensor(epoch + 1)

    global_average_weight['conv23.weight'] = global_average_weight_23_1['conv.weight']
    global_average_weight['bn23.weight'] = global_average_weight_23_2['bn_2.weight']
    global_average_weight['bn23.bias'] = global_average_weight_23_2['bn_2.bias']
    global_average_weight['bn23.running_mean'] = global_average_weight_23_2['bn_11.running_mean']
    global_average_weight['bn23.running_var'] = global_average_weight_23_2['bn_12.running_var']
    global_average_weight['bn23.num_batches_tracked'] = torch.tensor(epoch + 1)
    global_average_weight['conv24.weight'] = global_average_weight_23_2['conv.weight']
    global_average_weight['bn24.weight'] = global_average_weight_24_2['bn_2.weight']
    global_average_weight['bn24.bias'] = global_average_weight_24_2['bn_2.bias']
    global_average_weight['bn24.running_mean'] = global_average_weight_24_2['bn_11.running_mean']
    global_average_weight['bn24.running_var'] = global_average_weight_24_2['bn_12.running_var']
    global_average_weight['bn24.num_batches_tracked'] = torch.tensor(epoch + 1)

    global_average_weight['conv25.weight'] = global_average_weight_25_1['conv.weight']
    global_average_weight['bn25.weight'] = global_average_weight_25_2['bn_2.weight']
    global_average_weight['bn25.bias'] = global_average_weight_25_2['bn_2.bias']
    global_average_weight['bn25.running_mean'] = global_average_weight_25_2['bn_11.running_mean']
    global_average_weight['bn25.running_var'] = global_average_weight_25_2['bn_12.running_var']
    global_average_weight['bn25.num_batches_tracked'] = torch.tensor(epoch + 1)
    global_average_weight['conv26.weight'] = global_average_weight_25_2['conv.weight']
    global_average_weight['bn26.weight'] = global_average_weight_26_2['bn_2.weight']
    global_average_weight['bn26.bias'] = global_average_weight_26_2['bn_2.bias']
    global_average_weight['bn26.running_mean'] = global_average_weight_26_2['bn_11.running_mean']
    global_average_weight['bn26.running_var'] = global_average_weight_26_2['bn_12.running_var']
    global_average_weight['bn26.num_batches_tracked'] = torch.tensor(epoch + 1)

    # ---- Block 3
    global_average_weight['conv31.weight'] = global_average_weight_31_1['conv.weight']
    global_average_weight['bn31.weight'] = global_average_weight_31_2['bn_2.weight']
    global_average_weight['bn31.bias'] = global_average_weight_31_2['bn_2.bias']
    global_average_weight['bn31.running_mean'] = global_average_weight_31_2['bn_11.running_mean']
    global_average_weight['bn31.running_var'] = global_average_weight_31_2['bn_12.running_var']
    global_average_weight['bn31.num_batches_tracked'] = torch.tensor(epoch + 1)
    global_average_weight['conv32.weight'] = global_average_weight_31_2['conv.weight']
    global_average_weight['bn32.weight'] = global_average_weight_32_2['bn_2.weight']
    global_average_weight['bn32.bias'] = global_average_weight_32_2['bn_2.bias']
    global_average_weight['bn32.running_mean'] = global_average_weight_32_2['bn_11.running_mean']
    global_average_weight['bn32.running_var'] = global_average_weight_32_2['bn_12.running_var']
    global_average_weight['bn32.num_batches_tracked'] = torch.tensor(epoch + 1)

    global_average_weight['conv33.weight'] = global_average_weight_33_1['conv.weight']
    global_average_weight['bn33.weight'] = global_average_weight_33_2['bn_2.weight']
    global_average_weight['bn33.bias'] = global_average_weight_33_2['bn_2.bias']
    global_average_weight['bn33.running_mean'] = global_average_weight_33_2['bn_11.running_mean']
    global_average_weight['bn33.running_var'] = global_average_weight_33_2['bn_12.running_var']
    global_average_weight['bn33.num_batches_tracked'] = torch.tensor(epoch + 1)
    global_average_weight['conv34.weight'] = global_average_weight_33_2['conv.weight']
    global_average_weight['bn34.weight'] = global_average_weight_34_2['bn_2.weight']
    global_average_weight['bn34.bias'] = global_average_weight_34_2['bn_2.bias']
    global_average_weight['bn34.running_mean'] = global_average_weight_34_2['bn_11.running_mean']
    global_average_weight['bn34.running_var'] = global_average_weight_34_2['bn_12.running_var']
    global_average_weight['bn34.num_batches_tracked'] = torch.tensor(epoch + 1)

    global_average_weight['conv35.weight'] = global_average_weight_35_1['conv.weight']
    global_average_weight['bn35.weight'] = global_average_weight_35_2['bn_2.weight']
    global_average_weight['bn35.bias'] = global_average_weight_35_2['bn_2.bias']
    global_average_weight['bn35.running_mean'] = global_average_weight_35_2['bn_11.running_mean']
    global_average_weight['bn35.running_var'] = global_average_weight_35_2['bn_12.running_var']
    global_average_weight['bn35.num_batches_tracked'] = torch.tensor(epoch + 1)
    global_average_weight['conv36.weight'] = global_average_weight_35_2['conv.weight']
    global_average_weight['bn36.weight'] = global_average_weight_36_2['bn_2.weight']
    global_average_weight['bn36.bias'] = global_average_weight_36_2['bn_2.bias']
    global_average_weight['bn36.running_mean'] = global_average_weight_36_2['bn_11.running_mean']
    global_average_weight['bn36.running_var'] = global_average_weight_36_2['bn_12.running_var']
    global_average_weight['bn36.num_batches_tracked'] = torch.tensor(epoch + 1)
    # ---- Final
    global_average_weight['linear.weight'] = global_average_weight_4['linear.weight']
    global_average_weight['linear.bias'] = global_average_weight_4['linear.bias']

    # ---- Block 0
    del global_average_weight_0_1, global_average_weight_0_2
    # ---- Block 1
    del global_average_weight_11_1, global_average_weight_11_2, global_average_weight_12_2
    del global_average_weight_13_1, global_average_weight_13_2, global_average_weight_14_2
    del global_average_weight_15_1, global_average_weight_15_2, global_average_weight_16_2
    # ---- Block 2
    del global_average_weight_21_1, global_average_weight_21_2, global_average_weight_22_2
    del global_average_weight_23_1, global_average_weight_23_2, global_average_weight_24_2
    del global_average_weight_25_1, global_average_weight_25_2, global_average_weight_26_2
    # ---- Block 3
    del global_average_weight_31_1, global_average_weight_31_2, global_average_weight_32_2
    del global_average_weight_33_1, global_average_weight_33_2, global_average_weight_34_2
    del global_average_weight_35_1, global_average_weight_35_2, global_average_weight_36_2
    # ---- Final
    del global_average_weight_4

    global_average.load_state_dict(global_average_weight)
    
    return global_average, global_average_weight, \
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
            local_list_4