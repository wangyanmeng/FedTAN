import torch.utils.data
import copy
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from nets.ResNet20 import ResNet
from libs.initial_models import initial_subnet_model

# transform the representation of local models

# transform subnet form to whole network form
def subnet_2_whole(selected_local_device_index, epoch, node_number, device, half_flag,
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

    # initialize local models in clients (in whole network form)
    model = ResNet().to(device)
    if half_flag:
        model = model.half()
    local_list = [[]] * node_number
    for i in range(node_number):
        local_list[i] = copy.deepcopy(model)

    for idx in selected_local_device_index:
        # ---- block 0
        local_list[idx].conv0.weight.data = local_weights_DGD_0_1[idx]['conv.weight']
        local_list[idx].bn0.weight.data = local_weights_DGD_0_2[idx]['bn_2.weight']
        local_list[idx].bn0.bias.data = local_weights_DGD_0_2[idx]['bn_2.bias']
        local_list[idx].bn0.running_mean.data = local_weights_DGD_0_2[idx]['bn_11.running_mean']
        local_list[idx].bn0.running_var.data = local_weights_DGD_0_2[idx]['bn_12.running_var']
        local_list[idx].bn0.num_batches_tracked.data = torch.tensor(epoch + 1)
        # ---- Block 1
        local_list[idx].conv11.weight.data = local_weights_DGD_11_1[idx]['conv.weight']
        local_list[idx].bn11.weight.data = local_weights_DGD_11_2[idx]['bn_2.weight']
        local_list[idx].bn11.bias.data = local_weights_DGD_11_2[idx]['bn_2.bias']
        local_list[idx].bn11.running_mean.data = local_weights_DGD_11_2[idx]['bn_11.running_mean']
        local_list[idx].bn11.running_var.data = local_weights_DGD_11_2[idx]['bn_12.running_var']
        local_list[idx].bn11.num_batches_tracked.data = torch.tensor(epoch + 1)
        local_list[idx].conv12.weight.data = local_weights_DGD_11_2[idx]['conv.weight']
        local_list[idx].bn12.weight.data = local_weights_DGD_12_2[idx]['bn_2.weight']
        local_list[idx].bn12.bias.data = local_weights_DGD_12_2[idx]['bn_2.bias']
        local_list[idx].bn12.running_mean.data = local_weights_DGD_12_2[idx]['bn_11.running_mean']
        local_list[idx].bn12.running_var.data = local_weights_DGD_12_2[idx]['bn_12.running_var']
        local_list[idx].bn12.num_batches_tracked.data = torch.tensor(epoch + 1)

        local_list[idx].conv13.weight.data = local_weights_DGD_13_1[idx]['conv.weight']
        local_list[idx].bn13.weight.data = local_weights_DGD_13_2[idx]['bn_2.weight']
        local_list[idx].bn13.bias.data = local_weights_DGD_13_2[idx]['bn_2.bias']
        local_list[idx].bn13.running_mean.data = local_weights_DGD_13_2[idx]['bn_11.running_mean']
        local_list[idx].bn13.running_var.data = local_weights_DGD_13_2[idx]['bn_12.running_var']
        local_list[idx].bn13.num_batches_tracked.data = torch.tensor(epoch + 1)
        local_list[idx].conv14.weight.data = local_weights_DGD_13_2[idx]['conv.weight']
        local_list[idx].bn14.weight.data = local_weights_DGD_14_2[idx]['bn_2.weight']
        local_list[idx].bn14.bias.data = local_weights_DGD_14_2[idx]['bn_2.bias']
        local_list[idx].bn14.running_mean.data = local_weights_DGD_14_2[idx]['bn_11.running_mean']
        local_list[idx].bn14.running_var.data = local_weights_DGD_14_2[idx]['bn_12.running_var']
        local_list[idx].bn14.num_batches_tracked.data = torch.tensor(epoch + 1)

        local_list[idx].conv15.weight.data = local_weights_DGD_15_1[idx]['conv.weight']
        local_list[idx].bn15.weight.data = local_weights_DGD_15_2[idx]['bn_2.weight']
        local_list[idx].bn15.bias.data = local_weights_DGD_15_2[idx]['bn_2.bias']
        local_list[idx].bn15.running_mean.data = local_weights_DGD_15_2[idx]['bn_11.running_mean']
        local_list[idx].bn15.running_var.data = local_weights_DGD_15_2[idx]['bn_12.running_var']
        local_list[idx].bn15.num_batches_tracked.data = torch.tensor(epoch + 1)
        local_list[idx].conv16.weight.data = local_weights_DGD_15_2[idx]['conv.weight']
        local_list[idx].bn16.weight.data = local_weights_DGD_16_2[idx]['bn_2.weight']
        local_list[idx].bn16.bias.data = local_weights_DGD_16_2[idx]['bn_2.bias']
        local_list[idx].bn16.running_mean.data = local_weights_DGD_16_2[idx]['bn_11.running_mean']
        local_list[idx].bn16.running_var.data = local_weights_DGD_16_2[idx]['bn_12.running_var']
        local_list[idx].bn16.num_batches_tracked.data = torch.tensor(epoch + 1)
        # ---- Block 2
        local_list[idx].conv21.weight.data = local_weights_DGD_21_1[idx]['conv.weight']
        local_list[idx].bn21.weight.data = local_weights_DGD_21_2[idx]['bn_2.weight']
        local_list[idx].bn21.bias.data = local_weights_DGD_21_2[idx]['bn_2.bias']
        local_list[idx].bn21.running_mean.data = local_weights_DGD_21_2[idx]['bn_11.running_mean']
        local_list[idx].bn21.running_var.data = local_weights_DGD_21_2[idx]['bn_12.running_var']
        local_list[idx].bn21.num_batches_tracked.data = torch.tensor(epoch + 1)
        local_list[idx].conv22.weight.data = local_weights_DGD_21_2[idx]['conv.weight']
        local_list[idx].bn22.weight.data = local_weights_DGD_22_2[idx]['bn_2.weight']
        local_list[idx].bn22.bias.data = local_weights_DGD_22_2[idx]['bn_2.bias']
        local_list[idx].bn22.running_mean.data = local_weights_DGD_22_2[idx]['bn_11.running_mean']
        local_list[idx].bn22.running_var.data = local_weights_DGD_22_2[idx]['bn_12.running_var']
        local_list[idx].bn22.num_batches_tracked.data = torch.tensor(epoch + 1)

        local_list[idx].conv23.weight.data = local_weights_DGD_23_1[idx]['conv.weight']
        local_list[idx].bn23.weight.data = local_weights_DGD_23_2[idx]['bn_2.weight']
        local_list[idx].bn23.bias.data = local_weights_DGD_23_2[idx]['bn_2.bias']
        local_list[idx].bn23.running_mean.data = local_weights_DGD_23_2[idx]['bn_11.running_mean']
        local_list[idx].bn23.running_var.data = local_weights_DGD_23_2[idx]['bn_12.running_var']
        local_list[idx].bn23.num_batches_tracked.data = torch.tensor(epoch + 1)
        local_list[idx].conv24.weight.data = local_weights_DGD_23_2[idx]['conv.weight']
        local_list[idx].bn24.weight.data = local_weights_DGD_24_2[idx]['bn_2.weight']
        local_list[idx].bn24.bias.data = local_weights_DGD_24_2[idx]['bn_2.bias']
        local_list[idx].bn24.running_mean.data = local_weights_DGD_24_2[idx]['bn_11.running_mean']
        local_list[idx].bn24.running_var.data = local_weights_DGD_24_2[idx]['bn_12.running_var']
        local_list[idx].bn24.num_batches_tracked.data = torch.tensor(epoch + 1)

        local_list[idx].conv25.weight.data = local_weights_DGD_25_1[idx]['conv.weight']
        local_list[idx].bn25.weight.data = local_weights_DGD_25_2[idx]['bn_2.weight']
        local_list[idx].bn25.bias.data = local_weights_DGD_25_2[idx]['bn_2.bias']
        local_list[idx].bn25.running_mean.data = local_weights_DGD_25_2[idx]['bn_11.running_mean']
        local_list[idx].bn25.running_var.data = local_weights_DGD_25_2[idx]['bn_12.running_var']
        local_list[idx].bn25.num_batches_tracked.data = torch.tensor(epoch + 1)
        local_list[idx].conv26.weight.data = local_weights_DGD_25_2[idx]['conv.weight']
        local_list[idx].bn26.weight.data = local_weights_DGD_26_2[idx]['bn_2.weight']
        local_list[idx].bn26.bias.data = local_weights_DGD_26_2[idx]['bn_2.bias']
        local_list[idx].bn26.running_mean.data = local_weights_DGD_26_2[idx]['bn_11.running_mean']
        local_list[idx].bn26.running_var.data = local_weights_DGD_26_2[idx]['bn_12.running_var']
        local_list[idx].bn26.num_batches_tracked.data = torch.tensor(epoch + 1)
        # ---- Block 3
        local_list[idx].conv31.weight.data = local_weights_DGD_31_1[idx]['conv.weight']
        local_list[idx].bn31.weight.data = local_weights_DGD_31_2[idx]['bn_2.weight']
        local_list[idx].bn31.bias.data = local_weights_DGD_31_2[idx]['bn_2.bias']
        local_list[idx].bn31.running_mean.data = local_weights_DGD_31_2[idx]['bn_11.running_mean']
        local_list[idx].bn31.running_var.data = local_weights_DGD_31_2[idx]['bn_12.running_var']
        local_list[idx].bn31.num_batches_tracked.data = torch.tensor(epoch + 1)
        local_list[idx].conv32.weight.data = local_weights_DGD_31_2[idx]['conv.weight']
        local_list[idx].bn32.weight.data = local_weights_DGD_32_2[idx]['bn_2.weight']
        local_list[idx].bn32.bias.data = local_weights_DGD_32_2[idx]['bn_2.bias']
        local_list[idx].bn32.running_mean.data = local_weights_DGD_32_2[idx]['bn_11.running_mean']
        local_list[idx].bn32.running_var.data = local_weights_DGD_32_2[idx]['bn_12.running_var']
        local_list[idx].bn32.num_batches_tracked.data = torch.tensor(epoch + 1)

        local_list[idx].conv33.weight.data = local_weights_DGD_33_1[idx]['conv.weight']
        local_list[idx].bn33.weight.data = local_weights_DGD_33_2[idx]['bn_2.weight']
        local_list[idx].bn33.bias.data = local_weights_DGD_33_2[idx]['bn_2.bias']
        local_list[idx].bn33.running_mean.data = local_weights_DGD_33_2[idx]['bn_11.running_mean']
        local_list[idx].bn33.running_var.data = local_weights_DGD_33_2[idx]['bn_12.running_var']
        local_list[idx].bn33.num_batches_tracked.data = torch.tensor(epoch + 1)
        local_list[idx].conv34.weight.data = local_weights_DGD_33_2[idx]['conv.weight']
        local_list[idx].bn34.weight.data = local_weights_DGD_34_2[idx]['bn_2.weight']
        local_list[idx].bn34.bias.data = local_weights_DGD_34_2[idx]['bn_2.bias']
        local_list[idx].bn34.running_mean.data = local_weights_DGD_34_2[idx]['bn_11.running_mean']
        local_list[idx].bn34.running_var.data = local_weights_DGD_34_2[idx]['bn_12.running_var']
        local_list[idx].bn34.num_batches_tracked.data = torch.tensor(epoch + 1)

        local_list[idx].conv35.weight.data = local_weights_DGD_35_1[idx]['conv.weight']
        local_list[idx].bn35.weight.data = local_weights_DGD_35_2[idx]['bn_2.weight']
        local_list[idx].bn35.bias.data = local_weights_DGD_35_2[idx]['bn_2.bias']
        local_list[idx].bn35.running_mean.data = local_weights_DGD_35_2[idx]['bn_11.running_mean']
        local_list[idx].bn35.running_var.data = local_weights_DGD_35_2[idx]['bn_12.running_var']
        local_list[idx].bn35.num_batches_tracked.data = torch.tensor(epoch + 1)
        local_list[idx].conv36.weight.data = local_weights_DGD_35_2[idx]['conv.weight']
        local_list[idx].bn36.weight.data = local_weights_DGD_36_2[idx]['bn_2.weight']
        local_list[idx].bn36.bias.data = local_weights_DGD_36_2[idx]['bn_2.bias']
        local_list[idx].bn36.running_mean.data = local_weights_DGD_36_2[idx]['bn_11.running_mean']
        local_list[idx].bn36.running_var.data = local_weights_DGD_36_2[idx]['bn_12.running_var']
        local_list[idx].bn36.num_batches_tracked.data = torch.tensor(epoch + 1)
        # ---- Final
        local_list[idx].linear.weight.data = local_weights_DGD_4[idx]['linear.weight']
        local_list[idx].linear.bias.data = local_weights_DGD_4[idx]['linear.bias']

    del local_weights_DGD_0_1, local_weights_DGD_0_2
    del local_weights_DGD_11_1, local_weights_DGD_11_2, local_weights_DGD_12_2, local_weights_DGD_13_1, local_weights_DGD_13_2, local_weights_DGD_14_2, local_weights_DGD_15_1, local_weights_DGD_15_2, local_weights_DGD_16_2
    del local_weights_DGD_21_1, local_weights_DGD_21_2, local_weights_DGD_22_2, local_weights_DGD_23_1, local_weights_DGD_23_2, local_weights_DGD_24_2, local_weights_DGD_25_1, local_weights_DGD_25_2, local_weights_DGD_26_2
    del local_weights_DGD_31_1, local_weights_DGD_31_2, local_weights_DGD_32_2, local_weights_DGD_33_1, local_weights_DGD_33_2, local_weights_DGD_34_2, local_weights_DGD_35_1, local_weights_DGD_35_2, local_weights_DGD_36_2
    del local_weights_DGD_4

    return local_list


# transform whole network form to subnet form
def whole_2_subnet(node_number, global_average_weight, device, half_flag):

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
    model_4 \
    = initial_subnet_model(device, half_flag)

    # updating local model parameters
    # ---- block 0
    local_list_0_1, local_list_0_2=[[]] * node_number, [[]] * node_number
    local_list_0_Mean, local_list_0_Var=[[]] * node_number, [[]] * node_number
    # ---- block 1
    local_list_11_1, local_list_11_2, local_list_12_2=[[]] * node_number, [[]] * node_number, [[]] * node_number
    local_list_11_Mean, local_list_11_Var=[[]] * node_number, [[]] * node_number
    local_list_12_Mean, local_list_12_Var=[[]] * node_number, [[]] * node_number
    local_list_13_1, local_list_13_2, local_list_14_2=[[]] * node_number, [[]] * node_number, [[]] * node_number
    local_list_13_Mean, local_list_13_Var=[[]] * node_number, [[]] * node_number
    local_list_14_Mean, local_list_14_Var=[[]] * node_number, [[]] * node_number
    local_list_15_1, local_list_15_2, local_list_16_2=[[]] * node_number, [[]] * node_number, [[]] * node_number
    local_list_15_Mean, local_list_15_Var=[[]] * node_number, [[]] * node_number
    local_list_16_Mean, local_list_16_Var=[[]] * node_number, [[]] * node_number
    # ---- block 2
    local_list_21_1, local_list_21_2, local_list_22_2=[[]] * node_number, [[]] * node_number, [[]] * node_number
    local_list_21_Mean, local_list_21_Var=[[]] * node_number, [[]] * node_number
    local_list_22_Mean, local_list_22_Var=[[]] * node_number, [[]] * node_number
    local_list_23_1, local_list_23_2, local_list_24_2=[[]] * node_number, [[]] * node_number, [[]] * node_number
    local_list_23_Mean, local_list_23_Var=[[]] * node_number, [[]] * node_number
    local_list_24_Mean, local_list_24_Var=[[]] * node_number, [[]] * node_number
    local_list_25_1, local_list_25_2, local_list_26_2=[[]] * node_number, [[]] * node_number, [[]] * node_number
    local_list_25_Mean, local_list_25_Var=[[]] * node_number, [[]] * node_number
    local_list_26_Mean, local_list_26_Var=[[]] * node_number, [[]] * node_number
    # ---- block 3
    local_list_31_1, local_list_31_2, local_list_32_2=[[]] * node_number, [[]] * node_number, [[]] * node_number
    local_list_31_Mean, local_list_31_Var=[[]] * node_number, [[]] * node_number
    local_list_32_Mean, local_list_32_Var=[[]] * node_number, [[]] * node_number
    local_list_33_1, local_list_33_2, local_list_34_2=[[]] * node_number, [[]] * node_number, [[]] * node_number
    local_list_33_Mean, local_list_33_Var=[[]] * node_number, [[]] * node_number
    local_list_34_Mean, local_list_34_Var=[[]] * node_number, [[]] * node_number
    local_list_35_1, local_list_35_2, local_list_36_2=[[]] * node_number, [[]] * node_number, [[]] * node_number
    local_list_35_Mean, local_list_35_Var=[[]] * node_number, [[]] * node_number
    local_list_36_Mean, local_list_36_Var=[[]] * node_number, [[]] * node_number
    # ---- Final
    local_list_4 =[[]] * node_number


    for i in range(node_number):
        # ---- block 0
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
        local_list_0_1[i].conv.weight.data = copy.deepcopy(global_average_weight['conv0.weight'])
        local_list_0_2[i].bn_2.weight.data = copy.deepcopy(global_average_weight['bn0.weight'])
        local_list_0_2[i].bn_2.bias.data = copy.deepcopy(global_average_weight['bn0.bias'])
        local_list_0_2[i].bn_11.running_mean.data = copy.deepcopy(global_average_weight['bn0.running_mean'])
        local_list_0_2[i].bn_12.running_var.data = copy.deepcopy(global_average_weight['bn0.running_var'])
        # ---- Block 1
        local_list_11_1[i].conv.weight.data = copy.deepcopy(global_average_weight['conv11.weight'])
        local_list_11_2[i].bn_2.weight.data = copy.deepcopy(global_average_weight['bn11.weight'])
        local_list_11_2[i].bn_2.bias.data = copy.deepcopy(global_average_weight['bn11.bias'])
        local_list_11_2[i].bn_11.running_mean.data = copy.deepcopy(global_average_weight['bn11.running_mean'])
        local_list_11_2[i].bn_12.running_var.data = copy.deepcopy(global_average_weight['bn11.running_var'])
        local_list_11_2[i].conv.weight.data = copy.deepcopy(global_average_weight['conv12.weight'])
        local_list_12_2[i].bn_2.weight.data = copy.deepcopy(global_average_weight['bn12.weight'])
        local_list_12_2[i].bn_2.bias.data = copy.deepcopy(global_average_weight['bn12.bias'])
        local_list_12_2[i].bn_11.running_mean.data = copy.deepcopy(global_average_weight['bn12.running_mean'])
        local_list_12_2[i].bn_12.running_var.data = copy.deepcopy(global_average_weight['bn12.running_var'])

        local_list_13_1[i].conv.weight.data = copy.deepcopy(global_average_weight['conv13.weight'])
        local_list_13_2[i].bn_2.weight.data = copy.deepcopy(global_average_weight['bn13.weight'])
        local_list_13_2[i].bn_2.bias.data = copy.deepcopy(global_average_weight['bn13.bias'])
        local_list_13_2[i].bn_11.running_mean.data = copy.deepcopy(global_average_weight['bn13.running_mean'])
        local_list_13_2[i].bn_12.running_var.data = copy.deepcopy(global_average_weight['bn13.running_var'])
        local_list_13_2[i].conv.weight.data = copy.deepcopy(global_average_weight['conv14.weight'])
        local_list_14_2[i].bn_2.weight.data = copy.deepcopy(global_average_weight['bn14.weight'])
        local_list_14_2[i].bn_2.bias.data = copy.deepcopy(global_average_weight['bn14.bias'])
        local_list_14_2[i].bn_11.running_mean.data = copy.deepcopy(global_average_weight['bn14.running_mean'])
        local_list_14_2[i].bn_12.running_var.data = copy.deepcopy(global_average_weight['bn14.running_var'])

        local_list_15_1[i].conv.weight.data = copy.deepcopy(global_average_weight['conv15.weight'])
        local_list_15_2[i].bn_2.weight.data = copy.deepcopy(global_average_weight['bn15.weight'])
        local_list_15_2[i].bn_2.bias.data = copy.deepcopy(global_average_weight['bn15.bias'])
        local_list_15_2[i].bn_11.running_mean.data = copy.deepcopy(global_average_weight['bn15.running_mean'])
        local_list_15_2[i].bn_12.running_var.data = copy.deepcopy(global_average_weight['bn15.running_var'])
        local_list_15_2[i].conv.weight.data = copy.deepcopy(global_average_weight['conv16.weight'])
        local_list_16_2[i].bn_2.weight.data = copy.deepcopy(global_average_weight['bn16.weight'])
        local_list_16_2[i].bn_2.bias.data = copy.deepcopy(global_average_weight['bn16.bias'])
        local_list_16_2[i].bn_11.running_mean.data = copy.deepcopy(global_average_weight['bn16.running_mean'])
        local_list_16_2[i].bn_12.running_var.data = copy.deepcopy(global_average_weight['bn16.running_var'])
        # ---- Block 2
        local_list_21_1[i].conv.weight.data = copy.deepcopy(global_average_weight['conv21.weight'])
        local_list_21_2[i].bn_2.weight.data = copy.deepcopy(global_average_weight['bn21.weight'])
        local_list_21_2[i].bn_2.bias.data = copy.deepcopy(global_average_weight['bn21.bias'])
        local_list_21_2[i].bn_11.running_mean.data = copy.deepcopy(global_average_weight['bn21.running_mean'])
        local_list_21_2[i].bn_12.running_var.data = copy.deepcopy(global_average_weight['bn21.running_var'])
        local_list_21_2[i].conv.weight.data = copy.deepcopy(global_average_weight['conv22.weight'])
        local_list_22_2[i].bn_2.weight.data = copy.deepcopy(global_average_weight['bn22.weight'])
        local_list_22_2[i].bn_2.bias.data = copy.deepcopy(global_average_weight['bn22.bias'])
        local_list_22_2[i].bn_11.running_mean.data = copy.deepcopy(global_average_weight['bn22.running_mean'])
        local_list_22_2[i].bn_12.running_var.data = copy.deepcopy(global_average_weight['bn22.running_var'])

        local_list_23_1[i].conv.weight.data = copy.deepcopy(global_average_weight['conv23.weight'])
        local_list_23_2[i].bn_2.weight.data = copy.deepcopy(global_average_weight['bn23.weight'])
        local_list_23_2[i].bn_2.bias.data = copy.deepcopy(global_average_weight['bn23.bias'])
        local_list_23_2[i].bn_11.running_mean.data = copy.deepcopy(global_average_weight['bn23.running_mean'])
        local_list_23_2[i].bn_12.running_var.data = copy.deepcopy(global_average_weight['bn23.running_var'])
        local_list_23_2[i].conv.weight.data = copy.deepcopy(global_average_weight['conv24.weight'])
        local_list_24_2[i].bn_2.weight.data = copy.deepcopy(global_average_weight['bn24.weight'])
        local_list_24_2[i].bn_2.bias.data = copy.deepcopy(global_average_weight['bn24.bias'])
        local_list_24_2[i].bn_11.running_mean.data = copy.deepcopy(global_average_weight['bn24.running_mean'])
        local_list_24_2[i].bn_12.running_var.data = copy.deepcopy(global_average_weight['bn24.running_var'])

        local_list_25_1[i].conv.weight.data = copy.deepcopy(global_average_weight['conv25.weight'])
        local_list_25_2[i].bn_2.weight.data = copy.deepcopy(global_average_weight['bn25.weight'])
        local_list_25_2[i].bn_2.bias.data = copy.deepcopy(global_average_weight['bn25.bias'])
        local_list_25_2[i].bn_11.running_mean.data = copy.deepcopy(global_average_weight['bn25.running_mean'])
        local_list_25_2[i].bn_12.running_var.data = copy.deepcopy(global_average_weight['bn25.running_var'])
        local_list_25_2[i].conv.weight.data = copy.deepcopy(global_average_weight['conv26.weight'])
        local_list_26_2[i].bn_2.weight.data = copy.deepcopy(global_average_weight['bn26.weight'])
        local_list_26_2[i].bn_2.bias.data = copy.deepcopy(global_average_weight['bn26.bias'])
        local_list_26_2[i].bn_11.running_mean.data = copy.deepcopy(global_average_weight['bn26.running_mean'])
        local_list_26_2[i].bn_12.running_var.data = copy.deepcopy(global_average_weight['bn26.running_var'])
        # ---- Block 3
        local_list_31_1[i].conv.weight.data = copy.deepcopy(global_average_weight['conv31.weight'])
        local_list_31_2[i].bn_2.weight.data = copy.deepcopy(global_average_weight['bn31.weight'])
        local_list_31_2[i].bn_2.bias.data = copy.deepcopy(global_average_weight['bn31.bias'])
        local_list_31_2[i].bn_11.running_mean.data = copy.deepcopy(global_average_weight['bn31.running_mean'])
        local_list_31_2[i].bn_12.running_var.data = copy.deepcopy(global_average_weight['bn31.running_var'])
        local_list_31_2[i].conv.weight.data = copy.deepcopy(global_average_weight['conv32.weight'])
        local_list_32_2[i].bn_2.weight.data = copy.deepcopy(global_average_weight['bn32.weight'])
        local_list_32_2[i].bn_2.bias.data = copy.deepcopy(global_average_weight['bn32.bias'])
        local_list_32_2[i].bn_11.running_mean.data = copy.deepcopy(global_average_weight['bn32.running_mean'])
        local_list_32_2[i].bn_12.running_var.data = copy.deepcopy(global_average_weight['bn32.running_var'])

        local_list_33_1[i].conv.weight.data = copy.deepcopy(global_average_weight['conv33.weight'])
        local_list_33_2[i].bn_2.weight.data = copy.deepcopy(global_average_weight['bn33.weight'])
        local_list_33_2[i].bn_2.bias.data = copy.deepcopy(global_average_weight['bn33.bias'])
        local_list_33_2[i].bn_11.running_mean.data = copy.deepcopy(global_average_weight['bn33.running_mean'])
        local_list_33_2[i].bn_12.running_var.data = copy.deepcopy(global_average_weight['bn33.running_var'])
        local_list_33_2[i].conv.weight.data = copy.deepcopy(global_average_weight['conv34.weight'])
        local_list_34_2[i].bn_2.weight.data = copy.deepcopy(global_average_weight['bn34.weight'])
        local_list_34_2[i].bn_2.bias.data = copy.deepcopy(global_average_weight['bn34.bias'])
        local_list_34_2[i].bn_11.running_mean.data = copy.deepcopy(global_average_weight['bn34.running_mean'])
        local_list_34_2[i].bn_12.running_var.data = copy.deepcopy(global_average_weight['bn34.running_var'])

        local_list_35_1[i].conv.weight.data = copy.deepcopy(global_average_weight['conv35.weight'])
        local_list_35_2[i].bn_2.weight.data = copy.deepcopy(global_average_weight['bn35.weight'])
        local_list_35_2[i].bn_2.bias.data = copy.deepcopy(global_average_weight['bn35.bias'])
        local_list_35_2[i].bn_11.running_mean.data = copy.deepcopy(global_average_weight['bn35.running_mean'])
        local_list_35_2[i].bn_12.running_var.data = copy.deepcopy(global_average_weight['bn35.running_var'])
        local_list_35_2[i].conv.weight.data = copy.deepcopy(global_average_weight['conv36.weight'])
        local_list_36_2[i].bn_2.weight.data = copy.deepcopy(global_average_weight['bn36.weight'])
        local_list_36_2[i].bn_2.bias.data = copy.deepcopy(global_average_weight['bn36.bias'])
        local_list_36_2[i].bn_11.running_mean.data = copy.deepcopy(global_average_weight['bn36.running_mean'])
        local_list_36_2[i].bn_12.running_var.data = copy.deepcopy(global_average_weight['bn36.running_var'])
        # ---- Final
        local_list_4[i].linear.weight.data = copy.deepcopy(global_average_weight['linear.weight'])
        local_list_4[i].linear.bias.data = copy.deepcopy(global_average_weight['linear.bias'])

    return local_list_0_1, local_list_0_2, local_list_0_Mean, local_list_0_Var, \
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