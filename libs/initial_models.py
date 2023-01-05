import copy
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from nets.ResNet20 import ResNet
from nets.Sub_ResNet20 import SubNet_Mean, SubNet_Var, SubNet_1, SubNet_2, \
                              SubNet_3_woRes, SubNet_3_ResSeq, SubNet_3_ResPad, SubNet_Fc  # Subnet components in ResNet

# initialize subnet models
def initial_subnet_model(device, half_flag):
    # ---- block 0
    model_0_1 = SubNet_1(in_channels_num = 3, out_channels_num = 16, stride_num=1).to(device)
    model_0_2 = SubNet_3_woRes(num_features = 16).to(device)
    model_0_Mean, model_0_Var = SubNet_Mean(num_features = 16).to(device), SubNet_Var(num_features = 16).to(device)
    # ---- block 1
    model_11_1 = SubNet_1(in_channels_num = 16, out_channels_num = 16, stride_num=1).to(device)
    model_11_2 = SubNet_2(num_features = 16, stride_num = 1).to(device)
    model_12_2 = SubNet_3_ResSeq(num_features = 16).to(device)
    model_11_Mean, model_11_Var = SubNet_Mean(num_features = 16).to(device),SubNet_Var(num_features = 16).to(device)
    model_12_Mean, model_12_Var = SubNet_Mean(num_features = 16).to(device), SubNet_Var(num_features = 16).to(device)
    model_13_1 = SubNet_1(in_channels_num = 16, out_channels_num = 16, stride_num=1).to(device)
    model_13_2 = SubNet_2(num_features = 16, stride_num = 1).to(device)
    model_14_2 = SubNet_3_ResSeq(num_features = 16).to(device)
    model_13_Mean, model_13_Var = SubNet_Mean(num_features = 16).to(device), SubNet_Var(num_features = 16).to(device)
    model_14_Mean, model_14_Var = SubNet_Mean(num_features = 16).to(device), SubNet_Var(num_features = 16).to(device)
    model_15_1 = SubNet_1(in_channels_num = 16, out_channels_num = 16, stride_num=1).to(device)
    model_15_2 = SubNet_2(num_features = 16, stride_num = 1).to(device)
    model_16_2 = SubNet_3_ResSeq(num_features = 16).to(device)
    model_15_Mean, model_15_Var = SubNet_Mean(num_features = 16).to(device), SubNet_Var(num_features = 16).to(device)
    model_16_Mean, model_16_Var = SubNet_Mean(num_features = 16).to(device), SubNet_Var(num_features = 16).to(device)
    # ---- block 2
    model_21_1 = SubNet_1(in_channels_num = 16, out_channels_num = 32, stride_num=2).to(device)
    model_21_2 = SubNet_2(num_features = 32, stride_num = 1).to(device)
    model_22_2 = SubNet_3_ResPad(num_features = 32).to(device)
    model_21_Mean, model_21_Var = SubNet_Mean(num_features = 32).to(device), SubNet_Var(num_features = 32).to(device)
    model_22_Mean, model_22_Var = SubNet_Mean(num_features = 32).to(device), SubNet_Var(num_features = 32).to(device)
    model_23_1 = SubNet_1(in_channels_num = 32, out_channels_num = 32, stride_num=1).to(device)
    model_23_2 = SubNet_2(num_features = 32, stride_num = 1).to(device)
    model_24_2 = SubNet_3_ResSeq(num_features = 32).to(device)
    model_23_Mean, model_23_Var = SubNet_Mean(num_features = 32).to(device), SubNet_Var(num_features = 32).to(device)
    model_24_Mean, model_24_Var = SubNet_Mean(num_features = 32).to(device), SubNet_Var(num_features = 32).to(device)
    model_25_1 = SubNet_1(in_channels_num = 32, out_channels_num = 32, stride_num=1).to(device)
    model_25_2 = SubNet_2(num_features = 32, stride_num = 1).to(device)
    model_26_2 = SubNet_3_ResSeq(num_features = 32).to(device)
    model_25_Mean, model_25_Var = SubNet_Mean(num_features = 32).to(device), SubNet_Var(num_features = 32).to(device)
    model_26_Mean, model_26_Var = SubNet_Mean(num_features = 32).to(device), SubNet_Var(num_features = 32).to(device)
    # ---- block 3
    model_31_1 = SubNet_1(in_channels_num = 32, out_channels_num = 64, stride_num=2).to(device)
    model_31_2 = SubNet_2(num_features = 64, stride_num = 1).to(device)
    model_32_2 = SubNet_3_ResPad(num_features = 64).to(device)
    model_31_Mean, model_31_Var = SubNet_Mean(num_features = 64).to(device), SubNet_Var(num_features = 64).to(device)
    model_32_Mean, model_32_Var = SubNet_Mean(num_features = 64).to(device), SubNet_Var(num_features = 64).to(device)
    model_33_1 = SubNet_1(in_channels_num = 64, out_channels_num = 64, stride_num=1).to(device)
    model_33_2 = SubNet_2(num_features = 64, stride_num = 1).to(device)
    model_34_2 = SubNet_3_ResSeq(num_features = 64).to(device)
    model_33_Mean, model_33_Var = SubNet_Mean(num_features = 64).to(device), SubNet_Var(num_features = 64).to(device)
    model_34_Mean, model_34_Var = SubNet_Mean(num_features = 64).to(device), SubNet_Var(num_features = 64).to(device)
    model_35_1 = SubNet_1(in_channels_num = 64, out_channels_num = 64, stride_num=1).to(device)
    model_35_2 = SubNet_2(num_features = 64, stride_num = 1).to(device)
    model_36_2 = SubNet_3_ResSeq(num_features = 64).to(device)
    model_35_Mean, model_35_Var = SubNet_Mean(num_features = 64).to(device), SubNet_Var(num_features = 64).to(device)
    model_36_Mean, model_36_Var = SubNet_Mean(num_features = 64).to(device), SubNet_Var(num_features = 64).to(device)
    # ---- Final
    model_4 = SubNet_Fc(num_features = 64, num_classes = 10).to(device)

    if half_flag:
        # ---- block 0
        model_0_1, model_0_2, model_0_Mean, model_0_Var = model_0_1.half(), model_0_2.half(), model_0_Mean.half(), model_0_Var.half()
        # ---- block 1
        model_11_1, model_11_2, model_12_2 = model_11_1.half(), model_11_2.half(), model_12_2.half()
        model_11_Mean, model_11_Var, model_12_Mean, model_12_Var = model_11_Mean.half(), model_11_Var.half(), model_12_Mean.half(), model_12_Var.half()
        model_13_1, model_13_2, model_14_2 = model_13_1.half(), model_13_2.half(), model_14_2.half()
        model_13_Mean, model_13_Var, model_14_Mean, model_14_Var = model_13_Mean.half(), model_13_Var.half(), model_14_Mean.half(), model_14_Var.half()
        model_15_1, model_15_2, model_16_2 = model_15_1.half(), model_15_2.half(), model_16_2.half()
        model_15_Mean, model_15_Var, model_16_Mean, model_16_Var = model_15_Mean.half(), model_15_Var.half(), model_16_Mean.half(), model_16_Var.half()
        # ---- block 2
        model_21_1, model_21_2, model_22_2 = model_21_1.half(), model_21_2.half(), model_22_2.half()
        model_21_Mean, model_21_Var, model_22_Mean, model_22_Var = model_21_Mean.half(), model_21_Var.half(), model_22_Mean.half(), model_22_Var.half()
        model_23_1, model_23_2, model_24_2 = model_23_1.half(), model_23_2.half(), model_24_2.half()
        model_23_Mean, model_23_Var, model_24_Mean, model_24_Var = model_23_Mean.half(), model_23_Var.half(), model_24_Mean.half(), model_24_Var.half()
        model_25_1, model_25_2, model_26_2 = model_25_1.half(), model_25_2.half(), model_26_2.half()
        model_25_Mean, model_25_Var, model_26_Mean, model_26_Var = model_25_Mean.half(), model_25_Var.half(), model_26_Mean.half(), model_26_Var.half()
        # ---- block 3
        model_31_1, model_31_2, model_32_2 = model_31_1.half(), model_31_2.half(), model_32_2.half()
        model_31_Mean, model_31_Var, model_32_Mean, model_32_Var = model_31_Mean.half(), model_31_Var.half(), model_32_Mean.half(), model_32_Var.half()
        model_33_1, model_33_2, model_34_2 = model_33_1.half(), model_33_2.half(), model_34_2.half()
        model_33_Mean, model_33_Var, model_34_Mean, model_34_Var = model_33_Mean.half(), model_33_Var.half(), model_34_Mean.half(), model_34_Var.half()
        model_35_1, model_35_2, model_36_2 = model_35_1.half(), model_35_2.half(), model_36_2.half()
        model_35_Mean, model_35_Var, model_36_Mean, model_36_Var = model_35_Mean.half(), model_35_Var.half(), model_36_Mean.half(), model_36_Var.half()
        # ---- Final
        model_4 = model_4.half()

    return model_0_1, model_0_2, model_0_Mean, model_0_Var, \
           model_11_1, model_11_2, model_12_2, model_11_Mean, model_11_Var, model_12_Mean, model_12_Var, \
           model_13_1, model_13_2, model_14_2, model_13_Mean, model_13_Var, model_14_Mean, model_14_Var, \
           model_15_1, model_15_2, model_16_2, model_15_Mean, model_15_Var, model_16_Mean, model_16_Var, \
           model_21_1, model_21_2, model_22_2, model_21_Mean, model_21_Var, model_22_Mean, model_22_Var, \
           model_23_1, model_23_2, model_24_2, model_23_Mean, model_23_Var, model_24_Mean, model_24_Var, \
           model_25_1, model_25_2, model_26_2, model_25_Mean, model_25_Var, model_26_Mean, model_26_Var, \
           model_31_1, model_31_2, model_32_2, model_31_Mean, model_31_Var, model_32_Mean, model_32_Var, \
           model_33_1, model_33_2, model_34_2, model_33_Mean, model_33_Var, model_34_Mean, model_34_Var, \
           model_35_1, model_35_2, model_36_2, model_35_Mean, model_35_Var, model_36_Mean, model_36_Var, \
           model_4


# initialize the global model in server and the local models in clients
def initial_global_local_models(device, half_flag, node_number):
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

    # ============ initialize global model in server ============ #
    global_average = ResNet().to(device)
    if half_flag:
        global_average = global_average.half()
    global_average_weight = copy.deepcopy(global_average.state_dict())

    # ---- Initial
    global_average_weight['conv0.weight'] = model_0_1.state_dict()['conv.weight']
    global_average_weight['bn0.weight'] = model_0_2.state_dict()['bn_2.weight']
    global_average_weight['bn0.bias'] = model_0_2.state_dict()['bn_2.bias']
    global_average_weight['bn0.running_mean'] = model_0_2.state_dict()['bn_11.running_mean']
    global_average_weight['bn0.running_var'] = model_0_2.state_dict()['bn_12.running_var']
    #  global_average_weight['bn0.num_batches_tracked'] =
    # ---- block 1
    global_average_weight['conv11.weight'] = model_11_1.state_dict()['conv.weight']
    global_average_weight['bn11.weight'] = model_11_2.state_dict()['bn_2.weight']
    global_average_weight['bn11.bias'] = model_11_2.state_dict()['bn_2.bias']
    global_average_weight['bn11.running_mean'] = model_11_2.state_dict()['bn_11.running_mean']
    global_average_weight['bn11.running_var'] = model_11_2.state_dict()['bn_12.running_var']
    #  global_average_weight['bn11.num_batches_tracked'] =
    global_average_weight['conv12.weight'] = model_11_2.state_dict()['conv.weight']
    global_average_weight['bn12.weight'] = model_12_2.state_dict()['bn_2.weight']
    global_average_weight['bn12.bias'] = model_12_2.state_dict()['bn_2.bias']
    global_average_weight['bn12.running_mean'] = model_12_2.state_dict()['bn_11.running_mean']
    global_average_weight['bn12.running_var'] = model_12_2.state_dict()['bn_12.running_var']
    #  global_average_weight['bn12.num_batches_tracked'] =
    global_average_weight['conv13.weight'] = model_13_1.state_dict()['conv.weight']
    global_average_weight['bn13.weight'] = model_13_2.state_dict()['bn_2.weight']
    global_average_weight['bn13.bias'] = model_13_2.state_dict()['bn_2.bias']
    global_average_weight['bn13.running_mean'] = model_13_2.state_dict()['bn_11.running_mean']
    global_average_weight['bn13.running_var'] = model_13_2.state_dict()['bn_12.running_var']
    #  global_average_weight['bn13.num_batches_tracked'] =
    global_average_weight['conv14.weight'] = model_13_2.state_dict()['conv.weight']
    global_average_weight['bn14.weight'] = model_14_2.state_dict()['bn_2.weight']
    global_average_weight['bn14.bias'] = model_14_2.state_dict()['bn_2.bias']
    global_average_weight['bn14.running_mean'] = model_14_2.state_dict()['bn_11.running_mean']
    global_average_weight['bn14.running_var'] = model_14_2.state_dict()['bn_12.running_var']
    #  global_average_weight['bn14.num_batches_tracked'] =
    global_average_weight['conv15.weight'] = model_15_1.state_dict()['conv.weight']
    global_average_weight['bn15.weight'] = model_15_2.state_dict()['bn_2.weight']
    global_average_weight['bn15.bias'] = model_15_2.state_dict()['bn_2.bias']
    global_average_weight['bn15.running_mean'] = model_15_2.state_dict()['bn_11.running_mean']
    global_average_weight['bn15.running_var'] = model_15_2.state_dict()['bn_12.running_var']
    #  global_average_weight['bn15.num_batches_tracked'] =
    global_average_weight['conv16.weight'] = model_15_2.state_dict()['conv.weight']
    global_average_weight['bn16.weight'] = model_16_2.state_dict()['bn_2.weight']
    global_average_weight['bn16.bias'] = model_16_2.state_dict()['bn_2.bias']
    global_average_weight['bn16.running_mean'] = model_16_2.state_dict()['bn_11.running_mean']
    global_average_weight['bn16.running_var'] = model_16_2.state_dict()['bn_12.running_var']
    #  global_average_weight['bn16.num_batches_tracked'] =
    # ---- block 2
    global_average_weight['conv21.weight'] = model_21_1.state_dict()['conv.weight']
    global_average_weight['bn21.weight'] = model_21_2.state_dict()['bn_2.weight']
    global_average_weight['bn21.bias'] = model_21_2.state_dict()['bn_2.bias']
    global_average_weight['bn21.running_mean'] = model_21_2.state_dict()['bn_11.running_mean']
    global_average_weight['bn21.running_var'] = model_21_2.state_dict()['bn_12.running_var']
    #  global_average_weight['bn21.num_batches_tracked'] =
    global_average_weight['conv22.weight'] = model_21_2.state_dict()['conv.weight']
    global_average_weight['bn22.weight'] = model_22_2.state_dict()['bn_2.weight']
    global_average_weight['bn22.bias'] = model_22_2.state_dict()['bn_2.bias']
    global_average_weight['bn22.running_mean'] = model_22_2.state_dict()['bn_11.running_mean']
    global_average_weight['bn22.running_var'] = model_22_2.state_dict()['bn_12.running_var']
    #  global_average_weight['bn22.num_batches_tracked'] =
    global_average_weight['conv23.weight'] = model_23_1.state_dict()['conv.weight']
    global_average_weight['bn23.weight'] = model_23_2.state_dict()['bn_2.weight']
    global_average_weight['bn23.bias'] = model_23_2.state_dict()['bn_2.bias']
    global_average_weight['bn23.running_mean'] = model_23_2.state_dict()['bn_11.running_mean']
    global_average_weight['bn23.running_var'] = model_23_2.state_dict()['bn_12.running_var']
    #  global_average_weight['bn23.num_batches_tracked'] =
    global_average_weight['conv24.weight'] = model_23_2.state_dict()['conv.weight']
    global_average_weight['bn24.weight'] = model_24_2.state_dict()['bn_2.weight']
    global_average_weight['bn24.bias'] = model_24_2.state_dict()['bn_2.bias']
    global_average_weight['bn24.running_mean'] = model_24_2.state_dict()['bn_11.running_mean']
    global_average_weight['bn24.running_var'] = model_24_2.state_dict()['bn_12.running_var']
    #  global_average_weight['bn24.num_batches_tracked'] =
    global_average_weight['conv25.weight'] = model_25_1.state_dict()['conv.weight']
    global_average_weight['bn25.weight'] = model_25_2.state_dict()['bn_2.weight']
    global_average_weight['bn25.bias'] = model_25_2.state_dict()['bn_2.bias']
    global_average_weight['bn25.running_mean'] = model_25_2.state_dict()['bn_11.running_mean']
    global_average_weight['bn25.running_var'] = model_25_2.state_dict()['bn_12.running_var']
    #  global_average_weight['bn25.num_batches_tracked'] =
    global_average_weight['conv26.weight'] = model_25_2.state_dict()['conv.weight']
    global_average_weight['bn26.weight'] = model_26_2.state_dict()['bn_2.weight']
    global_average_weight['bn26.bias'] = model_26_2.state_dict()['bn_2.bias']
    global_average_weight['bn26.running_mean'] = model_26_2.state_dict()['bn_11.running_mean']
    global_average_weight['bn26.running_var'] = model_26_2.state_dict()['bn_12.running_var']
    #  global_average_weight['bn26.num_batches_tracked'] =
    # ---- block 3
    global_average_weight['conv31.weight'] = model_31_1.state_dict()['conv.weight']
    global_average_weight['bn31.weight'] = model_31_2.state_dict()['bn_2.weight']
    global_average_weight['bn31.bias'] = model_31_2.state_dict()['bn_2.bias']
    global_average_weight['bn31.running_mean'] = model_31_2.state_dict()['bn_11.running_mean']
    global_average_weight['bn31.running_var'] = model_31_2.state_dict()['bn_12.running_var']
    #  global_average_weight['bn31.num_batches_tracked'] =
    global_average_weight['conv32.weight'] = model_31_2.state_dict()['conv.weight']
    global_average_weight['bn32.weight'] = model_32_2.state_dict()['bn_2.weight']
    global_average_weight['bn32.bias'] = model_32_2.state_dict()['bn_2.bias']
    global_average_weight['bn32.running_mean'] = model_32_2.state_dict()['bn_11.running_mean']
    global_average_weight['bn32.running_var'] = model_32_2.state_dict()['bn_12.running_var']
    #  global_average_weight['bn32.num_batches_tracked'] =
    global_average_weight['conv33.weight'] = model_33_1.state_dict()['conv.weight']
    global_average_weight['bn33.weight'] = model_33_2.state_dict()['bn_2.weight']
    global_average_weight['bn33.bias'] = model_33_2.state_dict()['bn_2.bias']
    global_average_weight['bn33.running_mean'] = model_33_2.state_dict()['bn_11.running_mean']
    global_average_weight['bn33.running_var'] = model_33_2.state_dict()['bn_12.running_var']
    #  global_average_weight['bn33.num_batches_tracked'] =
    global_average_weight['conv34.weight'] = model_33_2.state_dict()['conv.weight']
    global_average_weight['bn34.weight'] = model_34_2.state_dict()['bn_2.weight']
    global_average_weight['bn34.bias'] = model_34_2.state_dict()['bn_2.bias']
    global_average_weight['bn34.running_mean'] = model_34_2.state_dict()['bn_11.running_mean']
    global_average_weight['bn34.running_var'] = model_34_2.state_dict()['bn_12.running_var']
    #  global_average_weight['bn34.num_batches_tracked'] =
    global_average_weight['conv35.weight'] = model_35_1.state_dict()['conv.weight']
    global_average_weight['bn35.weight'] = model_35_2.state_dict()['bn_2.weight']
    global_average_weight['bn35.bias'] = model_35_2.state_dict()['bn_2.bias']
    global_average_weight['bn35.running_mean'] = model_35_2.state_dict()['bn_11.running_mean']
    global_average_weight['bn35.running_var'] = model_35_2.state_dict()['bn_12.running_var']
    #  global_average_weight['bn35.num_batches_tracked'] =
    global_average_weight['conv36.weight'] = model_35_2.state_dict()['conv.weight']
    global_average_weight['bn36.weight'] = model_36_2.state_dict()['bn_2.weight']
    global_average_weight['bn36.bias'] = model_36_2.state_dict()['bn_2.bias']
    global_average_weight['bn36.running_mean'] = model_36_2.state_dict()['bn_11.running_mean']
    global_average_weight['bn36.running_var'] = model_36_2.state_dict()['bn_12.running_var']
    #  global_average_weight['bn36.num_batches_tracked'] =
    # ---- Final
    global_average_weight['linear.weight'] = model_4.state_dict()['linear.weight']
    global_average_weight['linear.bias'] = model_4.state_dict()['linear.bias']

    global_average.load_state_dict(global_average_weight)


    # ============ initialize local models in clients (in subnet form) ============ #
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

    return  global_average, global_average_weight, \
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