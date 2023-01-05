import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data

# ====================== training on whole network ====================== #
def train(train_loader, model, criterion, optimizer, device, half_flag):

    model.train()
    for i, (input, target) in enumerate(train_loader):

        target = target.to(device)
        input_var = input.to(device)
        target_var = target.to(device)
        if half_flag:
            input_var = input_var.half()
        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()



# ====================== training on subnet ====================== #
# (1) forward 1
def train_forward_1(model_1, model_Mean, out_list):

    input_list_1, mean_list_1 = [], []
    # compute output
    input_, batchsize, channels, height, width = model_1(out_list[0])
    mean = model_Mean(input_)

    input_list_1.append(input_)
    mean_list_1.append(mean)

    return input_list_1, mean_list_1, batchsize, channels, height, width


#  variance
def train_forward_var(model_Var, input_list, mean_list):

    bias_var_list = []
    # compute output
    bias_var = model_Var(input_list[0], mean_list[0])
    bias_var_list.append(bias_var)

    return bias_var_list


# (2) forward 2
def train_forward_2(model_2, model_Mean,
                    input_list_1, mean_list_1, bias_var_list_1,
                    batchsize, channels, height, width):

    model_2_running_mean = model_2.state_dict()['bn_11.running_mean']
    model_2_running_var = model_2.state_dict()['bn_12.running_var']

    input_list_2 = []
    mean_list_2 = []
    # compute output
    input_, batchsize, channels, height, width = model_2(input_list_1[0], mean_list_1[0], bias_var_list_1[0],
                                                         batchsize, channels, height, width)
    mean = model_Mean(input_)
    input_list_2.append(input_)
    mean_list_2.append(mean)

    model_2.bn_11.running_mean = model_2_running_mean
    model_2.bn_12.running_var = model_2_running_var

    return input_list_2, mean_list_2, batchsize, channels, height, width


# (3) forward 3
# ---- without ResNet
def train_forward_3_woRes(model_3, input_list_1, mean_list_1, bias_var_list_1,
                          batchsize, channels, height, width):
    model_3_running_mean = model_3.state_dict()['bn_11.running_mean']
    model_3_running_var = model_3.state_dict()['bn_12.running_var']

    out_list = []
    # compute output
    out = model_3(input_list_1[0], mean_list_1[0], bias_var_list_1[0],
                  batchsize, channels, height, width)
    out_list.append(out)

    model_3.bn_11.running_mean = model_3_running_mean
    model_3.bn_12.running_var = model_3_running_var

    return out_list


# ---- with ResNet
def train_forward_3_Res(model_3, out1_list, input_list_2, mean_list_2, bias_var_list_2,
                        batchsize, channels, height, width):

    model_3_running_mean = model_3.state_dict()['bn_11.running_mean']
    model_3_running_var = model_3.state_dict()['bn_12.running_var']

    out_list = []
    # compute output
    out = model_3(input_list_2[0], mean_list_2[0], bias_var_list_2[0], out1_list[0],
                  batchsize, channels, height, width)
    out_list.append(out)
    model_3.bn_11.running_mean = model_3_running_mean
    model_3.bn_12.running_var = model_3_running_var

    return out_list


# (4) fc
def hook_model_fc_origin(module, grad_input, grad_output):
    global grad_model_fc_origin_input_0
    grad_model_fc_origin_input_0 = grad_input[0]


def train_forward_fc(model_fc, out_list, target_list,
                     optimizer, criterion):

    model_fc.train()

    grad_model_fc_origin_input_0_list = []
    # obtain gradient
    model_fc.origin.register_full_backward_hook(hook_model_fc_origin)
    # compute output
    output = model_fc(out_list[0])
    # loss
    loss = criterion(output, target_list[0])
    # compute gradient and do SGD step
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()

    global grad_model_fc_origin_input_0
    grad_model_fc_origin_input_0_list.append(grad_model_fc_origin_input_0)

    return grad_model_fc_origin_input_0_list


# (3) backward 3
def hook_bn_11(module, grad_input, grad_output):
    global grad_bn_11_input_0
    grad_bn_11_input_0 = grad_input[0]
    global grad_bn_11_input_1
    grad_bn_11_input_1 = grad_input[1]

def hook_bn_12(module, grad_input, grad_output):
    global grad_bn_12_input_0
    grad_bn_12_input_0 = grad_input[0]

def hook_model_3_shortcut(module, grad_input, grad_output):
    global grad_model_3_shortcut_input_0
    grad_model_3_shortcut_input_0 = grad_input[0]


# ---- block 0 (ResNet to next block)
def train_backward_3_ResOut(model_3, optimizer, input_list_2, mean_list_2, bias_var_list_2,
                            batchsize, channels, height, width,
                            grad_model_1_conv_input_0_list, grad_model_3_shortcut_input_0_list_2, device):
    model_3.train()

    grad_bn_11_input_0_list_1 = []
    grad_bn_11_input_1_list_1 = []
    grad_bn_12_input_0_list_1 = []

    model_3.bn_11.register_full_backward_hook(hook_bn_11)
    model_3.bn_12.register_full_backward_hook(hook_bn_12)
    # compute output
    out = model_3(input_list_2[0], mean_list_2[0], bias_var_list_2[0],
                  batchsize, channels, height, width)
    # loss
    grad_model_3_relu_output = grad_model_1_conv_input_0_list[0].to(device) + grad_model_3_shortcut_input_0_list_2[0].to(device)
    loss = grad_model_3_relu_output.detach() * out
    # compute gradient
    optimizer.zero_grad()
    loss.backward(torch.ones_like(out), retain_graph=True)
    optimizer.step()

    global grad_bn_11_input_0
    global grad_bn_11_input_1
    global grad_bn_12_input_0
    grad_bn_11_input_0_list_1.append(grad_bn_11_input_0)
    grad_bn_11_input_1_list_1.append(grad_bn_11_input_1)
    grad_bn_12_input_0_list_1.append(grad_bn_12_input_0)

    return grad_bn_11_input_0_list_1, grad_bn_11_input_1_list_1, grad_bn_12_input_0_list_1


# ---- block 1, 2 (ResNet from previous block, ResNet to next block)
def train_backward_3_ResInOut(model_3, optimizer, out1_list, input_list_2, mean_list_2, bias_var_list_2,
                              batchsize, channels, height, width,
                              grad_model_1_conv_input_0_list, grad_model_3_shortcut_input_0_list_2, device):

    model_3.train()

    grad_bn_11_input_0_list_1 = []
    grad_bn_11_input_1_list_1 = []
    grad_bn_12_input_0_list_1 = []
    grad_model_3_shortcut_input_0_list_1 = []

    model_3.shortcut.register_full_backward_hook(hook_model_3_shortcut)
    model_3.bn_11.register_full_backward_hook(hook_bn_11)
    model_3.bn_12.register_full_backward_hook(hook_bn_12)
    # compute output
    out2 = model_3(input_list_2[0], mean_list_2[0], bias_var_list_2[0], out1_list[0],
                   batchsize, channels, height, width)
    # loss
    grad_model_3_relu_output = grad_model_1_conv_input_0_list[0].to(device) + grad_model_3_shortcut_input_0_list_2[0].to(device)
    loss = grad_model_3_relu_output.detach() * out2
    # compute gradient and do SGD step
    optimizer.zero_grad()
    loss.backward(torch.ones_like(out2), retain_graph=True)
    optimizer.step()

    global grad_model_3_shortcut_input_0
    global grad_bn_11_input_0
    global grad_bn_11_input_1
    global grad_bn_12_input_0
    grad_model_3_shortcut_input_0_list_1.append(grad_model_3_shortcut_input_0)
    grad_bn_11_input_0_list_1.append(grad_bn_11_input_0)
    grad_bn_11_input_1_list_1.append(grad_bn_11_input_1)
    grad_bn_12_input_0_list_1.append(grad_bn_12_input_0)

    return grad_model_3_shortcut_input_0_list_1, grad_bn_11_input_0_list_1, grad_bn_11_input_1_list_1, grad_bn_12_input_0_list_1


# ---- block 3 (ResNet from previous block)
def train_backward_3_ResIn(model_3, optimizer, out1_list, input_list_2, mean_list_2, bias_var_list_2,
                           batchsize, channels, height, width,
                           grad_model_fc_origin_input_0_list, device):
    model_3.train()

    grad_bn_11_input_0_list_1 = []
    grad_bn_11_input_1_list_1 = []
    grad_bn_12_input_0_list_1 = []
    grad_model_3_shortcut_input_0_list_1 = []

    model_3.shortcut.register_full_backward_hook(hook_model_3_shortcut)
    model_3.bn_11.register_full_backward_hook(hook_bn_11)
    model_3.bn_12.register_full_backward_hook(hook_bn_12)

    # compute output
    out2 = model_3(input_list_2[0], mean_list_2[0], bias_var_list_2[0], out1_list[0],
                   batchsize, channels, height, width)
    # loss
    grad_model_fc_origin_input_0 = grad_model_fc_origin_input_0_list[0].to(device)
    loss = grad_model_fc_origin_input_0.detach() * out2
    # compute gradient and do SGD step
    optimizer.zero_grad()
    loss.backward(torch.ones_like(out2), retain_graph=True)
    optimizer.step()

    global grad_model_3_shortcut_input_0
    global grad_bn_11_input_0
    global grad_bn_11_input_1
    global grad_bn_12_input_0
    grad_model_3_shortcut_input_0_list_1.append(grad_model_3_shortcut_input_0)
    grad_bn_11_input_0_list_1.append(grad_bn_11_input_0)
    grad_bn_11_input_1_list_1.append(grad_bn_11_input_1)
    grad_bn_12_input_0_list_1.append(grad_bn_12_input_0)

    return grad_model_3_shortcut_input_0_list_1, grad_bn_11_input_0_list_1, grad_bn_11_input_1_list_1, grad_bn_12_input_0_list_1


# (2) backward 2
# use previous "hook_bn_11"
# use previous "hook_bn_12"

def train_backward_2(model_2, optimizer, input_list_1, mean_list_1, bias_var_list_1,
                     batchsize, channels, height, width,
                     grad_bn_11_input_0_list_2, grad_bn_11_input_1_list_2, grad_bn_12_input_0_list_2,
                     input_list_2, mean_list_2, device):
    model_2.train()

    grad_bn_11_input_0_list_1 = []
    grad_bn_11_input_1_list_1 = []
    grad_bn_12_input_0_list_1 = []

    model_2.bn_11.register_full_backward_hook(hook_bn_11)
    model_2.bn_12.register_full_backward_hook(hook_bn_12)
    # compute output
    input_, batchsize, channels, height, width = model_2(input_list_1[0], mean_list_1[0], bias_var_list_1[0],
                                                         batchsize, channels, height, width)
    # loss
    grad_model_2_output = (grad_bn_11_input_0_list_2[0].to(device)
                           + grad_bn_11_input_1_list_2[0].to(device).unsqueeze(1) / (batchsize * height * width)
                           + grad_bn_12_input_0_list_2[0].to(device).unsqueeze(1)
                           * (input_list_2[0] - mean_list_2[0].unsqueeze(1)) * 2 / (batchsize * height * width))
    loss = grad_model_2_output.detach() * input_

    # compute gradient and do SGD step
    optimizer.zero_grad()
    loss.backward(torch.ones_like(input_), retain_graph=True)
    optimizer.step()

    global grad_bn_11_input_0
    global grad_bn_11_input_1
    global grad_bn_12_input_0
    grad_bn_11_input_0_list_1.append(grad_bn_11_input_0)
    grad_bn_11_input_1_list_1.append(grad_bn_11_input_1)
    grad_bn_12_input_0_list_1.append(grad_bn_12_input_0)

    return grad_bn_11_input_0_list_1, grad_bn_11_input_1_list_1, grad_bn_12_input_0_list_1


# (1) backward 1
# ---- block 1, 2, 3 (With Input)
def hook_model_1_conv(module, grad_input, grad_output):
    global grad_model_1_conv_input_0
    grad_model_1_conv_input_0 = grad_input[0]

def train_backward_1_Input(model_1, optimizer, out_list,
                           grad_bn_11_input_0_list_1, grad_bn_11_input_1_list_1, grad_bn_12_input_0_list_1,
                           input_list_1, mean_list_1):

    model_1.train()

    grad_model_1_conv_input_0_list = []
    model_1.conv.register_full_backward_hook(hook_model_1_conv)
    # compute output
    input_, batchsize, channels, height, width = model_1(out_list[0])
    # compute loss
    grad_model_1_output = (grad_bn_11_input_0_list_1[0]
                           + grad_bn_11_input_1_list_1[0].unsqueeze(1) / (batchsize * height * width)
                           + grad_bn_12_input_0_list_1[0].unsqueeze(1)
                           * (input_list_1[0] - mean_list_1[0].unsqueeze(1)) * 2 / (batchsize * height * width))
    loss = grad_model_1_output.detach() * input_

    # compute gradient
    optimizer.zero_grad()
    loss.backward(torch.ones_like(input_), retain_graph=True)
    optimizer.step()

    global grad_model_1_conv_input_0
    grad_model_1_conv_input_0_list.append(grad_model_1_conv_input_0)

    return grad_model_1_conv_input_0_list


# ---- block 0  (Without Input)
def train_backward_1_woInput(model_1, optimizer, out_list,
                             grad_bn_11_input_0_list_1, grad_bn_11_input_1_list_1, grad_bn_12_input_0_list_1,
                             input_list_1, mean_list_1):
    model_1.train()

    # compute output
    input_, batchsize, channels, height, width = model_1(out_list[0])
    # compute loss
    grad_model_1_output = (grad_bn_11_input_0_list_1[0]
                           + grad_bn_11_input_1_list_1[0].unsqueeze(1) / (batchsize * height * width)
                           + grad_bn_12_input_0_list_1[0].unsqueeze(1)
                           * (input_list_1[0] - mean_list_1[0].unsqueeze(1)) * 2 / (batchsize * height * width))
    loss = grad_model_1_output.detach() * input_
    # compute gradient
    optimizer.zero_grad()
    loss.backward(torch.ones_like(input_), retain_graph=True)
    optimizer.step()