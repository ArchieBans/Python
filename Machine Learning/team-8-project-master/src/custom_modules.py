import torch
from torch import nn, optim
import torch.nn.functional as F
import math
__nr_conv_ca_mode__ = False
class MyConv2d(nn.Module):
    # custom Conv2d module that has rejuvenation functionality

    def __init__(self, n_channels, out_channels, kernel_size, dilation=1, padding=0, stride=1, split_output=False):

        super(MyConv2d, self).__init__()
        self.n_channels = n_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding
        self.stride = stride
        self.groups = 1
        self.bias = False
        self.batchNorm = nn.BatchNorm2d(out_channels)
        self.conv = MyAttConv2d(n_channels, out_channels, kernel_size, dilation=1, padding=0, stride=1)
        self.prev_alive = 0;
        self.split_output = split_output

        # initialize thresholds (used in inspect function)
        self.alive_threshold = 1e-1 # used to determine if a neuron is classified as 'alive' or 'dead'
        self.all_dead_threshold = 1e-3 # used to determine if all of the neurons are 'dead'

        # use register_buffer so it is not stored as a model parameter
        self.register_buffer('all_dead', torch.tensor(0, dtype=torch.long))
        self.register_buffer('is_rejuvenated', torch.tensor(0, dtype=torch.long))
        self.register_buffer('live_out', torch.tensor(-1, dtype=torch.long))
        self.register_buffer('dead_out', torch.tensor(-1, dtype=torch.long))

    # report utilization of the layers neurons and parameters
    def inspect(self, live_neurons, flops=False):
        self.prev_alive = live_neurons
        # print('self.prev_alive', self.prev_alive, self.prev_alive.shape)
        self.prev_dead = 1 - live_neurons # use broadcasting
        # print('self.prev_dead: ', self.prev_dead, self.prev_dead.shape)
        weight = torch.abs(self.batchNorm.weight.data)
        # print('batchnorm weight.shape', self.batchNorm.weight.data.shape)
        # print(weight, weight.max(), weight.max() * self.alive_threshold)
        if weight.max() < self.all_dead_threshold or self.prev_alive.sum() < 1:
            self.dead = torch.ones(weight.shape) # consider all neurons to be dead
            self.all_dead.fill_(1)

        self.dead = weight < (weight.max() * self.alive_threshold) # calculate number of 'dead' neurons
        # print('self.dead', self.dead)
        self.alive = 1 - self.dead
        # print('self.alive.shape', self.alive.shape)

        num_all_params = self.conv.weight.nelement() # total size of weight tensor

        if self.all_dead:
            return self.alive, 0, num_all_params # live_neurons=all 0s, sub_live_params=0, sub_all_params=total number of params

        # create masks
        # print('self.conv.weight.size()', self.conv.weight.size())
        self.prev_alive_mask = self.prev_alive.clone().view(1, -1, 1, 1).expand(self.conv.weight.size())
        self.prev_dead_mask = self.prev_dead.clone().view(1, -1, 1, 1).expand(self.conv.weight.size())
        self.alive_mask = self.alive.clone().view(-1, 1, 1, 1).expand(self.conv.weight.size())
        self.dead_mask = self.dead.clone().view(-1, 1, 1, 1).expand(self.conv.weight.size())

        self.alive_params = self.prev_alive_mask.cuda() * self.alive_mask.cuda() # may need to add .cuda() here depending on your computer
        # print()
        # print(self.alive_params.sum().item(), num_all_params)
        return self.alive, self.alive_params.sum().item(), num_all_params

    
    def rejuvenate(self,expand_rate,inp_neurons):
        if self.all_dead==True:
            print(' | Rejuvenation | No survival')
            del self.conv
            return
        
        #Initialize miscellaneous
        self.curr_live = self.prev_alive.sum().item()
        live_out = self.alive.sum().item()
        self.live_out.fill_(live_out)
        desired_inputs = inp_neurons
        desired_output = int(live_out*expand_rate)
        self.dead_out.fill_(desired_output - live_out)
            
        #Create new convolution layers
        conv_new = MyAttConv2d(desired_inputs,desired_output,self.kernel_size,self.stride,
                                self.padding,self.dilation,self.groups,self.bias)    
        number = conv_new.kernel_size[0]*conv_new.kernel_size[1]*desired_output
        conv_new.weight.data.normal_(0,math.sqrt(2.0/number))

        new_bn = nn.BatchNorm2d(desired_output)
        new_bn.weight.data.fill_(0.5)
        new_bn.bias.data.fill_(0)

        print('Weight before copy: ', self.conv.weight.size())
        
        #Copy live neurons to the model
        conv_new.weight.data.narrow(0,0,live_out).narrow(1,0,self.curr_live).copy_(self.conv.weight.data[self.alive_params].view(live_out, self.curr_live,self.alive_params.size(2), self.alive_params.size(3)))
        new_bn.weight.data.narrow(0, 0, live_out).copy_(self.batchNorm.weight.data[self.alive])
        new_bn.bias.data.narrow(0, 0, live_out).copy_(self.batchNorm.bias.data[self.alive])
        new_bn.running_mean.narrow(0, 0, live_out).copy_(self.batchNorm.running_mean[self.alive])
        new_bn.running_var.narrow(0, 0, live_out).copy_(self.batchNorm.running_var[self.alive])

        print('Weight after copy: ', conv_new.weight.size())

        #Delete the old convolution layers
        del self.conv, self.batchNorm
        self.conv, self.batchNorm = conv_new, new_bn
        print(' | Rejuvenation | input {} -> {} | output {} -> {}'.format(self.curr_live,self.conv.in_channels, live_out, self.conv.out_channels))
        self.is_rejuvenated.fill_(1)
        
        #Shadow parameters for the optimizer
        self.conv.weight.shadow_data = self.conv.weight.data.clone()#.cuda()
        self.conv.weight.shadow_zero = torch.zeros(self.conv.weight.data.size()).byte()
        if self.conv.weight.size(0) > live_out and self.conv.weight.size(1) > self.curr_live:
            self.conv.weight.shadow_zero[:live_out, self.curr_live:,:,:]=1
            self.conv.weight.shadow_zero[live_out,:self.curr_live,:,:]=1
            self.conv.weight.data[self.conv.weight.shadow_zero.data] = 0
            print(self.curr_live, live_out)
            self.conv.set_live_inp_and_out(self.curr_live,live_out)
        self.alive_threshold = 1e-1    
        return desired_output    
            
    def forward(self, x):
        if self.is_rejuvenated.item() > 0:
            # print('Conv2d forward() rejuvenation case')
            if not type(x) == list:
                x = [x, x]
            self = self.cuda()    
            live_weight = self.conv.weight.narrow(0, 0, self.live_out.item())
            dead_weight = self.conv.weight.narrow(0, self.live_out.item(), self.dead_out.item())
            live_output = F.conv2d(x[0], live_weight, self.conv.bias, self.conv.stride,
                    self.conv.padding, self.conv.dilation, self.conv.groups)   
            dead_output = F.conv2d(x[1], dead_weight, self.conv.bias, self.conv.stride,
                    self.conv.padding, self.conv.dilation, self.conv.groups)
            output = self.batchNorm(torch.cat([live_output, dead_output], dim=1))
        else:
            output = self.batchNorm(self.conv(x))
        return output

class MySequential(nn.Sequential):

    def inspect(self, live_neurons=None, flops=False):
        if live_neurons is None:
            live_neurons = torch.ByteTensor([1, 1, 1])
        live_params, all_params = 0, 0
        for module in self._modules.values():
            # print(module)
            live_neurons, sub_live_params, sub_all_params = module.inspect(live_neurons, flops=flops)
            live_params += sub_live_params
            all_params += sub_all_params
        
        return live_neurons, live_params, all_params
    
    def rejuvenate(self,expand_rate=None, inp_neurons=None, flops_flag=False):
        if expand_rate is None:
            dummy, live_params, all_params = self.inspect(flops_flag=flops_flag)
            expand_rate = float(all_params)/float(live_params)
        if inp_neurons is None:
            inp_neurons = 3
        for module in self._modules.values():
            inp_neurons = module.rejuvenate(expand_rate,inp_neurons)
        return inp_neurons        

class MySGD(optim.SGD):

    def __init__(self, model, lr, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, sparsity=1e-4, zero_prb=1.0):
        self.model = model
        super(MySGD, self).__init__(model.parameters(), lr)
    
    def inspect(self, flops=False):
        # inspect model
        return self.model.inspect(flops=flops)

class MyReLU(nn.ReLU):

    def inspect(self, live_neurons, flops=False):
        return live_neurons, 0, 0
        
    def rejuvenate(self, expand_rate,inp_neurons):
        return inp_neurons

class MyMaxPool2d(nn.Module):

    def __init__(self, kernel_size, stride=None, padding=0, dilation=1,
            return_indices=False, ceil_mode=False):
        super(MyMaxPool2d, self).__init__()
        self.layer = nn.MaxPool2d(kernel_size, stride, padding, dilation,
                return_indices, ceil_mode)

    def forward(self, x):
        if type(x) == list:
            return [self.layer(ele) for ele in x]
        else:
            return self.layer(x)

    def inspect(self, live_neurons, flops=False):
        return live_neurons, 0, 0
    
    def rejuvenate(self,expand_rate,inp_neurons):
        return inp_neurons
        
class MyAttConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels,kernel_size,stride=1,
                    padding=0,dilation=1,groups=1,bias=True):

        super(MyAttConv2d,self).__init__(in_channels,out_channels,kernel_size,stride,
                                            padding,dilation,groups,bias)

        self.register_buffer('live_inp', torch.tensor(-1, dtype=torch.long))
        self.register_buffer('live_out', torch.tensor(-1, dtype=torch.long))
        self.__nr_conv_ca_mode__ = __nr_conv_ca_mode__
    def forward(self, x):
        weight = self.weight
        if self.live_inp.item() < 0:
            self = self.cuda()
            return F.conv2d(x, self.weight, self.bias, self.stride,
                    self.padding, self.dilation, self.groups)
        live_out, live_inp = self.live_out.item(), self.live_inp.item()
        dead_out, dead_inp = self.out_channels - live_out, self.in_channels - live_inp
        ll_weight = weight.narrow(0, 0, live_out).narrow(1, 0, live_inp)
        rr_weight = weight.narrow(0, live_out, dead_out).narrow(1, live_inp, dead_inp)
        lr_weight = weight.narrow(0, 0, live_out).narrow(1, live_inp, dead_inp)
        rl_weight = weight.narrow(0, live_out, dead_out).narrow(1, 0, live_inp)

        live_x, dead_x = x.narrow(1, 0, live_inp), x.narrow(1, live_inp, dead_inp)
        ll_output = F.conv2d(live_x, ll_weight, self.bias, self.stride,
                self.padding, self.dilation, self.groups)
        rr_output = F.conv2d(dead_x, rr_weight, self.bias, self.stride,
                self.padding, self.dilation, self.groups)
        lr_output = F.conv2d(dead_x, lr_weight, self.bias, self.stride,
                self.padding, self.dilation, self.groups)
        rl_output = F.conv2d(live_x, rl_weight, self.bias, self.stride,
                self.padding, self.dilation, self.groups)

        if self.__nr_conv_ca_mode__:
            ll_output = ll_output + lr_output * 2 * F.sigmoid(ll_output)
            rr_output = rr_output + rl_output * 2 * F.sigmoid(rr_output)
        print(torch.cat([ll_output, rr_output], dim=1).size())
        return torch.cat([ll_output, rr_output], dim=1)
    
    def set_live_inp_and_out(self, live_inp, live_out):
        print(live_inp, live_out)
        self.live_inp.fill_(live_inp)
        self.live_out.fill_(live_out)
        print(self)

class MyConvLinear(nn.Module):
    r"""
    linear layer with neural rejuvenation
    """

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True):
        super(MyConvLinear, self).__init__()
        self.linear = MyAttConv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,\
                padding=padding, dilation=dilation, bias=bias)
        self.kernel_size = kernel_size
        self.linear.__is_linear__ = True
        # to save in state_dict
        self.register_buffer('is_rejuvenated', torch.tensor(0, dtype=torch.long))
        self.register_buffer('live_inp', torch.tensor(-1, dtype=torch.long))
        self.register_buffer('dead_inp', torch.tensor(-1, dtype=torch.long))

    def forward(self, x):
        # print('MyConvLinear forward()')
        if self.is_rejuvenated.item() > 0:
            if type(x) == list:
                x = torch.cat(x, dim=1)
            live_input, dead_input = x.clone(), x.clone()
            dead_input.narrow(1, 0, self.live_inp.item()).zero_()
            live_input.narrow(1, self.live_inp.item(), self.dead_inp.item()).zero_()
            live_input = live_input.view(live_input.size(0), live_input.size(1), 1, 1) # reshape to match weight tensor
            dead_input = dead_input.view(dead_input.size(0), dead_input.size(1), 1, 1) # reshape to match weight tensor
            live_output = self.linear(live_input).unsqueeze(2)
            dead_output = self.linear(dead_input).unsqueeze(2)
            output = torch.cat([live_output, dead_output], 2)
        else:
            # output = self.linear(x.unsqueeze(2).unsqueeze(2)).squeeze(2).squeeze(2)
            
            x = x.view(x.size(0), x.size(1), 1, 1)
            output = self.linear(x).unsqueeze(2)
            output = output.view(output.size(0), output.size(1))
        return output

    def inspect(self, live_neurons, flops):
        self.live_prev = live_neurons
        self.dead_prev = self.live_prev.clone().fill_(1) - self.live_prev
        self.live_mask = self.live_prev.clone().view(1, -1, 1, 1)
        self.dead_mask = self.dead_prev.clone().view(1, -1, 1, 1)
        self.live_mask = self.live_mask.expand(self.linear.weight.size())
        self.dead_mask = self.dead_mask.expand(self.linear.weight.size())
        return live_neurons, 0, 0

    def rejuvenate(self, expand_rate, inp_neurons):
        self.live_inp.fill_(self.live_prev.sum().item())
        # desired_inp = int(self.live_inp.item() * expand_rate)
        desired_inp = inp_neurons
        self.dead_inp.fill_(desired_inp - self.live_inp.item())
        new_linear = MyAttConv2d(desired_inp, self.linear.out_channels, kernel_size=1)
        new_linear.weight.data.normal_(0, 0.01)
        to_copy = self.linear.weight.data[self.live_mask]
        new_linear.weight.data.narrow(1, 0, self.live_inp.item()).copy_(
                to_copy.view(self.linear.out_channels, self.live_inp.item(), 1, 1))
        new_linear.bias.data.copy_(self.linear.bias.data)
        del self.linear
        self.linear = new_linear
        self.linear.__is_linear__ = True
        self.is_rejuvenated.fill_(1)
        return inp_neurons

class MyCrossEntropyLoss(nn.Module):
    r"""
    cross entropy loss layer for neural rejuvenation
    """

    def __init__(self):
        super(MyCrossEntropyLoss, self).__init__()
        self.layer = nn.CrossEntropyLoss()

    def forward(self, inputs, targets):
        if len(inputs.size()) <= 2:
            return self.layer(inputs, targets)
        live_inputs = inputs.narrow(2, 0, 1).squeeze()
        if inputs.size(2) > 1:
            dead_inputs = inputs.narrow(2, 1, 1).squeeze()
            return self.layer(live_inputs, targets) + self.layer(dead_inputs, targets)
        else:
            return self.layer(live_inputs, targets)
            
class MyGlobalAvgPool2d(nn.Module):

    def __init__(self, output_size):
        super(MyGlobalAvgPool2d, self).__init__()
        self.layer = nn.AdaptiveAvgPool2d(output_size)

    def forward(self, x):
        # b x c x h x w
        if len(x.size()) == 4:
            return self.layer(x)
        # b x c x n x h x w
        elif len(x.size()) == 5:
            b, c, n, h, w = x.size()
            out = self.layer(x.view(b, c * n, h, w))
            return out.view(b, c, n)
        else:
            print ('Unsupported input dimension')
            print (x.size())
