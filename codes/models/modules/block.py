from collections import OrderedDict
import torch
import torch.nn as nn

import torch.nn.functional as F

from . import adaptive_norm as AN

####################
# Basic blocks
####################


def act(act_type, inplace=True, neg_slope=0.2, n_prelu=1):
    # helper selecting activation
    # neg_slope: for leakyrelu and init of prelu
    # n_prelu: for p_relu num_parameters
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer


def norm(norm_type, nc):
    # helper selecting normalization layer
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm_type == 'instance':
        layer = nn.InstanceNorm2d(nc, affine=True)
    else:
        raise NotImplementedError('normalization layer [{:s}] is not found'.format(norm_type))
    return layer


def pad(pad_type, padding):
    # helper selecting padding layer
    # if padding is 'zero', do by conv layers
    pad_type = pad_type.lower()
    if padding == 0:
        return None
    if pad_type == 'reflect':
        layer = nn.ReflectionPad2d(padding)
    elif pad_type == 'replicate':
        layer = nn.ReplicationPad2d(padding)
    else:
        raise NotImplementedError('padding layer [{:s}] is not implemented'.format(pad_type))
    return layer


def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding


class ConcatBlock(nn.Module):
    # Concat the output of a submodule to its input
    def __init__(self, submodule):
        super(ConcatBlock, self).__init__()
        self.sub = submodule

    def forward(self, x):
        output = torch.cat((x, self.sub(x)), dim=1)
        return output

    def __repr__(self):
        tmpstr = 'Identity .. \n|'
        modstr = self.sub.__repr__().replace('\n', '\n|')
        tmpstr = tmpstr + modstr
        return tmpstr


class ShortcutBlock(nn.Module):
    #Elementwise sum the output of a submodule to its input
    def __init__(self, submodule):
        super(ShortcutBlock, self).__init__()
        self.sub = submodule

    def forward(self, x):
        output = x + self.sub(x)
        return output

    def __repr__(self):
        tmpstr = 'Identity + \n|'
        modstr = self.sub.__repr__().replace('\n', '\n|')
        tmpstr = tmpstr + modstr
        return tmpstr


def sequential(*args):
    # Flatten Sequential. It unwraps nn.Sequential.
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


def conv_block(in_nc, out_nc, kernel_size, stride=1, dilation=1, groups=1, bias=True,
               pad_type='zero', norm_type=None, act_type='relu', mode='CNA', ada_ksize=None):
    '''
    Conv layer with padding, normalization, activation
    mode: CNA --> Conv -> Norm -> Act
        NAC --> Norm -> Act --> Conv (Identity Mappings in Deep Residual Networks, ECCV16)
    '''
    assert mode in ['CNA', 'NAC', 'CNAC', 'NCA'], 'Wong conv mode [{:s}]'.format(mode)
    padding = get_valid_padding(kernel_size, dilation)
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    padding = padding if pad_type == 'zero' else 0

    c = nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding,
                  dilation=dilation, bias=bias, groups=groups)
    a = act(act_type) if act_type else None
    if mode == 'CNA':
        if norm_type == 'adaptive_conv_res':
            n = AN.AdaptiveConvResNorm(out_nc, ada_ksize)
        else:
            n = norm(norm_type, out_nc) if norm_type else None
        return sequential(p, c, n, a)
    elif mode == 'NAC':
        if norm_type is None and act_type is not None:
            a = act(act_type, inplace=False)
            # Important!
            # input----ReLU(inplace)----Conv--+----output
            #        |________________________|
            # inplace ReLU will modify the input, therefore wrong output
        n = norm(norm_type, in_nc) if norm_type else None
        return sequential(n, a, p, c)
    elif mode == 'NCA':
        if norm_type == 'adaptive_conv_res':
            n = AN.AdaptiveConvResNorm(in_nc, ada_ksize)
        else:
            n = norm(norm_type, in_nc) if norm_type else None
        return sequential(n, p, c, a)


def computing_mean_variance(fea_list):
    fea_flatten_list = [fea.view(64, -1) for fea in fea_list]
    fea_flatten_cat = torch.cat(fea_flatten_list, 1)
    fea_mean = torch.mean(fea_flatten_cat, 1)
    fea_var = torch.var(fea_flatten_cat, 1)
    return fea_mean, fea_var

####################
# Useful blocks
####################


class ResNetBlock(nn.Module):
    '''
    ResNet Block, 3-3 style
    with extra residual scaling used in EDSR
    (Enhanced Deep Residual Networks for Single Image Super-Resolution, CVPRW 17)
    '''

    def __init__(self, in_nc, mid_nc, out_nc, kernel_size=3, stride=1, dilation=1, groups=1, bias=True,
                 pad_type='zero', norm_type=None, act_type='relu', mode='CNA', res_scale=1, ada_ksize=None):
        super(ResNetBlock, self).__init__()
        conv0 = conv_block(in_nc, mid_nc, kernel_size, stride, dilation, groups, bias, pad_type,
                           norm_type, act_type, mode, ada_ksize)
        if mode == 'CNA':
            act_type = None
        if mode == 'CNAC':  # Residual path: |-CNAC-|
            act_type = None
            norm_type = None
        conv1 = conv_block(mid_nc, out_nc, kernel_size, stride, dilation, groups, bias, pad_type,
                           norm_type, act_type, mode, ada_ksize)
        # if in_nc != out_nc:
        #     self.project = conv_block(in_nc, out_nc, 1, stride, dilation, 1, bias, pad_type, \
        #         None, None)
        #     print('Need a projecter in ResNetBlock.')
        # else:
        #     self.project = lambda x:x
        self.res = sequential(conv0, conv1)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.res(x).mul(self.res_scale)
        return x + res


class TwoStreamSRResNet(nn.Module):
    """
    residual block for modulate resnet
    """
    def __init__(self, in_nc, mid_nc, out_nc, kernel_size=3, stride=1, dilation=1, groups=1, \
                 bias=True, pad_type='zero', norm_type='gate', act_type='relu', mode='CNA', res_scale=1,
                 gate_conv_bias=True, ada_ksize=None, input_dim=3):
        super(TwoStreamSRResNet, self).__init__()

        self.conv_block0 = conv_block(in_nc, mid_nc, kernel_size, stride, dilation, groups, bias, pad_type,
                                      norm_type=None, act_type=None, mode=mode)
        if norm_type == 'sft':
            self.gate_norm0 = AN.GateNonLinearLayer(input_dim, conv_bias=gate_conv_bias)
        elif norm_type == 'sft_conv':
            self.gate_norm0 = AN.MetaLayer(input_dim, conv_bias=gate_conv_bias, kernel_size=ada_ksize)


        self.conv_block1 = conv_block(mid_nc, out_nc, kernel_size, stride, dilation, groups, bias, pad_type,
                                      norm_type=None, act_type=None, mode=mode)
        if norm_type == 'sft':
            self.gate_norm1 = AN.GateNonLinearLayer(input_dim, conv_bias=gate_conv_bias)
        elif norm_type == 'sft_conv':
            self.gate_norm1 = AN.MetaLayer(input_dim, conv_bias=gate_conv_bias, kernel_size=ada_ksize)

    def forward(self, x):
        fea = self.conv_block0(x[0])
        fea = F.relu(self.gate_norm0((fea, x[1])), inplace=True)
        fea = self.conv_block1(fea)
        res = self.gate_norm1((fea, x[1]))
        return x[0] + res, x[1]


class CondResNetBlock(nn.Module):
    '''
    ResNet Block, 3-3 style
    with extra residual scaling used in EDSR
    (Enhanced Deep Residual Networks for Single Image Super-Resolution, CVPRW 17)
    '''

    def __init__(self, in_nc, mid_nc, out_nc, kernel_size=3, stride=1, dilation=1, groups=1,
                 bias=True, num_classes=1, ada_ksize=None, norm_type=None, act_type='relu'):
        super(CondResNetBlock, self).__init__()
        padding = get_valid_padding(kernel_size, dilation)
        self.conv0 = nn.Conv2d(in_nc, mid_nc, kernel_size=kernel_size, stride=stride, padding=padding,
                               dilation=dilation, bias=bias, groups=groups)

        if norm_type == 'cond_adaptive_conv_res':
            self.cond_adaptive0 = AN.CondAdaptiveConvResNorm(mid_nc, num_classes=num_classes)
        elif norm_type == "interp_adaptive_conv_res":
            self.cond_adaptive0 = AN.InterpAdaptiveResNorm(mid_nc, ada_ksize)
        elif norm_type == "cond_instance":
            self.cond_adaptive0 = AN.CondInstanceNorm2d(mid_nc, num_classes=num_classes)
        elif norm_type == "cond_transform_res":
            self.cond_adaptive0 = AN.CondResTransformer(mid_nc, ada_ksize, num_classes=num_classes)

        self.act = act(act_type)

        self.conv1 = nn.Conv2d(mid_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding,
                               dilation=dilation, bias=bias, groups=groups)

        if norm_type == 'cond_adaptive_conv_res':
            self.cond_adaptive1 = AN.CondAdaptiveConvResNorm(out_nc, num_classes=num_classes)
        elif norm_type == "interp_adaptive_conv_res":
            self.cond_adaptive1 = AN.InterpAdaptiveResNorm(out_nc, ada_ksize)
        elif norm_type == "cond_instance":
            self.cond_adaptive1 = AN.CondInstanceNorm2d(out_nc, num_classes=num_classes)
        elif norm_type == "cond_transform_res":
            self.cond_adaptive1 = AN.CondResTransformer(out_nc, ada_ksize, num_classes=num_classes)

    def forward(self, x):
        fea = self.conv0(x[0])
        fea1 = self.cond_adaptive0(fea, x[1])
        fea2 = self.act(fea1)
        fea3 = self.conv1(fea2)
        fea4 = self.cond_adaptive1(fea3, x[1])
        # res
        return fea4+x[0], x[1]


class AdaptiveResNetBlock(nn.Module):
    """
    residual block for modulate resnet
    """
    def __init__(self, in_nc, mid_nc, out_nc, kernel_size=3, stride=1, dilation=1, groups=1,
                 bias=True, res_scale=1):
        super(AdaptiveResNetBlock, self).__init__()

        padding = get_valid_padding(kernel_size, dilation)
        self.conv0 = nn.Conv2d(in_nc, mid_nc, kernel_size=kernel_size, stride=stride, padding=padding,
                                dilation=dilation, bias=bias, groups=groups)
        self.batch_norm0 = nn.BatchNorm2d(mid_nc, affine=True, track_running_stats=True, momentum=0)
        self.conv1 = nn.Conv2d(mid_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding,
                               dilation=dilation, bias=bias, groups=groups)
        self.batch_norm1 = nn.BatchNorm2d(out_nc, affine=True, track_running_stats=True, momentum=0)

        self.res_scale = res_scale


    def forward(self, x):
        fea0_list = [self.conv0(data) for data in x]
        fea0_mean, fea0_var = computing_mean_variance(fea0_list)

        batch_norm0_dict = self.batch_norm0.state_dict()
        batch_norm0_dict['running_mean'] = fea0_mean
        batch_norm0_dict['running_var'] = fea0_var
        self.batch_norm0.load_state_dict(batch_norm0_dict)

        # generator
        def _batch_norm0_forward(fea_list):
            for fea in fea_list:
                yield F.relu(self.batch_norm0(fea), inplace=True)

        fea1_list = [self.conv1(batchnorm0) for batchnorm0 in _batch_norm0_forward(fea0_list)]
        fea1_mean, fea1_var = computing_mean_variance(fea1_list)

        batch_norm1_dict = self.batch_norm1.state_dict()
        batch_norm1_dict['running_mean'] = fea1_mean
        batch_norm1_dict['running_var'] = fea1_var
        self.batch_norm1.load_state_dict(batch_norm1_dict)

        batchnorm1_list = [self.batch_norm1(fea1)+data for fea1, data in zip(fea1_list, x)]

        return batchnorm1_list


class ResidualDenseBlock_5C(nn.Module):
    '''
    Residual Dense Block
    style: 5 convs
    The core module of paper: (Residual Dense Network for Image Super-Resolution, CVPR 18)
    '''

    def __init__(self, nc, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
            norm_type=None, act_type='leakyrelu', mode='CNA'):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = conv_block(nc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, \
            norm_type=norm_type, act_type=act_type, mode=mode)
        self.conv2 = conv_block(nc+gc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, \
            norm_type=norm_type, act_type=act_type, mode=mode)
        self.conv3 = conv_block(nc+2*gc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, \
            norm_type=norm_type, act_type=act_type, mode=mode)
        self.conv4 = conv_block(nc+3*gc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, \
            norm_type=norm_type, act_type=act_type, mode=mode)
        if mode == 'CNA':
            last_act = None
        else:
            last_act = act_type
        self.conv5 = conv_block(nc+4*gc, nc, 3, stride, bias=bias, pad_type=pad_type, \
            norm_type=norm_type, act_type=last_act, mode=mode)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat((x, x1), 1))
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5.mul(0.2) + x


class RRDB(nn.Module):
    '''
    Residual in Residual Dense Block
    (ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks)
    '''

    def __init__(self, nc, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
            norm_type=None, act_type='leakyrelu', mode='CNA'):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nc, kernel_size, gc, stride, bias, pad_type, \
            norm_type, act_type, mode)
        self.RDB2 = ResidualDenseBlock_5C(nc, kernel_size, gc, stride, bias, pad_type, \
            norm_type, act_type, mode)
        self.RDB3 = ResidualDenseBlock_5C(nc, kernel_size, gc, stride, bias, pad_type, \
            norm_type, act_type, mode)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out.mul(0.2) + x


####################
# Upsampler
####################


def pixelshuffle_block(in_nc, out_nc, upscale_factor=2, kernel_size=3, stride=1, bias=True, \
                        pad_type='zero', norm_type=None, act_type='relu', ada_ksize=None):
    '''
    Pixel shuffle layer
    (Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional
    Neural Network, CVPR17)
    '''
    conv = conv_block(in_nc, out_nc * (upscale_factor ** 2), kernel_size, stride, bias=bias,
                      pad_type=pad_type, norm_type=norm_type, act_type=None, ada_ksize=ada_ksize)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)

    # n = norm(norm_type, out_nc) if norm_type else None
    a = act(act_type) if act_type else None
    return sequential(conv, pixel_shuffle, a)
    # return sequential(conv, pixel_shuffle, n, a)


def upconv_blcok(in_nc, out_nc, upscale_factor=2, kernel_size=3, stride=1, bias=True, \
                pad_type='zero', norm_type=None, act_type='relu', mode='nearest'):
    # Up conv
    # described in https://distill.pub/2016/deconv-checkerboard/
    upsample = nn.Upsample(scale_factor=upscale_factor, mode=mode)
    conv = conv_block(in_nc, out_nc, kernel_size, stride, bias=bias, \
                        pad_type=pad_type, norm_type=norm_type, act_type=act_type)
    return sequential(upsample, conv)

