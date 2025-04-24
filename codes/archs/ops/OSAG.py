import torch.nn as nn
from archs.ops.esa import ESA
from utils.util import opt_get


class OSAG(nn.Module):
    def __init__(self, opt, channel_num=64):
        super(OSAG, self).__init__()

        ffn_bias = opt_get(opt, ['network_G', 'ffn_bias'], False)
        window_size = opt_get(opt, ['network_G', 'window_size'], 0)
        pe = opt_get(opt, ['network_G', 'pe'], False)
        # , bias=True, block_num=4
        bias = opt_get(opt, ['network_G', 'bias'], True)
        block_num = opt_get(opt, ['network_G', 'block_num'], 4)


        block_script_name = opt_get(opt, ['network_G', 'block_script_name'], 'OSA')
        block_class_name = opt_get(opt, ['network_G', 'block_class_name'], 'OSA_Block')

        script_name = "archs.ops." + block_script_name
        package = __import__(script_name, fromlist=True)
        block_class = getattr(package, block_class_name)
        group_list = []
        for _ in range(block_num):
            temp_res = block_class(channel_num, bias, ffn_bias=ffn_bias, window_size=window_size, with_pe=pe)
            group_list.append(temp_res)
        group_list.append(nn.Conv2d(channel_num, channel_num, 1, 1, 0, bias=bias))
        self.residual_layer = nn.Sequential(*group_list)
        esa_channel = max(channel_num // 4, 16)
        self.esa = ESA(esa_channel, channel_num)

    def forward(self, x, ):
        out = self.residual_layer(x)
        out = out + x
        return self.esa(out)
