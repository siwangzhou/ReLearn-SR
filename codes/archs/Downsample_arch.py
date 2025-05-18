import torch
import torch.nn as nn
from einops import rearrange
from utils.util import opt_get
import torch.nn.functional as F
from torch.autograd import Function, gradcheck
from torchvision.transforms import ToPILImage, ToTensor
from PIL import Image

from archs.adaptive_gridsampler.adaptive_gridsampler_cuda import forward, backward


class P2P_Conv(nn.Module):
    def __init__(self, opt):
        super(P2P_Conv, self).__init__()
        self.opt = opt
        scale = opt['network_Downsample']['scale']

        if scale ==8:
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=4 * 4 * 3, kernel_size=32, stride=32, padding=0)
            self.a_size = 4
        if scale == 4:
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=8 * 8 * 3, kernel_size=32, stride=32, padding=0)
            self.a_size = 8
        if scale == 2:
            kernel_size = opt['network_Downsample']['kernel_size']
            out_channels = opt['network_Downsample']['out_channels']
            self.a_size = out_channels
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=out_channels * out_channels * 3, kernel_size=kernel_size, stride=kernel_size, padding=0)
        else:
            print("Scale Error,Current scale is %d" % scale)

    def forward(self, x):
        x = self.conv1(x)
        x = rearrange(x, 'b (c a1 a2) h w -> b c (h a1) (w a2)', a1=self.a_size, a2=self.a_size, c=3)

        return x


class Bicubic(nn.Module):
    def __init__(self, opt, scale=4):
        super(Bicubic, self).__init__()
        self.scale = scale
        self.opt = opt

    def forward(self, x):
        x = F.interpolate(x, scale_factor=1 / self.scale, mode='bicubic', align_corners=True)
        return x


class Bicubic_PIL(nn.Module):
    def __init__(self, opt, scale=4):
        super(Bicubic_PIL, self).__init__()
        self.scale = scale
        self.opt = opt
        self.to_pil = ToPILImage()  # 将 Tensor 转换为 PIL 图像
        self.to_tensor = ToTensor()  # 将 PIL 图像转换为 Tensor

    def forward(self, x):
        # 获取输入张量的形状
        b, c, h, w = x.shape

        # 计算下采样后的目标尺寸
        target_h = int(h / self.scale)
        target_w = int(w / self.scale)

        # 初始化一个空列表，用于存储处理后的图像
        output_images = []

        # 遍历批次中的每一张图像
        for i in range(b):
            # 将 Tensor 转换为 PIL 图像
            img = self.to_pil(x[i])

            # 使用 PIL 的 resize 方法进行下采样
            img_resized = img.resize((target_w, target_h), Image.BICUBIC)

            # 将 PIL 图像转换回 Tensor
            img_tensor = self.to_tensor(img_resized)

            # 将处理后的图像添加到列表中
            output_images.append(img_tensor)

        # 将列表中的图像堆叠成一个批次张量
        output = torch.stack(output_images, dim=0).to(x.device)

        return output


class GSM(nn.Module):
    def __init__(self, opt):
        super(GSM, self).__init__()

        self.scale = opt['network_Downsample']['scale']

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=3 // 2)
        self.down_x2_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=2, stride=2, padding=0)
        self.pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.prelu = nn.PReLU()
        self.down_x2_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=3 // 2)

        if self.scale == 8:
            self.down_x2_3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=2, stride=2, padding=0)


    def forward(self, x):
        if self.scale == 2:
            x = self.conv1(x)
            redual = self.pool2d(x)
            out = self.down_x2_1(x)
            x = redual + out
            x = self.conv2(x)

        elif self.scale == 4:
            x = self.conv1(x)
            redual = self.pool2d(x)
            out = self.down_x2_1(x)
            x = redual + out
            redual = self.pool2d(x)
            out = self.down_x2_2(x)
            x = redual + out
            x = self.conv2(x)

        elif self.scale == 8:
            x = self.conv1(x)
            redual = self.pool2d(x)
            out = self.down_x2_1(x)
            x = redual + out
            redual = self.pool2d(x)
            out = self.down_x2_2(x)
            x = redual + out
            redual = self.pool2d(x)
            out = self.down_x2_3(x)
            x = redual + out
            x = self.conv2(x)

        return x



class GridSamplerFunction(Function):
    @staticmethod
    def forward(ctx, img, kernels, offsets_h, offsets_v, offset_unit, padding, downscale_factor):
        assert isinstance(downscale_factor, int)
        assert isinstance(padding, int)

        ctx.padding = padding
        ctx.offset_unit = offset_unit

        b, c, h, w = img.size()
        assert h // downscale_factor == kernels.size(2)
        assert w // downscale_factor == kernels.size(3)

        img = nn.ReflectionPad2d(padding)(img)
        # ctx.save_for_backward(img, kernels, offsets_h, offsets_v)

        output = img.new(b, c, h // downscale_factor, w // downscale_factor).zero_()
        forward(img, kernels, offsets_h, offsets_v, offset_unit, padding, output)
        # 保存必要变量供 backward 使用
        ctx.save_for_backward(img, kernels, offsets_h, offsets_v)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # 从 ctx 中加载前向传播保存的变量
        img, kernels, offsets_h, offsets_v = ctx.saved_tensors
        padding = ctx.padding
        offset_unit = ctx.offset_unit

        # 初始化梯度张量
        grad_img = None
        grad_kernels = torch.zeros_like(kernels, dtype=grad_output.dtype, device=grad_output.device)
        grad_offsets_h = torch.zeros_like(offsets_h, dtype=grad_output.dtype, device=grad_output.device)
        grad_offsets_v = torch.zeros_like(offsets_v, dtype=grad_output.dtype, device=grad_output.device)

        # 调用自定义 CUDA backward 函数
        backward(img, kernels, offsets_h, offsets_v, offset_unit, grad_output, padding,
                 grad_kernels, grad_offsets_h, grad_offsets_v)

        # 返回梯度，None 对应未定义梯度的输入参数
        return None, grad_kernels, grad_offsets_h, grad_offsets_v, None, None, None
        # raise NotImplementedError


class Gridsampler(nn.Module):
    def __init__(self, opt, ds=4, k_size=13):
        super(Gridsampler, self).__init__()
        # self.ds = opt_get(opt, opt['network_Downsample']['scale'], ds)
        # self.k_size = opt_get(opt, opt['network_Downsample']['ksize'], k_size)
        self.ds = ds
        self.k_size = k_size

    def forward(self, img, kernels, offsets_h, offsets_v, offset_unit):
        assert self.k_size ** 2 == kernels.size(1)
        return GridSamplerFunction.apply(img, kernels, offsets_h, offsets_v, offset_unit, self.k_size // 2, self.ds)


class CNN_CR(nn.Module):
    def __init__(self, opt, num_channels=3, num_features=64, num_blocks=10, scale_factor=4):
        """
        参数说明：
          num_channels: 输入图像通道数（通常为 3）
          num_features: 降采样层和中间卷积层的特征通道数（论文中为 64）
          num_blocks: 中间卷积层的层数
          scale_factor: 下采样因子（例如 2 表示 2× 下采样）
        """
        super(CNN_CR, self).__init__()
        self.scale_factor = scale_factor

        # 降采样层：卷积操作，stride = scale_factor
        self.downsampling = nn.Conv2d(num_channels, num_features, kernel_size=3, stride=scale_factor, padding=1)

        # 中间卷积层：每层采用 64 个 3×3 卷积核和 ReLU 激活
        layers = []
        for _ in range(num_blocks):
            layers.append(nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU(inplace=True))
        self.conv_layers = nn.Sequential(*layers)

        # 输出层：生成紧凑分辨率图像，采用 3×3 卷积核
        self.output = nn.Conv2d(num_features, num_channels, kernel_size=3, stride=1, padding=1)

        # 残差分支：直接计算输入图像的 Bicubic 下采样版本
        self.bicubic_down = nn.Upsample(scale_factor=1 / scale_factor, mode='bicubic', align_corners=False)

    def forward(self, x):
        # 计算 Bicubic 下采样版本：F(x)
        bicubic_x = self.bicubic_down(x)

        # CNN 主分支：降采样 -> 多层卷积 -> 输出细节分量
        out = self.downsampling(x)
        out = self.conv_layers(out)
        out = self.output(out)

        # 残差连接：最终输出 = CNN 学习的细节 + Bicubic 下采样结果
        return out + bicubic_x