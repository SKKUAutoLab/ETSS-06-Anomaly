import torch
import torch.nn.functional as F

def fuse_conv_and_bn(conv, bn):
        """Fuse Conv2d() and BatchNorm2d() layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/."""
        fusedconv = (
            torch.nn.Conv2d(
                conv.in_channels,
                conv.out_channels,
                kernel_size=conv.kernel_size,
                stride=conv.stride,
                padding=conv.padding,
                dilation=conv.dilation,
                groups=conv.groups,
                bias=True,
            )
            .requires_grad_(False)
            .to(conv.weight.device)
        )

        # Prepare filters
        w_conv = conv.weight.clone().view(conv.out_channels, -1)
        w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
        fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

        # Prepare spatial bias
        b_conv = torch.zeros(conv.weight.shape[0], device=conv.weight.device) if conv.bias is None else conv.bias
        b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
        fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

        return fusedconv

class ConvBlock(torch.nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()

        self.conv1 = torch.nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode='zeros', bias=False)

        self.conv2 = torch.nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.relu = torch.nn.LeakyReLU(negative_slope=0.1, inplace=False)

        self.norm1 = torch.nn.BatchNorm2d(out_planes)

        self.norm2 = torch.nn.BatchNorm2d(out_planes)

        # self.dropout = torch.nn.Dropout(p=0.1)

    def forward(self, x):

        # x = self.dropout(x)

        x = self.relu(self.norm1(self.conv1(x)))
        x = self.relu(self.norm2(self.conv2(x)))
        # x = self.relu(self.conv1(x))
        # x = self.relu(self.conv2(x))

        return x

    def forward_fuse(self, x):

        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))

        return x


class CNNEncoder(torch.nn.Module):
    def __init__(self, feature_dim_s16, context_dim_s16, feature_dim_s8, context_dim_s8):
        super(CNNEncoder, self).__init__()

        self.block_8_1 = ConvBlock(3, feature_dim_s8 * 2, kernel_size=8, stride=4, padding=2)

        self.block_8_2 = ConvBlock(3, feature_dim_s8, kernel_size=6, stride=2, padding=2)

        self.block_cat_8 = ConvBlock(feature_dim_s8 * 3, feature_dim_s8 + context_dim_s8, kernel_size=3, stride=1, padding=1)

        self.block_16_1 = ConvBlock(3, feature_dim_s16, kernel_size=6, stride=2, padding=2)

        self.block_8_16 = ConvBlock(feature_dim_s8 + context_dim_s8, feature_dim_s16, kernel_size=6, stride=2, padding=2)

        self.block_cat_16 = ConvBlock(feature_dim_s16 * 2, feature_dim_s16 + context_dim_s16 - 2, kernel_size=3, stride=1, padding=1)

    def init_pos(self, batch_size, height, width, device, amp):
        ys, xs = torch.meshgrid(torch.arange(height, dtype=torch.half if amp else torch.float, device=device), torch.arange(width, dtype=torch.half if amp else torch.float, device=device), indexing='ij')
        ys = (ys-height/2)
        xs = (xs-width/2)
        pos = torch.stack([ys, xs])
        return pos[None].repeat(batch_size,1,1,1)

    def init_bhwd(self, batch_size, height, width, device, amp):
        self.pos_s16 = self.init_pos(batch_size, height, width, device, amp)

    def forward(self, img):

        img = F.avg_pool2d(img, kernel_size=2, stride=2)
        x_8 = self.block_8_1(img)

        img = F.avg_pool2d(img, kernel_size=2, stride=2)
        x_8_2 = self.block_8_2(img)

        x_8 = self.block_cat_8(torch.cat([x_8, x_8_2], dim=1))

        img = F.avg_pool2d(img, kernel_size=2, stride=2)
        x_16 = self.block_16_1(img)

        x_16_2 = self.block_8_16(x_8)

        x_16 = self.block_cat_16(torch.cat([x_16, x_16_2], dim=1))

        x_16 = torch.cat([x_16, self.pos_s16], dim=1)

        return x_16, x_8
