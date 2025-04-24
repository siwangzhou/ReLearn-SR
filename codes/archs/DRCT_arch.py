import torch
from torch import nn
from archs.ops.DRCT_common import *
from utils.util import opt_get


class DRCT(nn.Module):
    def __init__(self, opt):
        super(DRCT, self).__init__()
        self.opt = opt

        # self.window_size = kwargs["window_size"]

        self.window_size = opt_get(opt, ['network_G', 'window_size'], 8)
        self.shift_size = self.window_size // 2
        self.overlap_ratio = opt_get(opt, ['network_G', 'overlap_ratio'], 0.5)

        # num_in_ch = kwargs["in_chans"]
        # num_out_ch = kwargs["in_chans"]
        num_in_ch = opt_get(opt, ['network_G', 'in_chans'], 3)
        num_out_ch = opt_get(opt, ['network_G', 'in_chans'], 3)

        num_feat = 64
        # self.img_range = kwargs["img_range"]
        self.img_range = opt_get(opt, ['network_G', 'img_range'], 1.0)
        if num_in_ch == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        # self.upscale = kwargs["upscale"]
        # self.upsampler = kwargs["upsampler"]
        self.upscale = opt_get(opt, ['network_G', 'upscale'], 4)
        self.upsampler = opt_get(opt, ['network_G', 'upsampler'], 'pixelshuffle')

        self.embed_dim = opt_get(opt, ['network_G', 'embed_dim'], 180)

        # ------------------------- 1, shallow feature extraction ------------------------- #
        self.conv_first = nn.Conv2d(num_in_ch, self.embed_dim, 3, 1, 1)

        # ------------------------- 2, deep feature extraction ------------------------- #
        self.depths = opt_get(opt, ['network_G', 'depths'], [6, 6, 6])
        # self.num_layers = len(kwargs["depths"])
        # self.embed_dim = kwargs["embed_dim"]
        self.num_layers = len(self.depths)

        self.ape = False
        self.patch_norm = True
        self.num_features = self.embed_dim
        # self.mlp_ratio = kwargs["mlp_ratio"]
        # self.img_size = kwargs["img_size"]

        self.mlp_ratio = opt_get(opt, ['network_G', 'mlp_ratio'], 2)
        self.img_size = opt_get(opt, ['network_G', 'img_size'], 64)

        patch_size = 1
        norm_layer = nn.LayerNorm
        drop_rate = 0.
        drop_path_rate = 0.1
        attn_drop_rate = 0.
        qkv_bias = True
        qk_scale = None
        gc = 32
        # self.resi_connection = kwargs["resi_connection"]
        # self.depths = kwargs["depths"]
        # self.num_heads =  kwargs["num_heads"]

        self.resi_connection = opt_get(opt, ['network_G', 'resi_connection'], '1conv')

        self.num_heads = opt_get(opt, ['network_G', 'num_heads'], [6, 6, 6])

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=self.img_size,
            patch_size=patch_size,
            in_chans=self.embed_dim,
            embed_dim=self.embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        # self.mlp_ratio = kwargs["mlp_ratio"]
        self.mlp_ratio = opt_get(opt, ['network_G', 'mlp_ratio'], 2)

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            img_size=self.img_size,
            patch_size=patch_size,
            in_chans=self.embed_dim,
            embed_dim=self.embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]  # stochastic depth decay rule

        # build
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = RDG(dim=self.embed_dim, input_resolution=(patches_resolution[0], patches_resolution[1]),
                        num_heads=self.num_heads[i_layer], window_size=self.window_size, depth=0,
                        shift_size=self.window_size // 2, mlp_ratio=self.mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dpr[sum(self.depths[:i_layer]):sum(self.depths[:i_layer + 1])],
                        norm_layer=norm_layer, gc=gc, img_size=self.img_size, patch_size=patch_size)

            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)

        # build the last conv layer in deep feature extraction
        if self.resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(self.embed_dim, self.embed_dim, 3, 1, 1)
        elif self.resi_connection == 'identity':
            self.conv_after_body = nn.Identity()

        # ------------------------- 3, high quality image reconstruction ------------------------- #
        if self.upsampler == 'pixelshuffle':
            # for classical SR
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(self.embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))
            self.upsample = Upsample(self.upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        x_size = (x.shape[2], x.shape[3])

        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x, x_size)

        x = self.norm(x)  # b seq_len c
        x = self.patch_unembed(x, x_size)

        return x

    def forward(self, x):
        # print(f"x_in.data:{x}")
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        if self.upsampler == 'pixelshuffle':
            # for classical SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))

        x = x / self.img_range + self.mean
        # print(f"x_out.data:{x}")
        return x
