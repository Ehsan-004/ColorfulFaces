import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels,
                 norm_layer=nn.BatchNorm2d, 
                 activation=nn.ReLU(inplace=True)):
        super(DoubleConv, self).__init__()        

        layers = []
        # Conv → Norm → Act (first layer)
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False))
        if norm_layer is not None:
            layers.append(norm_layer(out_channels))
        if activation is not None:
            layers.append(activation)

        # Conv → Norm → Act (second layer)
        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False))
        if norm_layer is not None:
            layers.append(norm_layer(out_channels))
        if activation is not None:
            # create a new object of activation class (not to share the layer)
            layers.append(type(activation)())

        self.double_conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, 
                 norm_layer=nn.BatchNorm2d, 
                 activation=nn.ReLU(inplace=False)):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels, norm_layer, activation)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, 
                 up_mode='upconv', 
                 norm_layer=nn.BatchNorm2d, 
                 activation=nn.ReLU(inplace=False)):
        """up sample the input, concatenates it with the skip connection and computes the convolution

        Args:
            in_channels (int): number of channels from down layer
            out_channels (int): number of output channels
            up_mode (str, optional): _description_. Defaults to 'upconv'.
            norm_layer (_type_, optional): _description_. Defaults to nn.BatchNorm2d.
            activation (_type_, optional): _description_. Defaults to nn.ReLU(inplace=False).

        Raises:
            ValueError: _description_
        """
        super(Up, self).__init__()

        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            # devided by 2 because of concatenate
        elif up_mode == 'interp':
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
            )
        else:
            raise ValueError("up_mode should be 'upconv' or 'interp'")

        self.conv = DoubleConv(in_channels, out_channels, norm_layer, activation)

    def forward(self, x1, x2):
        # x1 = input from previous layer (low-res)
        # x2 = skip connection from encoder (high-res)
        # x2.shape[1] == x1.shape[1] // 2
        x1 = self.up(x1)

        # if spatial sizes are not the same, add padding:
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        # [pad_left, pad_right, pad_top, pad_bottom]
        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                    diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)  # concat channels
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        return torch.nn.Tanh(self.conv(x))


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, 
                 depth=4, base_channels=64, 
                 norm_layer=nn.BatchNorm2d, activation=nn.ReLU(inplace=False)):
        """_summary_

        Args:
            n_channels (int): number of input channels
            n_classes (int): number of output channels
            depth (int, optional): _description_. Defaults to 4.
            base_channels (int, optional): _description_. Defaults to 64.
            norm_layer (_type_, optional): _description_. Defaults to nn.BatchNorm2d.
            activation (_type_, optional): _description_. Defaults to nn.ReLU(inplace=False).
        """
        super(UNet, self).__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.depth = depth

        # ================
        # Encoder
        # ================
        
        # convert input channels to base channels (e.g: 3 -> 64)
        self.inc = DoubleConv(n_channels, base_channels, norm_layer, activation)
        self.downs = nn.ModuleList()
        channels = base_channels
        for _ in range(depth):
            self.downs.append(Down(channels, channels*2, norm_layer, activation))  # spatial // 2, channels * 2
            channels *= 2

        # ================
        # Decoder
        # ================
        
        self.ups = nn.ModuleList()
        for _ in range(depth):
            self.ups.append(Up(channels, channels//2, norm_layer=norm_layer, activation=activation))
            channels //= 2

        # ================
        # Output
        # ================
        
        self.outc = OutConv(base_channels, n_classes)

    def forward(self, x):
        x_skip = []

        # ================
        # Encoder
        # ================
        
        # convert channels to base channel
        x1 = self.inc(x)
        x_skip.append(x1)
        x_temp = x1
        for down in self.downs:
            x_temp = down(x_temp)
            x_skip.append(x_temp)
        
        x_skip.pop()
        
        # ================
        # Decoder
        # ================
        
        for up in self.ups:
            skip = x_skip.pop()  # picking skip connection from the end of the list
            x_temp = up(x_temp, skip)

        # ================
        # Output
        # ================
        
        logits = self.outc(x_temp)
        return logits
