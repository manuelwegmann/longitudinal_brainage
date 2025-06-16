import torch.nn as nn
import torch


class EncoderBlock3D(nn.Module):

    def __init__(self, in_num_ch, out_num_ch, kernel_size=3, dropout=0):
        super(EncoderBlock3D, self).__init__()
        self.conv = nn.Sequential(
                        nn.Conv3d(in_num_ch, out_num_ch, kernel_size=kernel_size, padding=kernel_size // 2), #convolution
                        nn.BatchNorm3d(out_num_ch), #batch normalization
                        nn.LeakyReLU(0.2, inplace=True), #Leaky ReLU activation
                        nn.Dropout3d(dropout), #dropout
                        nn.AvgPool3d(2))
        self.init_model()

    def init_model(self): #weight initialization
        for layer in self.conv.children():
            if isinstance(layer, nn.Conv3d):
                for name, weight in layer.named_parameters():
                    if 'weight' in name:
                        nn.init.kaiming_normal_(weight)
                    if 'bias' in name:
                        nn.init.constant_(weight, 0.0)

    def forward(self, x): #forqard pass acccoring to __init__ above
        return self.conv(x)
    


class Encoder3D(nn.Module):
    """
    Building multiple convolutional blocks together for feature extraction.
    """
    def __init__(self, in_num_ch, num_block, inter_num_ch, kernel_size, dropout):
        """
        inter_num_ch: base number of output channels for the first block.
        """
        
        super(Encoder3D, self).__init__()

        conv_blocks = []
        for i in range(num_block):
            if i == 0: # initial block
                conv_blocks.append(EncoderBlock3D(in_num_ch, inter_num_ch, kernel_size=kernel_size, dropout=dropout))
            elif i == (num_block-1): # last block: compress features back to inter_num_ch as bottleneck
                print("Last block has number of input channels:", inter_num_ch * (2 ** (i - 1)))
                print("Last block has number of output channels:", inter_num_ch)
                conv_blocks.append(EncoderBlock3D(inter_num_ch * (2 ** (i - 1)), inter_num_ch, kernel_size=kernel_size, dropout=dropout))
            else:
                conv_blocks.append(EncoderBlock3D(inter_num_ch * (2 ** (i - 1)), inter_num_ch * (2 ** (i)), kernel_size=kernel_size, dropout=dropout))

        self.conv_blocks = nn.Sequential(*conv_blocks)


    def forward(self, x): #feed forward through all convolutional blocks

        for cb in self.conv_blocks:
            x = cb(x)

        return x
    

class CNNbasic3D(nn.Module): #todo: add conv_act and dropout arguments
    def __init__(self, inputsize = [128,128,128], channels = 1, n_of_blocks = 4, initial_channel = 16, kernel_size = 3, dropout = 0, additional_feature = 0):
        super(CNNbasic3D, self).__init__()

        self.feature_image = (torch.tensor(inputsize) / (2**(n_of_blocks)))
        self.feature_channel = initial_channel
        self.encoder = Encoder3D(in_num_ch=channels, num_block=n_of_blocks, inter_num_ch=initial_channel, kernel_size=kernel_size, dropout=dropout)
        
        self.flattened_feature_size = int((self.feature_channel * (self.feature_image.prod()).type(torch.int).item()))

        self.linear = nn.Sequential(
            nn.Linear(self.flattened_feature_size + additional_feature, 2048),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.shape[0], self.flattened_feature_size)
        y = self.linear(x)
        return y
    

def get_backbone(args = None):
    n_of_meta = len(args.optional_meta)

    backbone = CNNbasic3D(inputsize=args.image_size, channels=args.image_channel, n_of_blocks=args.n_of_blocks, initial_channel= args.initial_channel, kernel_size=args.kernel_size, dropout=args.dropout, additional_feature = n_of_meta)
    linear = backbone.linear
    backbone.linear = nn.Identity()

    return backbone, linear
    


class LILAC_plus(nn.Module):
    """
    Args:
        image_size: desired size of the input image
        image_channel: number of channels in the input image
        n_of_blocks: number of convolutional blocks
        initial_channel: number of feature maps after first and last conv block
        kernel_size: size of the convolutional kernel
        dropout: dropout rate for the convolutional layers
        optional_meta: additional features to be used in the linear layer
    """
    def __init__(self, args):
        super().__init__()
        self.backbone, self.linear = get_backbone(args)
        self.optional_meta = len(args.optional_meta)>0
        self.output_activation = nn.Softplus()

    def forward(self, x1, x2, meta = None):
        f = self.backbone(x2) - self.backbone(x1)
        if not self.optional_meta:
            return self.output_activation(self.linear(f))
        else:
            m = meta
            f = torch.concat((f, m), 1)
            return self.output_activation(self.linear(f))