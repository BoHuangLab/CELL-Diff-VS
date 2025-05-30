import torch.nn as nn

class CondConvNet(nn.Module):
    def __init__(self, in_channels, cond_out_channels):
        """
        Parameters:
        - in_channels (int): Number of channels per input image.
        - cond_out_channels (list of int): Output channels for each conv layer.
        - num_cell_images (int): Number of comma-separated cell images in config.cell_image.
        """
        super(CondConvNet, self).__init__()
        layers = []

        # First convolution layer
        layers.append(
            nn.Conv3d(
                in_channels, 
                cond_out_channels[0],
                kernel_size=2,
                stride=2,
                padding=0
            )
        )

        # Intermediate layers
        for i in range(len(cond_out_channels) - 1):
            layers.append(nn.SiLU())
            layers.append(
                nn.Conv3d(
                    cond_out_channels[i],
                    cond_out_channels[i + 1],
                    kernel_size=2,
                    stride=2,
                    padding=0
                )
            )

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
