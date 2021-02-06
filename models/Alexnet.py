import torch.nn as nn
import torchvision.models as models
from collections import OrderedDict


class Alexnet(nn.Module):
    """
    Alexnet convolution network architecture
    """

    def __init__(self, num_cls=1000):
        """
        Input
            number of class, default is 1k.
        """
        super(Alexnet, self).__init__()

        # make use of pytorch's alexnet implementation, get its module:feature & classifier
        pytorch_alexnet = models.alexnet(pretrained=True)

        self.features = pytorch_alexnet.features
        self.classifier = pytorch_alexnet.classifier

        # add nn.Softmax to get confidence of different class.
        self.softmax = nn.Sequential(
            nn.Softmax(dim=1)
        )

        # feature maps to save inference feature map
        self.feature_maps = OrderedDict()
        # switch variables to save the locations of the maxima within each pooling region
        self.pool_locs = OrderedDict()

        # make return_indices true for nn.Maxpool2d to return the locations of the maxima within each pooling region
        for idx, layer in enumerate(self.features):
            if isinstance(layer, nn.MaxPool2d):
                self.features[idx] = nn.MaxPool2d(kernel_size=layer.kernel_size,stride=layer.stride,\
                                                  padding=layer.padding,dilation=layer.dilation,\
                                                  return_indices=True)

        self.conv_layer_indices = [0,3,6,8,10]
        self.pool_indices = [2,5,12]

    def forward(self, x):

        for idx, layer in enumerate(self.features):
            if isinstance(layer, nn.MaxPool2d):
                x, location = layer(x)
            else:
                x = layer(x)

        x = x.view(x.size()[0], -1)
        x = self.classifier(x)
        output = self.softmax(x)
        return output


if __name__ == '__main__':
    model = Alexnet()
    print(model)
