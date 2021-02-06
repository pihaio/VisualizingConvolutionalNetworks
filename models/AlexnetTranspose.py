import torch.nn as nn
import torchvision.models as models

class AlexnetTranspose(nn.Module):
    """
    alexnet transpose convolution network architecture
        Use nn.ConvTranspose2d in order to make invert alexnet
    Remember:
        exchange the params (in_channels & out_channels) in the nn.Conv2d for nn.ConvTranspose2d
    """
    def __init__(self):
        super(AlexnetTranspose, self).__init__()

        self.features = nn.Sequential(
            nn.MaxUnpool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),

            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),

            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(384, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),

            nn.MaxUnpool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(192, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),

            nn.MaxUnpool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2)),
        )

        self.conv_layer_indices =  [ 0, 3, 6, 8, 10]
        self.dconv_layer_indices = [12, 9, 6, 4, 2 ]

        self.maxpool_indices =   [ 2, 5, 12]
        self.unpooling_indices = [10, 7, 0 ]
        # the indices of alexnet's Conv2d respond to indices of dealexnet's ConvTranspose2d
        self.conv2deconv_indices = {
                0:12, 3:9, 6:6, 8:4, 10:2
        }
        # the indices of alexnet's Maxpool2d respond to indices of dealexnet's MaxUnpool2d
        self.unpool2pool_indices = {
                10:2, 7:5, 0:12
        }
        # init weights, use weight of pytorch's pretrained alexnet to init weights of AlexnetTranspose's ConvTranspose2d
        self.init_weights()

    def init_weights(self):
        alexnet_pretrained = models.alexnet(pretrained=True)
        for idx, layer in enumerate(alexnet_pretrained.features):
            if isinstance(layer, nn.Conv2d):
                self.features[self.conv2deconv_indices[idx]].weight.data = layer.weight.data

    def forward(self, x, layer, pool_locs):
        '''
        according to the indic of alexnet's conv layer to get the indic of alexnettranspose's deconv layer,
        and perform start_idx to end to get the same size of src image.
        Params:
           x: the feature getted from alexnet inference
           layer: the indic of alexnet's conv layer
           pool_locs: the locations of the maxima within each pooling region
        '''
        if layer in self.conv2deconv_indices:
            start_idx = self.conv2deconv_indices[layer]
        else:
            raise ValueError('layer is not a conv feature map')
        for idx in range(start_idx, len(self.features)):
            if isinstance(self.features[idx], nn.MaxUnpool2d):
                x = self.features[idx]\
                (x, pool_locs[self.unpool2pool_indices[idx]])
            else:
                x = self.features[idx](x)
        return x



if __name__ == '__main__':
    model = AlexnetTranspose()
    print(model)