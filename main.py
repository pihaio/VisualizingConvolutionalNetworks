import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt

from functools import partial
from torchvision.transforms import transforms

from utils import decode_predictions

from models.Alexnet import Alexnet
from models.AlexnetTranspose import AlexnetTranspose

def LoadImages(img_path):
    # imread from img_path
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))

    # pytorch must normalize the pic by
    # mean = [0.485, 0.456, 0.406]
    # std = [0.229, 0.224, 0.225]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img = transform(img)
    img.unsqueeze_(0)
    # img_s = img.numpy()
    # img_s = np.transpose(img_s, (1, 2, 0))
    # cv2.imshow("test img", img_s)
    # cv2.waitKey()
    return img


def MakeHook(model):
    """
    make hook for feature map
    """

    def hook(module, input, output, key):
        if isinstance(module, nn.MaxPool2d):
            model.feature_maps[key] = output[0]
            model.pool_locs[key] = output[1]
        else:
            model.feature_maps[key] = output

    for idx, layer in enumerate(model._modules.get('features')):
        # _modules returns an OrderedDict
        layer.register_forward_hook(partial(hook, key=idx))

def MakeInvertImage(new_feat_map, top_value, top_indices, conv, deconv):

    num_feat = conv.feature_maps[layer].shape[1]
    top_indices_list = top_indices.numpy().tolist()
    top_value_list = top_value.numpy().tolist()
    for index in range(num_feat):
        if index in top_indices_list:
            choose_map = new_feat_map[0, index, :, :]
            max_activation = torch.max(choose_map)
            choose_map = torch.where(choose_map == max_activation,
                                     choose_map,
                                     torch.zeros(choose_map.shape)
                                     )
            new_feat_map[0, index, :, :] = choose_map

        else:
            new_feat_map[:, index, :, :] = 0

    deconv_output = deconv(new_feat_map, layer, conv.pool_locs)

    new_img = deconv_output.data.numpy()[0].transpose(1, 2, 0)  # (H, W, C)
    # normalize
    new_img = (new_img - new_img.min()) / (new_img.max() - new_img.min()) * 255
    new_img = new_img.astype(np.uint8)

    return new_img


def VisualizeFeature(layer, conv, deconv):
    """
    Visualize the layer deconv result
    """

    feature_map_channels = conv.feature_maps[layer].shape[1]
    print('feature_map_channels: ', feature_map_channels)

    # set other feature map activations to zero
    new_feat_map = conv.feature_maps[layer].clone()

    # get max activation of per channel
    max_activation_per_channel = []
    for i in range(0, feature_map_channels):
        choose_map = new_feat_map[0, i, :, :]
        activation = torch.max(choose_map)
        max_activation_per_channel.append(activation.item())

    print('max_activation_per_channel: ', max_activation_per_channel)
    # get top 9 activations and its related indic.
    pred = torch.tensor(max_activation_per_channel)
    top_value, top_indices = torch.topk(pred, 9)
    print('top_value: ', top_value)
    print('top_indices: ', top_indices)

    invert_image = MakeInvertImage(new_feat_map, top_value, top_indices, conv, deconv)

    img_name = str(layer) + '_' + '.jpg'
    #cv2.imwrite(img_name, invert_image)
    return invert_image


if __name__ == '__main__':

    img_path = './data/dog.jpg'

    # forward processing
    img = LoadImages(img_path)
    alexnet_conv = Alexnet()
    alexnet_conv.eval()
    MakeHook(alexnet_conv)
    conv_output = alexnet_conv(img)
    pool_locs = alexnet_conv.pool_locs
    print('Predicted:', decode_predictions(conv_output, top=3)[0])

    # backward processing
    alexnet_deconv = AlexnetTranspose()
    alexnet_deconv.eval()

    plt.figure(num=None, figsize=(16, 12), dpi=80)
    # show src image
    plt.subplot(2, 3, 1)
    plt.title('original picture')
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    plt.imshow(img)

    for idx, layer in enumerate(alexnet_conv.conv_layer_indices):
        plt.subplot(2, 3, idx + 2)
        invert_image = VisualizeFeature(layer, alexnet_conv, alexnet_deconv)
        plt.title(f'{layer} layer')
        plt.imshow(invert_image)

    plt.show()