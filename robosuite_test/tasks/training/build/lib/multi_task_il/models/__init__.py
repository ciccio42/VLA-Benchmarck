import torch
import cv2
import numpy as np


def get_model(name):
    if name == 'resnet':
        from .basic_embedding import ResNetFeats
        return ResNetFeats
    elif name == 'vgg':
        from .basic_embedding import VGGFeats
        return VGGFeats
    else:
        raise NotImplementedError


def get_class_activation_map(feature_map: np.array, input_image: np.array):
    # Resize the feature map to match the input image size
    resized_feature_map = cv2.resize(
        feature_map, (input_image.shape[0], input_image.shape[1]))

    # Compute the average over the feature maps
    if feature_map.shape[2] != 1:
        cam = np.mean(resized_feature_map, axis=2)
    else:
        cam = resized_feature_map
    cam_img = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))
    cam_img = np.uint8(255 * cam_img)
    cv2.imwrite("cam_no_color_jet.png", cam_img)

    # Add the CAM to the input image as an overlay
    output_image = cv2.applyColorMap(cam_img, cv2.COLORMAP_JET)
    output_image = cv2.addWeighted(input_image, 0.5, output_image, 0.5, 0)

    cv2.imwrite("cam.png", output_image)

    return cam_img, output_image
