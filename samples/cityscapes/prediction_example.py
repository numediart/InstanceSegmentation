from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from absl import app
from absl import flags
from absl import logging

import numpy as np
import pandas as pd
import os
import cv2
from collections import Counter
import argparse
import pickle
import scipy.io as sio
import tensorflow as tf
from PIL import Image

import cityscapes

from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
from mrcnn.config import Config
import mrcnn.model as modellib


from mrcnn.model import log

import matplotlib.pyplot as plt

gfile = tf.gfile

parser = argparse.ArgumentParser(description='create instance masks for cityscapes')
parser.add_argument('--weights', type=str, help='path to weights file', required=True)
parser.add_argument('--image_path', type=str, help='path to image file', required=True)

args = parser.parse_args()

if __name__ == '__main__':

    DEVICE = '/gpu:0'
    TEST_MODE = 'inference'

    cityscapes_weights_path = args.weights

    config = cityscapes.cityscapesConfig()


    class InferenceConfig(config.__class__):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        IMAGE_RESIZE_MODE = 'none'


    config = InferenceConfig()
    config.display()

    class_ids = {1: 24, 2: 25, 3: 26, 4: 27, 5: 28, 6: 31, 7: 32, 8: 33}

    with tf.device(DEVICE):
        model = modellib.MaskRCNN(mode="inference", model_dir='../../',
                                  config=config)

    print("Loading weights ", cityscapes_weights_path)
    model.load_weights(cityscapes_weights_path, by_name=True)

    with Image.open(args.image_path) as tmp:
        if tmp.size[0] != 2048 or tmp.size[1] != 1024:
            tmp = tmp.resize((2048, 1024), Image.ANTIALIAS)
            base_name = os.path.basename(args.image_path)
            base_name = base_name.split('.')
            dir_path = os.path.dirname(args.image_path)
            resized_name = '{}_resized.png'.format(base_name[0])
            resized_path = os.path.join(dir_path, resized_name)
            tmp.save(resized_path, 'PNG')
        img = np.asarray(tmp)
    results = model.detect([img], verbose=1)
    r = results[0]
    keep = np.where((r['rois'][:, 2] - r['rois'][:, 0] > 30) &
                    (r['rois'][:, 3] - r['rois'][:, 1] > 30) &
                    (r['scores'] >= 0.9))

    r['masks'] = r['masks'][:, :, keep[0]]
    r['class_ids'] = r['class_ids'][keep[0]]
    unique_instances = np.unique(r['class_ids'])
    unique_instances_counts = {}
    for j in unique_instances:
        unique_instances_counts[j] = 0
    mask = np.zeros((r['masks'].shape[0], r['masks'].shape[1]), dtype=np.uint16)

    for j in range(r['masks'].shape[2]):
        instance_class = r['class_ids'][j]
        cs_instance_class = class_ids[instance_class]
        count = unique_instances_counts[instance_class]
        pix_value = str(cs_instance_class) + '{0:03d}'.format(count)
        mask[r['masks'][:, :, j] != 0] = pix_value
        unique_instances_counts[instance_class] += 1

    base_name = os.path.basename(args.image_path)
    dir_path = os.path.dirname(args.image_path)
    base_name = base_name.split('.')
    instance_name = '{}_instance.jpg'.format(base_name[0])
    instance_path = os.path.join(dir_path, instance_name)
    mask_img = Image.fromarray(mask.astype(np.uint16))
    mask_img.save(instance_path, 'PNG')







