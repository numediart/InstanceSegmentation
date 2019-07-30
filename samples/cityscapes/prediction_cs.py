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

import mrcnn.visualize as visualize

from mrcnn.model import log

import matplotlib.pyplot as plt

gfile = tf.gfile

parser = argparse.ArgumentParser(description='create instance masks for cityscapes')
parser.add_argument('--weights', type=str, help='weights path', required=True)
parser.add_argument('--dataset', type=str, help='dataset to annotate', required=True)
parser.add_argument('--files_list', type=str, help='img list as txt', required=True)
parser.add_argument('--city', type=str, help='city to process', required=True)

args = parser.parse_args()


if __name__ == '__main__':

    DEVICE = '/gpu:0'
    TEST_MODE = 'inference'

    #dataset_dir = '/media/ambroise/cvdatasets/'
    #dataset_dir = '/media/memory/ambroise/datasets/'
    #dataset_path = os.path.join(dataset_dir, args.dataset)
    dataset_path = args.dataset
    file_list_path = args.files_list
    cityscapes_weights_path = args.weights

    config = cityscapes.cityscapesConfig()
    class InferenceConfig(config.__class__):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        IMAGE_RESIZE_MODE = 'none'

    config = InferenceConfig()
    config.display()

    with open(file_list_path, 'r') as file:
        img_files = file.readlines()

    img_files = [f.rstrip() for f in img_files]
    img_files = [f.replace('png', 'jpg') for f in img_files]

    class_ids = {1:24, 2:25, 3:26, 4:27, 5:28, 6:31, 7:32, 8:33}

    with tf.device(DEVICE):
        model = modellib.MaskRCNN(mode="inference", model_dir='../../',
                                  config=config)

    print("Loading weights ", cityscapes_weights_path)
    model.load_weights(cityscapes_weights_path, by_name=True)

    for i in img_files:
        path = os.path.join(dataset_path, i)
        dir_path = os.path.dirname(path)
        dir_path = dir_path.replace('rightImg8bit_sequence', 'rightImg8bit_instance')
        #dir_path = dir_path.replace('leftImg8bit_sequence', 'leftImg8bit_instance')
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        with Image.open(path) as img:
            a = np.asarray(img)

        results = model.detect([a], verbose=1)

        r = results[0]
        keep = np.where((r['rois'][:,2] - r['rois'][:,0] > 30) &
                        (r['rois'][:,3] - r['rois'][:,1] > 30) &
                        (r['scores'] >= 0.9))

        r['masks'] = r['masks'][:,:,keep[0]]
        #print('before filtering, classes in the image: ', r['class_ids'])
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
            mask[r['masks'][:,:,j] != 0] = pix_value
            unique_instances_counts[instance_class] += 1
        base_name = os.path.basename(i)
        instance_name = base_name.replace('rightImg8bit.jpg', 'rightImg8bit_instance.png')
        #instance_name = base_name.replace('leftImg8bit.jpg', 'leftImg8bit_instance.png')
        instance_path = os.path.join(dir_path, instance_name)
        mask_img = Image.fromarray(mask.astype(np.uint16))
        mask_img.save(instance_path, 'PNG')

	
	
        








