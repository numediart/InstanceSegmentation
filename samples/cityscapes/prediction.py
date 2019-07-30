import os
import sys
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import cityscapes
import cv2


ROOT_DIR = os.path.abspath('../../')
sys.path.append(ROOT_DIR)


from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
from mrcnn.config import Config
import mrcnn.model as modellib

import mrcnn.visualize as visualize

from mrcnn.model import log


MODEL_DIR = ROOT_DIR

cityscapes_weights_path = '../../mask_rcnn_cityscapes.h5'

config = cityscapes.cityscapesConfig()
cityscapes_dir = '/media/ambroise/cvdatasets/cityscapes/'

class InferenceConfig(config.__class__):
    GPU_COUNT=1
    IMAGES_PER_GPU=1
    IMAGE_RESIZE_MODE = 'none'

config = InferenceConfig()
config.display()


DEVICE = '/cpu:0'

TEST_MODE = 'inference'


def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax

dataset = cityscapes.cityscapesDataset()
dataset.load_cityscapes(cityscapes_dir, 'val')

dataset.prepare()

class_ids = [1, 2, 3, 4, 5, 6, 7, 8]
valid_class_ids = {1:24, 2:25, 3:26, 4:27, 5:28, 6:31, 7:32, 8:33}

print('image: {}\nClasses: {}'.format(len(dataset.image_ids), dataset.class_names))



with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                              config=config)

print("Loading weights ", cityscapes_weights_path)
model.load_weights(cityscapes_weights_path, by_name=True)

image_id = random.choice(dataset.image_ids)
image, image_meta, gt_class_id, gt_bbox, gt_mask =\
    modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
info = dataset.image_info[image_id]
print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id,
                                       dataset.image_reference(image_id)))

# Run object detection



image = cv2.imread('/media/ambroise/cvdatasets/cityscapes/leftImg8bit_sequence/train/aachen/aachen_000000_000000_leftImg8bit.jpg',-1)

results = model.detect([image], verbose=1)

# Display results
ax = get_ax(1)
r = results[0]
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                            dataset.class_names, r['scores'], ax=ax,
                            title="Predictions")
log("gt_class_id", gt_class_id)
log("gt_bbox", gt_bbox)
log("gt_mask", gt_mask)

plt.show()
#