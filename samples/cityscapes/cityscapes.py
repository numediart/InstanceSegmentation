
import os
import sys
import time
import numpy as np
import imgaug  # https://github.com/aleju/imgaug (pip3 install imgaug)

# Download and install the Python COCO tools from https://github.com/waleedka/coco
# That's a fork from the original https://github.com/pdollar/coco with a bug
# fix for Python 3.
# I submitted a pull request https://github.com/cocodataset/cocoapi/pull/50
# If the PR is merged then use the original repo.
# Note: Edit PythonAPI/Makefile and replace "python" with "python3".
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

import zipfile
import urllib.request
import shutil
import cv2

ROOT_DIR = os.path.abspath('../../')

sys.path.append(ROOT_DIR)

from mrcnn.config import Config
from mrcnn import model as modellib, utils

COCO_MODEL_PATH = os.path.join(ROOT_DIR, 'mask_rcnn_coco.h5')

DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, 'logs')
DEFAULT_DATASET_YEAR = '2014'


class cityscapesConfig(Config):

    NAME= 'cityscapes'

    IMAGES_PER_GPU = 1
    GPU_COUNT = 1

    NUM_CLASSES = 9


class cityscapesDataset(utils.Dataset):
    def load_cityscapes(self, dataset_dir, subset, class_ids=None, class_names=None):

        assert subset in ['train', 'val']
        if not class_names:
            class_names = ['person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bike']
        if not class_ids:
            class_ids = [1, 2, 3, 4, 5, 6, 7, 8]
        valid_class_ids = [24, 25, 26, 27, 28, 31, 32, 33]
        subset_path_annotations = os.path.join(dataset_dir, 'gtFine', subset)

        img_files = []
        img_names = []
        img_width = []
        img_height = []
        ann_files = []
        ann_names = []



        for subdir, dirs, files in os.walk(subset_path_annotations):
            files = sorted(files)
            for file in files:
                if file.endswith('instanceIds.png'):
                    dataset_mask = cv2.imread(os.path.join(subdir,file),-1)
                    ids = np.unique(dataset_mask)
                    add_image = False
                    for idd in ids:
                        if int(str(idd)[:2]) in valid_class_ids:
                            add_image = True
                    if add_image == True:
                        h = dataset_mask.shape[0]
                        w = dataset_mask.shape[1]
                        img_width.append(w)
                        img_height.append(h)

                        file_path = os.path.join(subdir, file)
                        ann_names.append(file)
                        ann_files.append(file_path)

                        file_2 = file.replace('gtFine_instanceIds', 'leftImg8bit')
                        subdir_2 = subdir.replace('gtFine', 'leftImg8bit')
                        img_names.append(file_2)
                        file_path = os.path.join(subdir_2, file_2)
                        img_files.append(file_path)

        for i in range(len(class_names)):
            self.add_class('cityscapes', class_ids[i], class_names[i])

        for i in range(len(img_files)):
            self.add_image('cityscapes', image_id=img_names[i], path=img_files[i],
                           width=img_width[i], height=img_height[i], mask_id=ann_names[i], mask_path = ann_files[i])

    def load_mask(self, image_id):

        image_info = self.image_info[image_id]
        if image_info['source'] != 'cityscapes':
            return super(self.__class__, self).load_mask(image_id)

        dataset_mask = cv2.imread(image_info['mask_path'], -1)
        ids = np.unique(dataset_mask)

        valid_class_ids = {24:1, 25:2,
                           26:3, 27:4,
                           28:5, 31:6,
                           32:7, 33:8}
        instance_mask = []
        class_ids = []

        for i, p in enumerate(ids):
            p_class = int(str(p)[:2])
            if p_class in valid_class_ids:
                temp_mask = np.zeros(dataset_mask.shape, dtype=bool)
                temp_mask[dataset_mask==p] = True
                instance_mask.append(temp_mask)
                class_ids.append(valid_class_ids[p_class])

        if len(instance_mask) > 1:
            instance_mask = np.dstack(instance_mask)
        else:
            instance_mask = np.array(instance_mask)
            instance_mask = instance_mask.squeeze()
            if len(instance_mask.shape) < 3:
                instance_mask = instance_mask[..., np.newaxis]
        return instance_mask, np.array(class_ids, dtype=np.int32)

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        if info['source'] == 'cityscapes':
            return info['path']
        else:
            super(self.__class__, self).image_reference(image_id)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN on Cityscapes')
    parser.add_argument('command',
                        metavar='<command>',
                        help='train or evaluate on Cityscapes')

    parser.add_argument('--dataset', required=True,
                        metavar='path/to/cityscapes',
                        help = 'Directory of Cityscapes')

    parser.add_argument('--model', required=True,
                        metavar='/path/to/weights.h5',
                        help='path to weights .h5 file or coco')
    parser.add_argument('--logs',required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar='Logs and checkpoints directory')

    args = parser.parse_args()
    print('Command: ', args.command)
    print('Model: ', args.model)
    print('Dataset: ', args.dataset)
    print('logs: ', args.logs)

    if args.command == 'train':
        config = cityscapesConfig()
    else:
        class InferenceConfig(cityscapesConfig):
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = 0
        config = InferenceConfig()
    config.display()


    if args.command == 'train':
        model = modellib.MaskRCNN(mode='training', config=config, model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode='inference', config=config, model_dir=args.logs)

    if args.model.lower() == 'coco':
        model_path = COCO_MODEL_PATH
    elif args.model.lower() == 'last':
        model_path = model.find_last()
    elif args.model.lower() == 'imagenet':
        model_path = model.get_imagenet_weights()
    else:
        model_path = args.model
    model.load_weights(model_path, by_name=True, exclude=[
        "mrcnn_class_logits", "mrcnn_bbox_fc",
        "mrcnn_bbox", "mrcnn_mask"])

    if args.command == 'train':
        dataset_train = cityscapesDataset()
        dataset_train.load_cityscapes(args.dataset, 'train')
        dataset_train.prepare()

        dataset_val = cityscapesDataset()
        dataset_val.load_cityscapes(args.dataset, 'val')
        dataset_val.prepare()

        augmentation = imgaug.augmenters.Fliplr(0.5)


        print("Training network heads")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=40,
                    layers='heads',
                    augmentation=augmentation)

        # Training - Stage 2
        # Finetune layers from ResNet stage 4 and up
        print("Fine tune Resnet stage 4 and up")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=120,
                    layers='4+',
                    augmentation=augmentation)

        # Training - Stage 3
        # Fine tune all layers
        print("Fine tune all layers")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE / 10,
                    epochs=160,
                    layers='all',
                    augmentation=augmentation)
    elif args.command == 'evaluate':
        dataset_val = cityscapesDataset()
        cs = dataset_val.load_cityscapes(args.dataset, 'val')
        dataset_val.prepare()













