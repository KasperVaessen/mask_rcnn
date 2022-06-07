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

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log

from samples.boulders import boulders

MODEL_DIR = os.path.join("C:/Users/kaspe/Documents", "logs")
BOULDER_WEIGHTS_PATH = "C:/Users/kaspe/Documents/logs/boulder20220525T1506/mask_rcnn_boulder_0018.h5"

config = boulders.BoulderConfig()
BOULDER_DIR = os.path.join(ROOT_DIR, "data")

class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

DEVICE = "/cpu:0" # /cpu:0 or /gpu:0
TEST_MODE = "inference"

def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax

dataset = boulders.BoulderDataset()
dataset.load_boulder(BOULDER_DIR, "val")
dataset.prepare()
print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))

with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                              config=config)

weights_path = model.find_last()
model.load_weights(weights_path, by_name=True)

TP = 0
FP = 0
FN = 0
for image_id in dataset.image_ids:
    image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset, config, image_id)
    info = dataset.image_info[image_id]

    # Run object detection
    results = model.detect([image], verbose=1)

    # Display results
    ax = get_ax(1)
    r = results[0]
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                dataset.class_names, r['scores'], ax=ax,
                                title="Predictions", show_bbox=False)

    def calc_area(box):
        # print(box)
        return (box[2]-box[0])*(box[3]-box[1])

    print(gt_bbox.shape)
    print(r['rois'].shape)
    print(r['scores'])

    for gt_box in gt_bbox:
            iou = utils.compute_iou(gt_box, r['rois'], calc_area(gt_box), [calc_area(x) for x in r['rois']])
            if np.count_nonzero(np.array(iou) > 0.5) > 0:
                TP += 1
            else:
                FN += 1

    for box in r['rois']:
        iou = utils.compute_iou(box, gt_bbox, calc_area(box), [calc_area(x) for x in gt_bbox])
        if np.count_nonzero(np.array(iou) > 0.5) == 0:
            FP += 1

precision = TP/(TP+FP)
recall = TP/(TP+FN)

print(precision, recall)





