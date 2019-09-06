"""
Mask R-CNN
Train a toy custom dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
Improved by Nicolás Mazza

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python custom.py train --dataset=/path/to/custom/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python custom.py train --dataset=/path/to/custom/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python custom.py train --dataset=/path/to/custom/dataset --weights=imagenet

    # Apply color splash to an image
    python custom.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python custom.py splash --weights=last --video=<URL or path to file>
"""

import os
import sys
import json
import numpy as np
import skimage.draw

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils


class CustomConfig(Config):
    '''
    Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    -------------------------------------------------------------------------------------
    * name >>> (str) Name of the configuration
    -------------------------------------------------------------------------------------
      images_per_gpu >>> (int) 6GB RAM 2 Images recomended <6GB 1 Image recomended
    -------------------------------------------------------------------------------------
      num_classes >>> (int) Number of custom classes #Classes = (background + num_classes)
    -------------------------------------------------------------------------------------
      steps_per_epoch >>> (int) Number of training steps per epoch
    -------------------------------------------------------------------------------------
      detection_confidence >>> (float) Skip detections with < detection_confidence confidence
    -------------------------------------------------------------------------------------
    '''
    def __init__(self, name, images_per_gpu=1, num_classes=1, steps_per_epoch=100, num_epochs=30, detection_confidence=0.9):
        self.NAME = name
        self.IMAGES_PER_GPU = images_per_gpu
        self.NUM_CLASSES = 1 + num_classes 
        self.STEPS_PER_EPOCH = steps_per_epoch
        self.NUM_EPOCHS = num_epochs
        self.DETECTION_MIN_CONFIDENCE = detection_confidence

class CustomDataset(utils.Dataset):

    def load_custom(self, dataset_dir, subset, annotations_file, class_name):
        '''
        Load a subset of the custom dataset
        -------------------------------------------------------------------------------------
        *  dataset_dir >>> (str) Root directory of the dataset
        -------------------------------------------------------------------------------------
        *  subset >>> (str) Subset to load: train or val
        -------------------------------------------------------------------------------------
        *  annotations_file >>> (str) Annotations file name
        -------------------------------------------------------------------------------------
        *  class_name >>> (str) Name of the class to mask
        -------------------------------------------------------------------------------------
        '''
        # Add classes. We have only one class to add.
        # TODO: add multiple class support
        self.class_name = class_name
        self.add_class(class_name, 1, class_name)

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Load annotations
        # We mostly care about the x and y coordinates of each region
        # Note: In VIA 2.0, regions was changed from a dict to a list.
        # TODO: Add other format support
        json_data   = self.load_json(dataset_dir, annotations_file)#"via_region_data.json")
        annotations = self.get_json_annotations(json_data)  # don't need the dict keys
        annotations = self.remove_unannotated_images(annotations)

        # Add images
        for annotation in annotations:
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. These are stores in the
            # shape_attributes (see json format above)
            # The if condition is needed to support VIA versions 1.x and 2.x.
            
            # TODO: create new tool that saves width and height https://github.com/Mazzanicolas/image-annotator WIP
            # This comparation is too expensive
            if type(annotation['regions']) is dict:
                polygons = [region['shape_attributes'] for region in annotation['regions'].values()]
            else:
                polygons = [region['shape_attributes'] for region in annotation['regions']] 

            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, annotation['filename'])
            # TODO: from PIL import Image; img = Image.open(img_dir); is faster
            # TODO: create new tool that saves width and height https://github.com/Mazzanicolas/image-annotator WIP
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(class_name, image_id=annotation['filename'],  # use file name as a unique image id
                           path=image_path, width=width, height=height, polygons=polygons)

    def remove_unannotated_images(self, annotations):
        return [annotation for annotation in annotations if annotation['regions']]

    def load_json(self, dataset_dir, file_name):
        return json.load(open(os.path.join(dataset_dir, file_name)))

    def get_json_annotations(self, json_data):
        return list(json_data.values())

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a custom dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != self.class_name:
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for idx, polygon in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(polygon['all_points_y'], polygon['all_points_x'])
            mask[rr, cc, idx] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        return info["path"] if info["source"] == "mark" else super(self.__class__, self).image_reference(image_id)




