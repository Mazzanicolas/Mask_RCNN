from custom import CustomConfig, CustomDataset
from color_splash import detect_and_color_splash
from mrcnn import model as modellib, utils
import argparse
import os

ROOT_DIR = os.path.abspath("../../")
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train Mask R-CNN to detect leather marks.')
parser.add_argument("command", metavar="<command>", help="'train' or 'splash'")
parser.add_argument('--dataset', required=False, metavar="/path/to/custom/dataset/", help='Directory of the Custom dataset')
parser.add_argument('--weights', required=True, metavar="/path/to/weights.h5", help="Path to weights .h5 file or 'coco'")
parser.add_argument('--logs', required=False, default=DEFAULT_LOGS_DIR, metavar="/path/to/logs/", help='Logs and checkpoints directory (default=logs/)')
parser.add_argument('--image', required=False, metavar="path or URL to image", help='Image to apply the color splash effect on')
parser.add_argument('--video', required=False, metavar="path or URL to video", help='Video to apply the color splash effect on')
args = parser.parse_args()

if __name__ == '__main__':
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video, "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = CustomConfig("mask")
    else:
        class InferenceConfig(CustomConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()


    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config, model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config, model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "splash":
        detect_and_color_splash(model, image_path=args.image,
                                video_path=args.video)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))



    # Training dataset.
    dataset_train = CustomDataset('mask')
    dataset_train.load_custom(args.dataset, "train", 'via_region_data.json', 'mask')
    dataset_train.prepare()

    # Validation dataset
    dataset_val = CustomDataset('mask')
    dataset_val.load_custom(args.dataset, "val", 'via_region_data.json', 'mask')
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("-------------------Training network heads-------------------")
    model.train(
        dataset_train, dataset_val,
        learning_rate=config.LEARNING_RATE,
        epochs=config.NUM_EPOCHS,
        layers='heads'
    )