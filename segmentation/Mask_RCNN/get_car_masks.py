import argparse
import os

import numpy as np
import skimage.io

import mrcnn
from mrcnn import utils
from samples.coco import coco


class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_dir")
    parser.add_argument("masks_dir")
    return parser.parse_args()


def main():
    args = parse_args()
    classes_to_detect = [1, 3]  # person and car
    cocopath = os.path.abspath("./mask_rcnn_coco.h5")
    if not os.path.exists(cocopath):
        utils.download_trained_weights(cocopath)
    modelpath = os.path.abspath("./logs")
    model = mrcnn.model.MaskRCNN(mode="inference", model_dir=modelpath, config=InferenceConfig())
    model.load_weights(cocopath, by_name=True)
    counter = 0
    for name in os.listdir(args.image_dir):
        counter += 1
        print(counter, name)
        path = os.path.join(args.image_dir, name)
        image = skimage.io.imread(path)
        results = model.detect([image])
        r = results[0]
        idxs = [idx for idx in range(len(r["class_ids"])) if r["class_ids"][idx] in classes_to_detect]
        mask = np.zeros(r["masks"].shape[:-1]).astype(bool)
        for idx in idxs:
            mask = mask | r["masks"][:, :, idx]
        skimage.io.imsave(os.path.join(args.masks_dir, name), mask.astype(np.uint8) * 255)
    print("Done!")


if __name__ == '__main__':
    main()
