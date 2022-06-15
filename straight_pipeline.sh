#! /bin/bash

if [[ -z $1 || -z $2 ]]; then
	echo "Usage: ./straight_pipeline.sh <images_dir> <result_dir>"
	exit 2
fi

# Check whether there is docker images with names "openmvg" and "mask_rcnn" and build it if not present
OPENMVG_IMAGE_NAME="openmvg"
MASK_RCNN_IMAGE_NAME="mask_rcnn"
CURRENT_DOCKER_IMAGES=`docker images | awk '{print $1}' | sed s/REPOSITORY//`
OPENMVG_IMAGE_FOUND=`echo $CURRENT_DOCKER_IMAGES | grep -o $OPENMVG_IMAGE_NAME`
MASK_RCNN_IMAGE_FOUND=`echo $CURRENT_DOCKER_IMAGES | grep -o $MASK_RCNN_IMAGE_NAME`
if [[ -z $OPENMVG_IMAGE_FOUND ]]; then
	cd SfM/OpenMVG
	docker build -t $OPENMVG_IMAGE_NAME .
	cd ../..
fi
if [[ -z $MASK_RCNN_IMAGE_FOUND ]]; then
	cd segmentation/Mask_RCNN
	docker build -t $MASK_RCNN_IMAGE_NAME .
	cd ../..
fi

# rm -rf masks

mkdir -p masks
mkdir -p $2

WD=`pwd`
SOURCE_DIR=$WD/$1
MASKS_DIR=$WD/masks
RESULT_DIR=$WD/$2

echo "docker run --rm -ti \
	--volume $SOURCE_DIR:/input:ro \
	--volume $MASKS_DIR:/output:rw \
	mask_rcnn"


docker run --rm --env args='pipeline' \
	--volume $SOURCE_DIR:/input:rw \
	--volume $RESULT_DIR:/output:rw \
	--volume $MASKS_DIR:/masks:ro \
	openmvg
