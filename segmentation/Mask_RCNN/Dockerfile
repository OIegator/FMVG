FROM python:3.6

RUN git clone https://github.com/matterport/Mask_RCNN.git mask_rcnn

RUN apt-get update && apt-get install -y python3-opencv 

RUN cd mask_rcnn && pip install -r requirements.txt

# Patch build with the correct tensorflow version
RUN pip uninstall keras keras-nightly keras-Preprocessing keras-vis tensorflow -y && \
  pip install tensorflow==1.13.1 keras==2.1.0

RUN pip install pycocotools

RUN pip install "h5py<3.0.0"

COPY ./get_car_masks.py ./mask_rcnn

CMD ["python", "./mask_rcnn/get_car_masks.py", "/input", "/output"]
