# Should be built from the root of the project, so the context is the root of the project
# and copying the data files works as expected
FROM tensorflow/tensorflow:2.15.0-gpu

# install opencv deps
RUN apt-get update && apt -y install libgl1-mesa-glx libglib2.0-0  nano
RUN pip install --ignore-installed \
  pillow==10.3.0 \
  opencv-python==4.9.0.80 \
  psycopg2-binary==2.9.9 \
  label_studio_sdk==0.0.34 \
  tqdm==4.66.4 \
  matplotlib==3.9.0 \
  mlflow==2.14.0 \
  minio==7.2.7 
  
ENV LOGNAME "cnn-clf-generic-docker"
ENV PYTHONPATH "${PYTHONPATH}:/experiment"
ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS "4"

WORKDIR /experiment

COPY ./ball-detection/classifier_cnn/ ./ball-detection/classifier_cnn/
COPY ./tools/ ./tools/
COPY ./data/classification_patches_yuv422_y_only_pil_legacy_border0 ./data/classification_patches_yuv422_y_only_pil_legacy_border0

WORKDIR /experiment/ball-detection/classifier_cnn