# Should be built from the root of the project, so the context is the root of the project
# and copying the data files works as expected
FROM tensorflow/tensorflow:2.15.0-gpu

# install opencv deps
# mlflow has somehow a dependency to git
RUN apt-get update && apt -y install libgl1-mesa-glx libglib2.0-0 git nano

RUN groupadd -g 1000 naoth && useradd -m -u 1000 -g 1000 -s /bin/bash naoth
USER naoth

RUN pip install --ignore-installed \
  pillow==10.3.0 \
  opencv-python==4.9.0.80 \
  psycopg2-binary==2.9.9 \
  label_studio_sdk==0.0.34 \
  tqdm==4.66.4 \
  matplotlib==3.9.0 \
  mlflow==2.14.0 \
  minio==7.2.7 
  
ENV LOGNAME "cnn-classifier-docker"
ENV PYTHONPATH "${PYTHONPATH}:/experiment"
ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS "4"
ENV TF_FORCE_GPU_ALLOW_GROWTH "true"

WORKDIR /experiment
RUN mkdir /experiment/data
#RUN mkdir /data

COPY --chown=1000:1000 ./ball-detection/classifier_cnn/ ./ball-detection/classifier_cnn/
COPY --chown=1000:1000 ./tools/ ./tools/

WORKDIR /experiment/ball-detection/classifier_cnn