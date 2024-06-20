# Should be built from the root of the project, so the context is the root of the project
# and copying the data files works as expected
FROM tensorflow/tensorflow:2.15.0-gpu
RUN pip install --ignore-installed mlflow==2.11.3 h5py==3.10.*  scikit-learn==1.4.*

ENV LOGNAME "cnn-clf-generic-docker"
ENV PYTHONPATH "${PYTHONPATH}:/experiment"
ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS "4"

WORKDIR /experiment

COPY ./ball-detection/classifier_cnn_generic/ ./classifier_cnn_generic/
COPY ./tools/ ./tools/
COPY ./data/ ./data/

WORKDIR /experiment/classifier_cnn_generic

CMD ["sleep", "36000"]