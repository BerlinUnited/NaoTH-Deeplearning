FROM tensorflow/tensorflow:2.15.0-gpu

RUN pip install scikit-learn==1.4.*

WORKDIR /experiment

COPY . .

ENV OMP_NUM_THREADS "4"

# CMD ["python", "classifier_cnn_ball_gopen24.py"]