from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer

import cv2

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_101_FPN_3x.yaml"))
# cfg.DATASETS.TRAIN = tuple(dataset_lists2)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
#    "COCO-Detection/retinanet_R_101_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.MODEL.WEIGHTS = "output/ball_model_21/model_final.pth"
# only has one class (ball).
# (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

pred = DefaultPredictor(cfg)
im = cv2.imread("tests/test.png")
v = Visualizer(im[:, :, ::-1], )

outputs = pred(im)
print(outputs)
for box in outputs["instances"].pred_boxes.to('cpu'):
    v.draw_box(box)
v = v.get_output()
img = v.get_image()[:, :, ::-1]
cv2.imshow('image', img)
cv2.waitKey(0)

# closing all open windows
cv2.destroyAllWindows()