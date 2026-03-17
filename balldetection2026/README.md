# NaoTH Deep Learning - German Open 2026 Edition
balldetection2026/
└── autolabeling/ #include the yolo pipeline for autogenerating new labels with yolo model
    ├── create_training_data_yolo.py
    ├── run_model_yolo.py #run the yolo model to annotate not-annotated images, 
    convert yolo .txt labels back to .json
    ├── train_model_yolo.py
    ├── data_bottom.yaml
    ├── data_top.yaml
└── classifier_patch_based/ #includes the classificiationmodel and the train-script for it
    ├── model.py 
    ├── train.py #the training (include data augumentation and converting keras in tflite)
└── data/ #includes all data (images, annotation, patches)


Stay in balldetection2026

1.) _Download annotated images_ and annotations from labelstudio.

```
uv run autolabeling/download.py -c TOP/BOTTOM - "annotated", "not_annotated", "both"
```

