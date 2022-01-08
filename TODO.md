# SPL Tasks - Get a good auto annotation model
- improve image extraction from logs
- add functionality to upload annotations to cvat

# NaoTH Tasks
- train a new model and export it to c++
- move dataset stuff from ball_detection_nao to dataset folder
- improve patch generation
- improve evaluation (i need some cool plots for my thesis)
  - evaluate patchsize is a nice experiment
    - should be reproduced on rc19 data, for that we need the patch generation with cppyy to work
  - evaluate splitting in classification and detection part compared to just detection
  - evaluate color vs. bw
  - balancierungsexperimente
    - verschiedene arten von balancierung

# Later Tasks
- fix cvat branch, we should always rebase our changes on top of current develop
- add documentation for manual tasks cvat
- segmentation data should be downloaded to global dataset path
