# SPL Tasks - Get a good auto annotation model
- add functionality to use the detectron inference result for uploading to cvat

# NaoTH Tasks
- cleanup image extraction from logs
  - use get_representations_from_log better
  - use tempfile approach for speedup
- train a new model and export it to c++
- improve patch generation
- improve evaluation (i need some cool plots/experimentes for my thesis)
  - evaluate patchsize is a nice experiment
    - should be reproduced on rc19 data, for that we need the patch generation with cppyy to work
  - evaluate splitting in classification and detection part compared to just detection
  - evaluate color vs. bw
  - balancierungsexperimente
    - verschiedene arten von balancierung
  - seperated upper/lower cam vs combined model
  - pruning or other techniques
  - check gaussian smoothing before inference
- find a good solution for global config.toml
- for tk03 create val set manually and specify it in dataset creation

# Later Tasks
- fix cvat branch, we should always rebase our changes on top of current develop
- add documentation for manual tasks cvat
- 