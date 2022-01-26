# Next Todos
- test what happens if you upload annotations in a track with some frames left out. I hope it will automatically interpolate
  - looks like it works, check on actual useful data again
- cleanup upload tools

# NaoTH Tasks
- fix /vol/repl261-vol4/naoth/logs/2019-07-02_RC19/2019-07-05_10-45-00_BnB_vs_SwiftArk_half2/videos
- cleanup image extraction from logs
  - use get_representations_from_log better
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
- Blurredness:
  - think about what to do with blurredness. 
    - For ball patches: We could not extract the patches from images that are too blurred, 
    - for robots use dortmund approach
      - it might be cool to set annotations per image, then we could download each image from a task, calculate the blurredness factor and upload the annotation via curl patch call.

# Later Tasks
- add documentation for manual tasks cvat
- update premake5
  - windows must still be done, then we can fix the warnings code
- add functionality to use the detectron inference result for uploading to cvat
  - requires my windows machine
- Fix cvat api to use org everwhere

### Importing SPL style annotations
things to think about:
- merging must be somehow supported
- probably only useful format for importing is cvat format because it supports annotations and tracks
  - i havent used the tracks feature much. 