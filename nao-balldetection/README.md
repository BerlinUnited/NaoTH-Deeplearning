# Balldetection for Nao v6
See the naoth-2020 repo for details (model: mbc_36ksm_finetuned_crop as classifier and mbd_gopen_56k as detector).


# generate dataset
Set in 'main' if you want to download the patch-dataset the naodevils provided us (07.2026 on RC26)
or if you want to download patches based on our annotations in Labelstudio.

TODO: 
- Add current labelstudio annitation patches to the dataset server, as well as an download fct for those. 
- Include existing patch to download all predictions in labelstudio (also those not accepted by an human)
- Include and Improve Script downloading list of Images that probably have a ball, based on the state of the robot (from logs)

# train
uv run train.py

TODO: 
- Add more information in README.md