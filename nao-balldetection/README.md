# Balldetection for Nao v6
From 2024 till 2026 we played with a ball classifier (mbc_36ksm_finetuned_crop) and detector (mbd_gopen_56k)
See the naoth-2020 repo for details


# Generate Dataset
All the datasets are generated with the `create_datasets.sh` script. But you need to download the data in the right format before with the `get_data.py` script

Set in 'main' if you want to download the patch-dataset the naodevils provided us (07.2026 on RC26)
or if you want to download patches based on our annotations in Labelstudio.

TODO: 
- Add current labelstudio annotation patches to the dataset server, as well as an download fct for those. 
- Include existing patch to download all predictions in labelstudio (also those not accepted by an human)
- Include and Improve Script downloading list of Images that probably have a ball, based on the state of the robot (from logs)

# train
uv run train.py

TODO: 
- Add more information in README.md