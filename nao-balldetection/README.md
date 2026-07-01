# Balldetection for Nao v6
Currently we use mbc_36ksm_finetuned_crop as classifier and mbd_gopen_56k as detector. See the naoth-2020 repo for details.


# generate dataset
./test.sh

# train
uv run train.py