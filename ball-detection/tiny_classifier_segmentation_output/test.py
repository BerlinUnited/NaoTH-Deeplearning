import h5py

# with h5py.File("training_ds_y.h5", "r") as f:
#    print(len(f['Y']))
#    mask = f['X'][0]
#    print(mask)

with h5py.File("fy1500_segmentationdata_y.h5", "r") as h5f:
    for image in h5f["X"]:
        print(image)
