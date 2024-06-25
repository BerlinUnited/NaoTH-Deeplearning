# NaoTH Dataset Tools

Contains scripts for creating, persisting and downloading various datasets we use in this repository

## Naoth Datasets Archive

We store our datasets at https://datasets.naoth.de  
If you create or update datasets after making improvements or bugfixes, please update the stored dataset accordingly.  
The current default way to do this is to mount the HU repl via sshfs locally and upload the files through your file manager.

Example sshfs mount command (make sure to replace your target mount point):
`sshfs 'naoth@gruenau6.informatik.hu-berlin.de:/vol/repl261-vol4/naoth/' '/Users/max/sshfs' -o transform_symlinks -o idmap=user  -o auto_cache`

## CNN Ball classification Patches

These are the patches we use to train our Ball Classifier CNNs (keras).  
We save the datasets in the `.h5` file format and store the following numpy arrays:

- X, an array of images of shape [n_samples, HEIGHT, WIDTH, CHANNELS]
- y, an array of target classes with shape [n_samples, 1] , where the targets are 1 for ball and 0 for not-ball

You can load the h5 dataset file like this:

```py
import h5py

with File(file_path, "r") as h5_file:
    X: np.ndarray = h5_file["X"][:]
    y: np.ndarray = h5_file["y"][:]
```

You can run the script `dataset_tools/ball_classification_patches_cnn_datasets/create_datasets.sh` to create multiple distinct datasets for training and model evaluation. The script is highly parametrized, and by default creates all currently valid combinations of parameters. These include parameters like patch size, color mode, source camera and many more. For more details, check out `dataset_tools/ball_classification_patches_cnn_datasets/create_datasets.py`.

## CNN Ball radius and center location regression Patches

These are the patches we use to train our Ball Radius and Center Regression CNNs (keras).  
We save the datasets in the `.h5` file format and store the following numpy arrays:

- X, an array of images of shape [n_samples, HEIGHT, WIDTH, CHANNELS]
- y, an array of target values of shape [n_samples, 3], where target values are [ball_radius, ball_center_x, ball_center_y]

You can load the h5 dataset file like this:

```py
import h5py

with File(file_path, "r") as h5_file:
    X: np.ndarray = h5_file["X"][:]
    y: np.ndarray = h5_file["y"][:]
```

You can run the script `dataset_tools/ball_radius_center_patches_cnn_datasets/create_datasets.sh` to create multiple distinct datasets for training and model evaluation. The script is highly parametrized, and by default creates all currently valid combinations of parameters. These include parameters like patch size, color mode, source camera and many more. For more details, check out `dataset_tools/ball_radius_center_patches_cnn_datasets/create_datasets.py`.
