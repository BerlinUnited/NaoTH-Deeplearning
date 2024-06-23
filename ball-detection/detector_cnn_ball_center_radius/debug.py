from tensorflow import keras
from utils import load_h5_dataset_X_y, make_detection_dataset, plot_images_with_ball_center_and_radius

if __name__ == "__main__":
    X, y = load_h5_dataset_X_y(
        "../../data/ball_center_radius_patches_yuv422_y_only_pil_legacy_border0/ball_center_radius_patches_yuv422_y_only_pil_legacy_border0_combined_16x16_validation_ball_center_radius_X_y.h5"
    )

    X_mean = X.mean()
    X = X.astype("float32") - X_mean
    X = X / 255.0

    print(X.shape, y.shape)
    print("Img min:", X.min(), "Img max:", X.max())
    print("Img mean:", X.mean(), "Img std:", X.std())

    print("Center x min:", y[:, 0].min(), "Center x max:", y[:, 0].max())
    print("Center y min:", y[:, 1].min(), "Center y max:", y[:, 1].max())
    print("Radius min:", y[:, 2].min(), "Radius max:", y[:, 2].max())

    print(y[0])
    print(y[-1])

    ds = make_detection_dataset(X, y, batch_size=128, augment=True, rescale=False, prob=1, stddev=0.03, delta=0.1)

    X_aug, y_aug = ds.as_numpy_iterator().next()
    print(X_aug.shape, y_aug.shape)

    X_aug, y_aug = X_aug[-16:], y_aug[-16:]

    plot_images_with_ball_center_and_radius(X_aug, y_aug)
