from tools import (
    load_image_as_yuv422_cv2,
    load_image_as_yuv422_pil,
    load_image_as_yuv422_y_only_cv2,
    load_image_as_yuv422_y_only_pil,
)

TEST_FILE = "./4602.png"


def test_dimensions_are_the_same():
    cv2_yuv422 = load_image_as_yuv422_cv2(TEST_FILE)
    pil_yuv422 = load_image_as_yuv422_pil(TEST_FILE)

    cv2_y_only = load_image_as_yuv422_y_only_cv2(TEST_FILE)
    pil_y_only = load_image_as_yuv422_y_only_pil(TEST_FILE)

    assert cv2_y_only.shape == pil_y_only.shape
    assert cv2_yuv422.shape == pil_yuv422.shape


def plot_test_images():
    import matplotlib.pyplot as plt

    cv2_yuv422 = load_image_as_yuv422_cv2(TEST_FILE)
    pil_yuv422 = load_image_as_yuv422_pil(TEST_FILE)

    cv2_y_only = load_image_as_yuv422_y_only_cv2(TEST_FILE)
    pil_y_only = load_image_as_yuv422_y_only_pil(TEST_FILE)

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    axs[0, 0].imshow(cv2_yuv422[..., 0])
    axs[0, 0].set_title("cv2 yuv422")
    axs[0, 1].imshow(pil_yuv422[..., 0])
    axs[0, 1].set_title("pil yuv422")
    axs[1, 0].imshow(cv2_y_only[..., 0], cmap="gray")
    axs[1, 0].set_title("cv2 y only")
    axs[1, 1].imshow(pil_y_only[..., 0], cmap="gray")
    axs[1, 1].set_title("pil y only")

    plt.savefig("yuv_test_images.png")


if __name__ == "__main__":
    test_dimensions_are_the_same()
    plot_test_images()
