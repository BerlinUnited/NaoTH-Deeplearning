"""
    expects a image with added metadate like the ones exported with the image_example script and prints
    all the keys and values found in the meta data
"""
from PIL import Image

if __name__ == "__main__":
    image_path = "4602.png"
    img = Image.open(image_path)

    for k, v in img.info.items():
        print("%s: %s" % (k, v))
