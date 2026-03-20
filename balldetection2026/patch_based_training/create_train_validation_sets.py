from pathlib import Path
import random
import sys
import argparse
import shutil

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--camera", type=str, help="Set BOTTOM or TOP")
    args = parser.parse_args()

    if args.camera is None:
        print("The camera is not set.\nSet with option -c, --camera TOP/BOTTOM")
        sys.exit()
        
    image_dir = Path(f"data/{args.camera}/images/")
    image_all_dir = image_dir / "all"
    label_dir = Path(f"data/{args.camera}/labels/")
    label_all_dir = label_dir / "all"

    train_img = image_dir / "train"
    val_img = image_dir / "val"
    train_lbl = label_dir / "train"
    val_lbl = label_dir / "val"

    
    train_img.mkdir(parents=True, exist_ok=True)
    val_img.mkdir(parents=True, exist_ok=True)
    train_lbl.mkdir(parents=True, exist_ok=True)
    val_lbl.mkdir(parents=True, exist_ok=True)

    images = list(image_all_dir.glob("*.png"))
    random.shuffle(images)

    split_ratio = 0.8
    split_idx = int(len(images) * split_ratio)

    train_images = images[:split_idx]
    val_images = images[split_idx:]

    def copy_file(src_path, dest_dir):
        if src_path.exists():
            dest_path = dest_dir / src_path.name
            shutil.copy2(src_path, dest_path)
            return dest_path
        return None

    for img_path in train_images:
        lbl_path = label_all_dir / (img_path.stem + ".txt")
        copy_file(img_path, train_img)
        copy_file(lbl_path, train_lbl)

    for img_path in val_images:
        lbl_path = label_all_dir / (img_path.stem + ".txt")
        copy_file(img_path, val_img)
        copy_file(lbl_path, val_lbl)

    print(f"Train images: {len(train_images)}\nVal images: {len(val_images)}")