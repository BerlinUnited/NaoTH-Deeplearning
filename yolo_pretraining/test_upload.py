import zipfile
from pathlib import Path


def create_zip_file():
    filenames = ["yolo-full-size-detection_dataset_top_2024-04-05.yaml"]
    directory = Path("yolo-full-size-detection_dataset_top_2024-04-05")

    with zipfile.ZipFile("multiple_files.zip", mode="w") as archive:
        for filename in filenames:
            archive.write(filename)

        for file_path in directory.rglob("*"):
            archive.write(file_path, arcname=file_path.relative_to(directory))


if __name__ == "__main__":
    create_zip_file()
