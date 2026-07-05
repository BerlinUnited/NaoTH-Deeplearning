import requests
import os
import zipfile
import os
from pathlib import Path
import requests

from create_ball_patches import create_ball_patches, download_annotated_ball_images_labelstudio, create_non_ball_patches


# naodevils labels
def get_naodevils_data(output_dir=Path("data/naodevils")):
    """
    Set of functions that download patch data from naodevils in the form
    0/
     non ball patches
    1/
      ball patches
    """
    url = "https://datasets.berlin-united.com/2026_07_02_patches_ruhrbot_devils.zip"
    output_path = output_dir / Path(f"2026_07_02_patches_ruhrbot_devils.zip")
    extraction_path = output_dir / Path(f"naodevils_data")

    if os.path.exists(output_path):
        print(f"'{output_path}' already exists. Skipping download.")
    else:
        print("Downloading file...")
        response = requests.get(url)

        # Check if the request was successful
        if response.status_code == 200:
            with open(output_path, "wb") as file:
                file.write(response.content)
            print("Download complete!")
        else:
            print(f"Failed to download. Status code: {response.status_code}")
            return

    # 2. Unzip the data
    print("Extracting data...")
    try:
        with zipfile.ZipFile(output_path, "r") as zip_ref:
            zip_ref.extractall(extraction_path)
        print(f"Extraction complete! Files saved to '{extraction_path}'")
    except zipfile.BadZipFile:
        print("Error: The downloaded file is corrupted or not a valid zip archive.")


# our labels
def get_patch_data_from_vat_server():
    """
    Set of functions that download patch data from vat and labelstudio in the form
    0/
     non ball patches
    1/
      ball patches
    """
    # download_annotated_ball_images_labelstudio(
    #     output_dir=Path("data/ball_images/TOP"), event_id=10, camera="TOP"
    # )
    # download_annotated_ball_images_labelstudio(
    #     output_dir=Path("data/ball_images/BOTTOM"), event_id=10, camera="BOTTOM"
    # )
    # create_ball_patches(
    #     input_dir=Path("./data/ball_images/TOP"),
    #     output_dir=Path("./data/ball_patches/TOP"),
    # )
    # create_ball_patches(
    #     input_dir=Path("./data/ball_images/BOTTOM"),
    #     output_dir=Path("./data/ball_patches/BOTTOM"),
    # )
    # create_non_ball_patches(
    #     input_dir=Path("./data/ball_images/TOP"),
    #     output_dir=Path("./data/non_ball_patches/TOP"),
    # )
    # create_non_ball_patches(
    #     input_dir=Path("./data/ball_images/BOTTOM"),
    #     output_dir=Path("./data/non_ball_patches/BOTTOM"),
    # )

    download_annotated_ball_images_labelstudio(
         output_dir=Path("data/ball_images/"), event_id=10, camera="TOP"
    )
    download_annotated_ball_images_labelstudio(
        output_dir=Path("data/ball_images/"), event_id=10, camera="BOTTOM"
    )

    create_ball_patches(
        input_dir=Path("./data/ball_images/"),
        output_dir=Path("./data/go26_patches/1.00"),
    )
    create_non_ball_patches(
        input_dir=Path("./data/ball_images/"),
        output_dir=Path("./data/go26_patches/0.00"),
    )


if __name__ == "__main__":
    #get_naodevils_data()
    get_patch_data_from_vat_server()
