"""
    Set of functions that download patch data in the form
    0/
     non ball patches
    1/
      ball patches
"""
import requests
import os
import zipfile

def get_naodevils_data():
    url = "https://datasets.berlin-united.com/2026_07_02_patches_ruhrbot_devils.zip"
    output_path = "./data/2026_07_02_patches_ruhrbot_devils.zip"
    extraction_path = (
        "./data/naodevils_data"  # Folder where the unzipped files will go
    )

    # 1. Check if the zip file already exists
    if os.path.exists(output_path):
        print(f"'{output_path}' already exists. Skipping download.")
    else:
        print("Downloading file...")
        # Send a GET request to the URL
        response = requests.get(url)

        # Check if the request was successful
        if response.status_code == 200:
            with open(output_path, "wb") as file:
                file.write(response.content)
            print("Download complete!")
        else:
            print(f"Failed to download. Status code: {response.status_code}")
            return  # Exit the function early if the download failed

    # 2. Unzip the data
    print("Extracting data...")
    try:
        with zipfile.ZipFile(output_path, "r") as zip_ref:
            zip_ref.extractall(extraction_path)
        print(f"Extraction complete! Files saved to '{extraction_path}'")
    except zipfile.BadZipFile:
        print(
            "Error: The downloaded file is corrupted or not a valid zip archive."
        )

def get_patch_data_from_vat_server():
    pass

get_naodevils_data()