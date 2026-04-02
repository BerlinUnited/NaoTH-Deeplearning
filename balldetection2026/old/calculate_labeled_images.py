from vaapi.client import Vaapi
import argparse
import os

log_ids = [679, 678, 677, 676, 675]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--camera", type=str, help="Set BOTTOM or TOP")
    args = parser.parse_args()
    v_client = Vaapi(
        base_url=os.environ.get("VAT_API_URL"),
        api_key=os.environ.get("VAT_API_TOKEN"),
    )

    my_sum = 0
    for log_id in log_ids:
        image_obj_list = list(v_client.image.list(log=log_id, camera=args.camera, validated=True))
        print(f"number of validated images: {len(image_obj_list)}")
        my_sum += len(image_obj_list)

    print()
    print(my_sum)

    # TODO create function that counts all bounding boxes