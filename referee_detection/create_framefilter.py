"""
Create a framefilter for all images in standby and maybe relevant robots
"""
import os
from vaapi.client import Vaapi
import argparse


def main(log_id=282):
    client = Vaapi(
        base_url=os.environ.get("VAT_API_URL"),
        api_key=os.environ.get("VAT_API_TOKEN"),
    )
    response = client.behavior_frame_option.filter(
        log=log_id,
        option_name="decide_game_state",
        state_name="standby",
    )
    frame_numbers = [frame.frame_number for frame in response]
    print(sorted(frame_numbers))
    print()
    frame_filter_id = 1
    url = f"{os.environ.get("VAT_API_URL")}/api/log/{log_id}?filter={frame_filter_id}"
    print(url)
    #url: log
    # TODO make it possible to create a url for the frame filter
    # I guess we need a better redirect logic in the frontend for this


if __name__ == "__main__":
    # TODO argparse for log id
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--log", required=True, type=int)
    args = parser.parse_args()
    main(args.log)