from naoth.log import Reader as LogReader
from vaapi.client import Vaapi
import argparse
import requests
import os


def save_frame_numbers(log_path, output_path):
    with LogReader(log_path) as reader, open(output_path, "w") as f:

        for frame in reader.read():
            f.write(f"Frame: {frame.number}\n")

            if 'BallCandidates' in frame.get_names():
                try: 
                    bc = frame['BallCandidates']

                    for patch in bc.patches:
                        f.write("BallCandidate patch:\n")
                        f.write(
                            f"min_x: {patch.min.x}, "
                            f"min_y: {patch.min.y}, "
                            f"max_x: {patch.max.x}, "
                            f"max_y: {patch.max.y}\n"
                        )
                    # TODO: download image here
                    # api call looks something like this: https://vat.berlin-united.com/api/images/?log=679&camera=BOTTOM&frame=12312312
                except Exception as e:
                    f.write(f"No  BallCandidates: {e}\n")
            # Optional: top camera
            # if 'BallCandidatesTop' in frame.get_names():
            #     bc_top = frame['BallCandidatesTop']
            #     for patch in bc_top.patches:
            #         f.write("BallCandidatesTop patch:\n")
            #         f.write(
            #             f"min_x: {patch.min.x}, "
            #             f"min_y: {patch.min.y}, "
            #             f"max_x: {patch.max.x}, "
            #             f"max_y: {patch.max.y}\n"
            #         )

            f.write("\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument("-l", "--log", required=True, help="Path to log file")
    #parser.add_argument("-o", "--output", default="ball_candidates.txt", help="Output txt file")

    args = parser.parse_args()

    v_client = Vaapi(
        base_url=os.environ.get("VAT_API_URL"),
        api_key=os.environ.get("VAT_API_TOKEN"),
    )
    my_log = v_client.logs.get(679)
    print(my_log.log_path)

    url = "https://logs.berlin-united.com/" + my_log.log_path
    response = requests.get(url)

    # Open a local file in 'write-binary' mode
    with open(f"{my_log.id}_game.log", "wb") as f:
        f.write(response.content)

    print("Download complete!")
    
    #save_frame_numbers(args.log, args.output)
