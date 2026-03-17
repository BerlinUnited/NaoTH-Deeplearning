from naoth.log import Reader as LogReader
from naoth.log import Parser
import argparse
from vaapi.client import Vaapi 
import requests 
import os
from pathlib import Path

def save_all_candidates(log_path, log_id):
    my_parser = Parser()
    my_parser.register("BallCandidatesTop", "BallCandidates")

    with LogReader(log_path, my_parser) as reader, open(f"data/logs/{log_id}_ball_candidates.txt", "w") as f:

        for frame in reader.read():
            f.write(f"Frame: {log_id}_000{frame.number}\n")

            if 'BallCandidates' in frame.get_names():
                try: 
                    bc = frame['BallCandidates']

                    for patch in bc.patches:
                        f.write("BallCandidateBottom patch:\n")
                        f.write(
                            f"min_x: {patch.min.x}, "
                            f"min_y: {patch.min.y}, "
                            f"max_x: {patch.max.x}, "
                            f"max_y: {patch.max.y}\n"
                        )
                except Exception as e:
                    # f.write(f"No  BallCandidates: {e}\n")
                    continue
            
            if 'BallCandidatesTop' in frame.get_names():
                try: 
                    bc_top = frame['BallCandidatesTop']
                    for patch in bc_top.patches:
                        f.write("BallCandidatesTop patch:\n")
                        f.write(
                            f"min_x: {patch.min.x}, "
                            f"min_y: {patch.min.y}, "
                            f"max_x: {patch.max.x}, "
                            f"max_y: {patch.max.y}\n"
                        )
                except Exception as e:
                    # f.write(f"No  BallCandidates: {e}\n")
                    continue

            f.write("\n")

def download_log(log_path, log_id):
    v_client = Vaapi(
        base_url=os.environ.get("VAT_API_URL"),
        api_key=os.environ.get("VAT_API_TOKEN"),
    )
     
    my_log = v_client.logs.get(log_id)

    url = "https://logs.berlin-united.com/" + my_log.log_path
    response = requests.get(url)

    # Open a local file in 'write-binary' mode
    with open(log_path, "wb") as f:
        f.write(response.content)

    print("Download complete!")

    return log_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--log_ids", required=True, type=lambda s: s.split(","), help="log id")

    args = parser.parse_args()

    logs_dir = Path(f"data/logs")
    logs_dir.mkdir(parents=True, exist_ok=True)

    for log_id in args.log_ids:
        log_path = Path(f"data/logs/{log_id}_game.log")
        if not log_path.exists():
            log_path = download_log(log_path, log_id)

        save_all_candidates(log_path, log_id)
