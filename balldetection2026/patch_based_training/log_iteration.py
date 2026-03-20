from naoth.log import Reader as LogReader
import argparse


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
    parser.add_argument("-l", "--log", required=True, help="Path to log file")
    parser.add_argument("-o", "--output", default="ball_candidates.txt", help="Output txt file")

    args = parser.parse_args()

    save_frame_numbers(args.log, args.output)
