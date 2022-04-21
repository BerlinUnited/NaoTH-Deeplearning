"""
    Contains various functions to crawl over the naoth log folder. It's written for local execution relative to
    the logs. It's tested by using sshfs.

    Features:
        - combine game.log and image.log files
        - export frames from video files
        - export images from naoth logfiles
        - TODO export list of representations from logs
"""
import os
import sys
import subprocess
from pathlib import Path
import shutil
import tempfile
import numpy as np
from PIL import PngImagePlugin
from PIL import Image as PIL_Image

from naoth.log import Parser
from naoth.log import Reader as LogReader
from naoth.pb.Framework_Representations_pb2 import Image
from common_tools.main import get_logs_root


def get_images(frame):
    try:
        image_top = image_from_proto(frame["ImageTop"])
    except KeyError:
        image_top = None

    try:
        cm_top = frame["CameraMatrixTop"]
    except KeyError:
        cm_top = None

    try:
        image_bottom = image_from_proto(frame["Image"])
    except KeyError:
        image_bottom = None

    try:
        cm_bottom = frame["CameraMatrix"]
    except KeyError:
        cm_bottom = None

    return [frame.number, image_bottom,
            image_top, cm_bottom, cm_top]


def image_from_proto(message):
    # read each channel of yuv422 separately
    yuv422 = np.frombuffer(message.data, dtype=np.uint8)
    y = yuv422[0::2]
    u = yuv422[1::4]
    v = yuv422[3::4]

    # convert from yuv422 to yuv888
    yuv888 = np.zeros(message.height * message.width * 3, dtype=np.uint8)

    yuv888[0::3] = y
    yuv888[1::6] = u
    yuv888[2::6] = v
    yuv888[4::6] = u
    yuv888[5::6] = v

    yuv888 = yuv888.reshape((message.height, message.width, 3))

    # convert the image to rgb and save it
    img = PIL_Image.frombytes('YCbCr', (message.width, message.height), yuv888.tostring())
    return img


def list_events():
    logs_root = Path(get_logs_root())
    for path in sorted(logs_root.iterdir()):
        if path.is_dir():
            print(path)


def extract_frames_from_videos(event_name="2019-07-02_RC19-others"):
    event_root = Path(get_logs_root()) / event_name
    # iterate over all games of the event
    for game in sorted(event_root.iterdir()):
        if not game.is_dir():
            # ignore files inside the event root
            continue
        print(game)

        # ignore test games. This might be useful to change later. For now the games are more interesting
        if "test" in str(game).lower():
            print("\ttest games will be ignored")
            continue
        if "invisible" in str(game).lower():
            print("\tgames against invisible will be ignored")
            continue

        # check if frames.zip already exists if so put out a warning and continue
        output_zipfile = Path(game) / "extracted" / "frames.zip"
        if output_zipfile.is_file():
            print("\tframes.zip file already exists. Will continue with the next game")
            continue

        # check if only one video file exists
        video_folder = game / "videos"

        # TODO test this in windows. Maybe it finds the same file twice there
        video_files1 = list(Path(video_folder).rglob('*.MP4'))
        video_files2 = list(Path(video_folder).rglob('*.mp4'))
        video_files = video_files1 + video_files2

        if len(video_files) == 0:
            print("\tno video files found will continue with the next game")
            continue
        elif len(video_files) > 1:
            print(
                "\ttoo many video files found. Can't decide which one is the correct video. Maybe you need to combine the videos? ")
            continue
        else:
            video_file = video_files[0]

        # create extracted folder
        extracted_folder = Path(game) / "extracted"
        extracted_folder.mkdir(parents=True, exist_ok=True)

        f = tempfile.mkdtemp()
        temp_video_file_name = Path(f) / video_file.name
        temp_zipfile = Path(f) / "frames.zip"
        temp_frame_folder = Path(f) / "frames"
        temp_frame_folder.mkdir(parents=True, exist_ok=True)
        print("\tdownloading video file to temp folder: ", f)
        shutil.copy(str(video_file), f)

        # ffmpeg executable is named differently on windows and linux
        if sys.platform == 'win32':
            ffmpeg_exe = "ffmpeg.exe"
        else:
            ffmpeg_exe = "ffmpeg"

        # build the ffmpeg command 
        # FIXME "/%06d.jpg\" must be changed to "\%06d.jpg\" in windows. Maybe i can figure out a way to make it work in both envs
        cmd = f"{ffmpeg_exe} -i \"{str(temp_video_file_name)}\" " + "\"" + str(
            temp_frame_folder) + "/%06d.jpg\"" + " -loglevel quiet -stats"
        print("\trun ffmpg with: ", cmd)

        try:
            subprocess.call(cmd, shell=True)
        except subprocess.CalledProcessError:
            print("\tERROR: ffmpeg executable not found.")
            quit()

        print("\tzipping frames ...")
        shutil.make_archive(str(temp_zipfile.parent / temp_zipfile.stem), 'zip', str(temp_frame_folder))

        print("\tcopy zip file to repl")
        shutil.copy(str(temp_zipfile), str(output_zipfile))

        print("\tremove temp folder")
        shutil.rmtree(f)


def combine_logfiles(event_name="2019-11-21_Koeln"):
    """
    images.log contains only images we want to combine it with game.log files. So that we have the corresponding
    camera matrix to the image in the same log. Later it can be used to write the camera matrix or other information
    inside the png header. This feature is used in the patch extraction.
    """

    def combine(gamelog, image_logfile, combined_log):
        print("Indexing image log...")
        image_log_index = create_image_log_dict(str(image_logfile))
        print('Writing new log to: "{}"...'.format(str(combined_log)))
        with open(str(combined_log), 'wb') as output, open(str(image_logfile), 'rb') as image_log, LogReader(
                str(gamelog)) as reader:
            for frame in reader.read():
                # only write frames which have corresponding images
                if frame.number in image_log_index:
                    # load image data
                    offset, size, camera_bottom = image_log_index[frame.number]
                    image_name = 'Image' if camera_bottom else 'ImageTop'
                    image_log.seek(offset)
                    image_data = image_log.read(size)

                    # add image from image.log
                    msg = Image()
                    msg.height = 480
                    msg.width = 640
                    msg.format = Image.YUV422
                    msg.data = image_data

                    frame.add_field(image_name, msg)

                    output.write(bytes(frame))
                    # Frames are indexed by the log reader. Remove the image of already processed frames to
                    # preserve memory
                    frame.remove(image_name)

    event_root = Path(get_logs_root()) / event_name
    # iterate over all games of the event
    for game in event_root.iterdir():
        print(game)
        for robot in (Path(game) / "game_logs").iterdir():
            print("\t", robot)
            gamelog_file = Path(robot) / "game.log"
            image_logfile = Path(robot) / "images.log"
            combined_log = Path(robot) / "combined.log"
            if combined_log.is_file():
                print("\t\tcombined.log file already exists, will continue to next robot")
                continue
            if not gamelog_file.is_file():
                print("\t\tgame.log file does not exists, will continue to next robot")
                continue
            else:
                if os.stat(str(gamelog_file)).st_size == 0:
                    print("\t\tgame.log is empty, will continue to next robot")
                    continue
            if not image_logfile.is_file():
                print("\t\timages.log file does not exists, will continue to next robot")
                continue
            else:
                if os.stat(str(image_logfile)).st_size == 0:
                    print("\t\timages.log is empty, will continue to next robot")
                    continue

            # run combining the logs
            combine(gamelog_file, image_logfile, combined_log)


def export_images(logfile, img):
    """
        creates two folders:
            <logfile name>_top
            <logfile name>_bottom

        and saves the images inside those folders
    """
    # TODO save in extraced folder and zipped
    logfile_name = Path(logfile).parent / Path(logfile).stem
    output_folder_top = Path(str(logfile_name) + "_top")
    output_folder_bottom = Path(str(logfile_name) + "_bottom")

    output_folder_top.mkdir(exist_ok=True, parents=True)
    output_folder_bottom.mkdir(exist_ok=True, parents=True)

    for i, img_b, img_t, cm_b, cm_t in img:
        if img_b:
            img_b = img_b.convert('RGB')
            save_image_to_png(i, img_b, cm_b, output_folder_bottom, cam_id=1, name=logfile)

        if img_t:
            img_t = img_t.convert('RGB')
            save_image_to_png(i, img_t, cm_t, output_folder_top, cam_id=0, name=logfile)

        print("saving images from frame ", i)


def save_image_to_png(j, img, cm, target_dir, cam_id, name):
    meta = PngImagePlugin.PngInfo()
    meta.add_text("Message", "rotation maxtrix is saved column wise")
    meta.add_text("logfile", str(name))
    meta.add_text("CameraID", str(cam_id))

    if cm:
        meta.add_text("t_x", str(cm.pose.translation.x))
        meta.add_text("t_y", str(cm.pose.translation.y))
        meta.add_text("t_z", str(cm.pose.translation.z))

        meta.add_text("r_11", str(cm.pose.rotation[0].x))
        meta.add_text("r_21", str(cm.pose.rotation[0].y))
        meta.add_text("r_31", str(cm.pose.rotation[0].z))

        meta.add_text("r_12", str(cm.pose.rotation[1].x))
        meta.add_text("r_22", str(cm.pose.rotation[1].y))
        meta.add_text("r_32", str(cm.pose.rotation[1].z))

        meta.add_text("r_13", str(cm.pose.rotation[2].x))
        meta.add_text("r_23", str(cm.pose.rotation[2].y))
        meta.add_text("r_33", str(cm.pose.rotation[2].z))

    filename = target_dir / (str(j) + ".png")
    img.save(filename, pnginfo=meta)


def get_representations_from_log(event_name="2019-07-02_RC19"):
    """
        for example this is useful to find logs that contain images
    """
    event_root = Path(get_logs_root()) / event_name
    for game in sorted(event_root.iterdir()):
        print(game)
        game_log_folder = Path(get_logs_root()) / event_name / game / "game_logs"

        for robot in sorted(game_log_folder.iterdir()):
            # TODO save the result in a json file
            # TODO add images.log representations
            print(f"\t{robot}")
            gamelog = robot / "game.log"
            img_log = robot / "images.log"

            logged_representation = set()
            my_parser = Parser()

            log = LogReader(gamelog, my_parser)
            for i, frame in enumerate(log):
                dict_keys = frame.get_names()
                for key in dict_keys:
                    logged_representation.add(key)
                # only check the first few frames
                if i > 20:
                    break

            print(f"\t\t{logged_representation}")

        break
    return
    

def export_images_from_logs(event_name="2019-11-21_Koeln"):
    event_root = Path(get_logs_root()) / event_name
    # iterate over all games of the event
    for game in event_root.iterdir():
        print(game)
        for robot in (Path(game) / "game_logs").iterdir():
            print("\t", robot)
            # TODO add abort feature if images were already extracted
            gamelog_file = Path(robot) / "game.log"
            combined_log = Path(robot) / "combined.log"
            if combined_log.is_file():
                # export from combined log
                log = str(combined_log)
            elif gamelog_file.is_file() and os.stat(str(gamelog_file)).st_size > 0:
                # try to export from game.log
                log = str(gamelog_file)
                representation_set = get_representations_from_log(str(gamelog_file))
                print("\t\t", representation_set)
                if "Image" not in representation_set or "ImageTop" not in representation_set:
                    print("\t\tno images in game.log, will continue")
                    continue
                quit()
            else:
                print("\t\tno suitable logfile found")
                continue

            my_parser = Parser()
            with LogReader(log, my_parser) as reader:
                images = map(get_images, reader.read())
                export_images(log, images)


def create_image_log_dict(image_log):
    """
    Return a dictionary with frame numbers as key and (offset, size, is_camera_bottom) tuples of image data as values.
    """
    # parse image log
    width = 640
    height = 480
    bytes_per_pixel = 2
    image_data_size = width * height * bytes_per_pixel

    file_size = os.path.getsize(image_log)

    images_dict = dict()

    with open(image_log, 'rb') as f:
        is_camera_bottom = False  # assumes the first image is a top image
        while True:
            frame = f.read(4)
            if len(frame) != 4:
                break
            frame_number = int.from_bytes(frame, byteorder='little')

            offset = f.tell()
            f.seek(offset + image_data_size)

            # handle the case of incomplete image at the end of the logfile
            if f.tell() >= file_size:
                print("Info: frame {} in {} incomplete, missing {} bytes. Stop."
                      .format(frame_number, image_log, f.tell() + 1 - file_size))
                print("Info: Last frame seems to be incomplete.")
                break

            images_dict[frame_number] = (offset, image_data_size, is_camera_bottom)
            # next image is of the other cam
            is_camera_bottom = not is_camera_bottom

    return images_dict


get_representations_from_log()
