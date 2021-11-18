import os
import sys
import argparse
import csv
import cppyy
import PIL.Image
from typing import NamedTuple, Tuple, List
import numpy as np
import time
import ctypes
import xml.etree.ElementTree as ET
from cppyy import addressof, bind_object
import cppyy.ll
from cppyy_tools import *
from pathlib import Path



class Point2D(NamedTuple):
    x: float
    y: float


class Rectangle(NamedTuple):
    top_left: Point2D
    bottom_right: Point2D

    def intersection_over_union(self, xtl: float, ytl: float, xbr: float, ybr: float):

        # compute x and y coordinates of the intersection rectangle
        intersection_topleft = Point2D(
            max(self.top_left.x, xtl), max(self.top_left.y, ytl))
        intersection_bottomright = Point2D(
            min(self.bottom_right.x, xbr), min(self.bottom_right.y, ybr))
        intersection = Rectangle(intersection_topleft,
                                 intersection_bottomright)

        # compute the intersection, self and other area
        intersectionArea = max(0, intersection.bottom_right.x - intersection.top_left.x + 1) * \
            max(0, intersection.bottom_right.y - intersection.top_left.y + 1)

        selfArea = (self.bottom_right.x - self.top_left.x + 1) * \
            (self.bottom_right.y - self.top_left.y + 1)
        otherArea = (xbr - xtl + 1) * (ybr - ytl + 1)

        # return the intersecton over union
        return intersectionArea / float(selfArea + otherArea - intersectionArea)


class Frame(NamedTuple):
    file: str
    bottom: bool
    balls: List[Rectangle]
    cam_matrix_translation: Tuple[float, float, float]
    cam_matrix_rotation: np.array


def get_frames_for_dir(d, transform_to_squares=False):
    file_names = os.listdir(d)
    file_names.sort()
    imageFiles = [os.path.join(d, f) for f in file_names if os.path.isfile(
        os.path.join(d, f)) and f.endswith(".png")]

    result = list()
    for file in imageFiles:

        # open image to get the metadata
        img = PIL.Image.open(file)
        bottom = img.info["CameraID"] == "1"
        # parse camera matrix using metadata in the PNG file
        cam_matrix_translation = (float(img.info["t_x"]), float(
            img.info["t_y"]), float(img.info["t_z"]))

        cam_matrix_rotation = np.array(
            [
                [float(img.info["r_11"]), float(
                    img.info["r_12"]), float(img.info["r_13"])],
                [float(img.info["r_21"]), float(
                    img.info["r_22"]), float(img.info["r_23"])],
                [float(img.info["r_31"]), float(
                    img.info["r_32"]), float(img.info["r_33"])]
            ])

        # add ground truth (all actual balls) to the frame
        balls = list()

        frame = Frame(file, bottom, balls, cam_matrix_translation,
                      cam_matrix_rotation)

        result.append(frame)

    return result


def load_image(image_filename):
    """
        this functions loads an image in the correct format for the naoth library
    """
    # don't import cv globally, because the dummy simulator shared library might need to load a non-system library
    # and we need to make sure loading the dummy simulator shared library happens first
    import cv2
    cv_img = cv2.imread(image_filename)

    # convert image for bottom to yuv422
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2YUV).tobytes()
    yuv422 = np.ndarray(480*640*2, np.uint8)
    for i in range(0, 480 * 640, 2):
        yuv422[i*2] = cv_img[i*3]
        yuv422[i*2 + 1] = (cv_img[i*3 + 1] + cv_img[i*3 + 4]) / 2.0
        yuv422[i*2 + 2] = cv_img[i*3 + 3]
        yuv422[i*2 + 3] = (cv_img[i*3 + 2] + cv_img[i*3 + 5]) / 2.0
    return yuv422





class PatchExecutor:

    def __init__(self):
        naoth_dir = get_naoth_dir()
        toolchain_dir = get_toolchain_dir()

        # load shared library: all depending libraries should be found automatically
        shared_lib_name = "libscriptsim.so"
        if sys.platform.startswith("win32"):
            shared_lib_name = "scriptsim.dll"
        elif sys.platform.startswith("darwin"):
            shared_lib_name = "libscriptsim.dylib"

        cppyy.load_library(os.path.join(
            naoth_dir, "NaoTHSoccer/dist/Native/" + shared_lib_name))

        # add relevant include paths to allow mapping our code
        cppyy.add_include_path(os.path.join(
            naoth_dir, "Framework/Commons/Source"))
        cppyy.add_include_path(os.path.join(naoth_dir, "NaoTHSoccer/Source"))
        cppyy.add_include_path(os.path.join(
            toolchain_dir, "toolchain_native/extern/include"))

        cppyy.add_include_path(os.path.join(
            toolchain_dir, "toolchain_native/extern/include/glib-2.0"))
        cppyy.add_include_path(os.path.join(
            toolchain_dir, "toolchain_native/extern/lib/glib-2.0/include"))

        # include platform
        cppyy.include(os.path.join(
            naoth_dir, "Framework/Commons/Source/PlatformInterface/Platform.h"))
        cppyy.include(os.path.join(
            naoth_dir, "Framework/Platforms/Source/DummySimulator/DummySimulator.h"))

        # change working directory so that the configuration is found
        orig_working_dir = os.getcwd()
        os.chdir(os.path.join(naoth_dir, "NaoTHSoccer"))

        # start dummy simulator
        cppyy.gbl.g_type_init()
        self.sim = cppyy.gbl.DummySimulator(False, False, 5401)
        cppyy.gbl.naoth.Platform.getInstance().init(self.sim)

        # create the cognition and motion objects
        cog = cppyy.gbl.createCognition()
        mo = cppyy.gbl.createMotion()

        # cast to callable
        callable_cog = cppyy.bind_object(
            cppyy.addressof(cog), cppyy.gbl.naoth.Callable)
        callable_mo = cppyy.bind_object(
            cppyy.addressof(mo), cppyy.gbl.naoth.Callable)

        self.sim.registerCognition(callable_cog)
        self.sim.registerMotion(callable_mo)

        # make more representations available to cppyy
        cppyy.include(os.path.join(
            naoth_dir, "Framework/Commons/Source/ModuleFramework/ModuleManager.h"))
        cppyy.include(os.path.join(
            naoth_dir, "NaoTHSoccer/Source/Cognition/Cognition.h"))
        cppyy.include(os.path.join(
            naoth_dir, "NaoTHSoccer/Source/Representations/Perception/BallCandidates.h"))
        cppyy.include(os.path.join(
            naoth_dir, "NaoTHSoccer/Source/Representations/Perception/CameraMatrix.h"))

        # get access to the module manager and return it to the calling function
        self.moduleManager = cppyy.gbl.getModuleManager(cog)

        # get the ball detector module
        self.ball_detector = self.moduleManager.getModule(
            "CNNBallDetector").getModule()

        cppyy.cppdef("""
               Pose3D* my_convert(CameraMatrix* m) { return static_cast<Pose3D*>(m); }
                """)

        # restore original working directory
        os.chdir(orig_working_dir)

    def set_current_frame(self, frame):
        # get reference to the image input representation
        if frame.bottom:
            imageRepresentation = self.ball_detector.getRequire().at("Image")
            # set other image to black
            black = np.zeros(640*480*2, np.uint8)
            self.ball_detector.getRequire().at("ImageTop").copyImageDataYUV422(
                black.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), black.size)
        else:
            imageRepresentation = self.ball_detector.getRequire().at("ImageTop")
            # set other image to black
            black = np.zeros(640*480*2, np.uint8)
            self.ball_detector.getRequire().at("Image").copyImageDataYUV422(
                black.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), black.size)

        # load image in YUV422 format
        yuv422 = load_image(frame.file)
        p_data = yuv422.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))

        # move image into representation
        imageRepresentation.copyImageDataYUV422(p_data, yuv422.size)
        # set camera matrix
        if frame.bottom:
            camMatrix = self.ball_detector.getRequire().at("CameraMatrix")
            # invalidate other camera matrix
            self.ball_detector.getRequire().at("CameraMatrixTop").valid = False
        else:
            camMatrix = self.ball_detector.getRequire().at("CameraMatrixTop")
            # invalidate other camera matrix
            self.ball_detector.getRequire().at("CameraMatrix").valid = False      

        p = cppyy.gbl.my_convert(camMatrix)
        p.translation.x = frame.cam_matrix_translation[0]
        p.translation.y = frame.cam_matrix_translation[1]
        p.translation.z = frame.cam_matrix_translation[2]

        for c in range(0, 3):
            for r in range(0, 3):
                p.rotation.c[c][r] = frame.cam_matrix_rotation[r, c]

    def export_debug_images(self, frame: Frame):
        """
            this function exports the input images with the calculated patches overlayed
        """
        import cv2

        # get the ball candidates from the module
        if frame.bottom:
            detected_balls = self.ball_detector.getProvide().at("BallCandidates")
        else:
            detected_balls = self.ball_detector.getProvide().at("BallCandidatesTop")

        img = cv2.imread(frame.file)
        for p in detected_balls.patchesYUVClassified:
            cv2.rectangle(img, (p.min.x, p.min.y), (p.max.x, p.max.y), (0, 0, 255))

        output_file = self.output_folder / "debug_images" / Path(frame.file).name
        Path(output_file.parent).mkdir(exist_ok=True, parents=True)
        cv2.imwrite(str(output_file), img)

    def export_patches():
        """
            This should export patches as images for future training
        """
        pass

    def get_output_folder(self, directory):
        """
            TODO can this be done cooler?
            finds the parent folder of obj_train_data. In this folder new folders for various output are created.
            This assumes we have yolo output
        """
        for parent_folder in Path(directory).parents:
            if parent_folder.name == "obj_train_data":
                return parent_folder.parent

        print("arg")
        sys.exit()

    def execute(self, directories):
        for d in directories:
            frames = get_frames_for_dir(d, False)
            self.output_folder = self.get_output_folder(d)

            # HACK: run first frame twice
            for f in frames:
                self.set_current_frame(f)
                self.sim.executeFrame()
                self.export_debug_images(f)
                break

            for f in frames:
                self.set_current_frame(f)
                self.sim.executeFrame()
                self.export_debug_images(f)


if __name__ == "__main__":
    evaluator = PatchExecutor()
    with cppyy.ll.signals_as_exception():
        bla = ["/home/stella/RoboCup/Repositories/naoth-deeplearning/data_cvat/RoboCup2019/finished/7/obj_train_data/rc19-experiment/INDOOR/GOALY_SET_top/"]
        evaluator.execute(bla)

