import cppyy
import os
from cppyy_tools import get_naoth_dir, get_toolchain_dir, setup_shared_lib
from typing import NamedTuple, Tuple, List
import numpy as np
from pathlib import Path
import PIL.Image
import sys
import ctypes
import json
from PIL import PngImagePlugin


def load_image_as_yuv422(image_filename):
    """
        this functions loads an image from a file to the correct format for the naoth library
    """
    # don't import cv globally, because the dummy simulator shared library might need to load a non-system library
    # and we need to make sure loading the dummy simulator shared library happens first
    import cv2
    cv_img = cv2.imread(image_filename)

    # convert image for bottom to yuv422
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2YUV).tobytes()
    yuv422 = np.ndarray(480 * 640 * 2, np.uint8)
    for i in range(0, 480 * 640, 2):
        yuv422[i * 2] = cv_img[i * 3]
        yuv422[i * 2 + 1] = (cv_img[i * 3 + 1] + cv_img[i * 3 + 4]) / 2.0
        yuv422[i * 2 + 2] = cv_img[i * 3 + 3]
        yuv422[i * 2 + 3] = (cv_img[i * 3 + 2] + cv_img[i * 3 + 5]) / 2.0
    return yuv422


class Point2D(NamedTuple):
    x: float
    y: float


class Rectangle(NamedTuple):
    top_left: Point2D
    bottom_right: Point2D

    def intersection_over_union(self, xtl: float, ytl: float, xbr: float, ybr: float):
        # compute x and y coordinates of the intersection rectangle
        intersection_topleft = Point2D(
            max(self.top_left[0], xtl), max(self.top_left[1], ytl))
        intersection_bottomright = Point2D(
            min(self.bottom_right[0], xbr), min(self.bottom_right[1], ybr))
        intersection = Rectangle(intersection_topleft,
                                 intersection_bottomright)

        # compute the intersection, self and other area
        intersection_area = max(0, intersection.bottom_right[0] - intersection.top_left[0] + 1) * \
                            max(0, intersection.bottom_right[1] - intersection.top_left[1] + 1)

        self_area = (self.bottom_right[0] - self.top_left[0] + 1) * \
                    (self.bottom_right[1] - self.top_left[1] + 1)
        other_area = (xbr - xtl + 1) * (ybr - ytl + 1)

        # return the intersecton over union
        return intersection_area / float(self_area + other_area - intersection_area)

    def get_radius(self):
        width = round((self.bottom_right[0] - self.top_left[0]) / 2)
        height = round((self.bottom_right[1] - self.top_left[1]) / 2)

        return min(width, height)

    def get_center(self):
        """
            this will calculate the center of the rectangle in the coordinate frame the coordinates are in
        """
        width = self.bottom_right[0] - self.top_left[0]
        height = self.bottom_right[1] - self.top_left[1]

        x = round(self.top_left[0] + width / 2)
        y = round(self.top_left[1] + height / 2)
        return x, y


class Frame(NamedTuple):
    file: str
    bottom: bool
    gt_balls: List[Rectangle]
    cam_matrix_translation: Tuple[float, float, float]
    cam_matrix_rotation: np.array


class PatchExecutor:
    """
    TODO add documentation here
    """

    def __init__(self):
        orig_working_dir = os.getcwd()
        naoth_dir = get_naoth_dir()
        toolchain_dir = get_toolchain_dir()

        setup_shared_lib(naoth_dir, toolchain_dir)

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

        # get access to the module manager and return it to the calling function
        self.moduleManager = cppyy.gbl.getModuleManager(cog)

        # get the ball detector module
        self.ball_detector = self.moduleManager.getModule("CNNBallDetector").getModule()

        # disable the modules providing the camera matrix, because we want to use our own
        self.moduleManager.getModule("CameraMatrixFinder").setEnabled(False)
        self.moduleManager.getModule("FakeCameraMatrixFinder").setEnabled(False)

        cppyy.cppdef("""
               Pose3D* toPose3D(CameraMatrix* m) { return static_cast<Pose3D*>(m); }
                """)

        # restore original working directory
        os.chdir(orig_working_dir)

    def get_frames_for_dir(self, d):
        """
            This code assumes that annotations are in coco format for now
        """
        # ----------------------------------------------------------------------------------------
        # Load COCO annotation data
        annotation_file = Path(self.output_folder) / "annotations/instances_default.json"
        print("annotation file", annotation_file)
        with open(annotation_file) as json_data:
            annotation_data = json.load(json_data)
        # ----------------------------------------------------------------------------------------
        file_names = os.listdir(d)
        file_names.sort()
        image_files = [os.path.join(d, f) for f in file_names if os.path.isfile(
            os.path.join(d, f)) and f.endswith(".png")]

        result = list()
        for file in image_files:
            # ----------------------------------------------------------------------------------------
            # actually load all the groundtruth ball data
            gt_balls = list()
            # get image id in coco annotation file for given image path
            image_path_anno = file.split("images/")[-1]
            for img in annotation_data["images"]:
                if img["file_name"] == image_path_anno:
                    id = img["id"]
            # use id to locate the bounding box annotation data
            for anno in annotation_data["annotations"]:
                # when multiple balls are present the if clause hits multiple times
                if anno["image_id"] == id:
                    top_left = (round(anno["bbox"][0]), round(anno["bbox"][1]))
                    bottom_right = (round(anno["bbox"][0] + anno["bbox"][2]), round(anno["bbox"][1] + anno["bbox"][3]))
                    gt_rectangle = Rectangle(top_left, bottom_right)
                    gt_balls.append(gt_rectangle)
            # ----------------------------------------------------------------------------------------
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

            frame = Frame(file, bottom, gt_balls, cam_matrix_translation, cam_matrix_rotation)
            result.append(frame)

        return result

    @staticmethod
    def set_camera_matrix_representation(frame, cam_matrix):
        """
            reads the camera matrix information from a frame object and writes it to the
            naoth camMatrix representation
        """
        p = cppyy.gbl.toPose3D(cam_matrix)
        p.translation.x = frame.cam_matrix_translation[0]
        p.translation.y = frame.cam_matrix_translation[1]
        p.translation.z = frame.cam_matrix_translation[2]

        for c in range(0, 3):
            for r in range(0, 3):
                p.rotation.c[c][r] = frame.cam_matrix_rotation[r, c]

        return p

    # helper: write a numpy array of data to an image representation
    @staticmethod
    def write_data_to_image_representation(data, image):
        # create a pointer
        p_data = data.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))

        # move image data into the Image representation that is defined in the Commons C++ project
        # the copyImageDataYUV422 function is defined there
        image.copyImageDataYUV422(p_data, data.size)

    def set_current_frame(self, frame):

        # get access to relevant representations
        image_bottom = self.ball_detector.getRequire().at("Image")
        image_top = self.ball_detector.getRequire().at("ImageTop")
        cam_matrix_bottom = self.ball_detector.getRequire().at("CameraMatrix")
        cam_matrix_top = self.ball_detector.getRequire().at("CameraMatrixTop")

        cam_matrix_bottom.valid = False
        cam_matrix_top.valid = False

        # load image in YUV422 format
        yuv422 = load_image_as_yuv422(frame.file)
        black = np.zeros(640 * 480 * 2, np.uint8)

        """
        get reference to the image input representation, if the current image is from the bottom camera
        we set the image for bottom image and the top image to black
        """
        if frame.bottom:
            self.write_data_to_image_representation(yuv422, image_bottom)
            self.write_data_to_image_representation(black, image_top)

            self.set_camera_matrix_representation(frame, cam_matrix_bottom)
            cam_matrix_bottom.valid = True
        else:  # image is from top camera
            self.write_data_to_image_representation(black, image_bottom)
            self.write_data_to_image_representation(yuv422, image_top)

            self.set_camera_matrix_representation(frame, cam_matrix_top)
            cam_matrix_bottom.valid = False

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

        # draw groundtruth
        for gt_ball in frame.gt_balls:
            cv2.rectangle(img, gt_ball.top_left, gt_ball.bottom_right, (0, 255, 0))

        output_file = self.output_folder / "debug_images" / Path(frame.file).name
        Path(output_file.parent).mkdir(exist_ok=True, parents=True)
        cv2.imwrite(str(output_file), img)

    def export_patches2(self, frame: Frame):
        """
            This should export patches as images for future training
        """
        import cv2

        # get the ball candidates from the module
        if frame.bottom:
            detected_balls = self.ball_detector.getProvide().at("BallCandidates")
            cam_id = 1
        else:
            detected_balls = self.ball_detector.getProvide().at("BallCandidatesTop")
            cam_id = 0

        img = cv2.imread(frame.file)
        # create folder for the patches

        patch_folder = self.output_folder / f"all_patches"
        Path(patch_folder).mkdir(exist_ok=True, parents=True)

        for idx, p in enumerate(detected_balls.patchesYUVClassified):
            iou = 0.0
            x = 0.0
            y = 0.0
            radius = 0.0
            # calculate the best iou for a given patch
            for gt_ball in frame.gt_balls:
                new_iou = gt_ball.intersection_over_union(p.min.x, p.min.y, p.max.x, p.max.y)
                if new_iou > iou:
                    iou = new_iou
                    # those values are relativ to the origin (top left) of the patch 
                    x, y = gt_ball.get_center()
                    x = x - p.min.x
                    y = y - p.min.y
                    radius = gt_ball.get_radius()


            # crop full image to calculated patch
            # TODO use naoth like resizing (subsampling) like in Patchwork.cpp line 39
            crop_img = img[p.min.y:p.max.y, p.min.x:p.max.x]
            # don't resize here. do it later
            #crop_img = cv2.resize(crop_img, (patch_size, patch_size), interpolation=cv2.INTER_NEAREST)

            patch_file_name = patch_folder / (Path(frame.file).stem + f"_{idx}.png")
            cv2.imwrite(str(patch_file_name), crop_img)

            # section for writing meta data
            meta = PngImagePlugin.PngInfo()
            meta.add_text("CameraID", str(cam_id))
            meta.add_text("iou", str(iou))
            meta.add_text("center_x", str(x))
            meta.add_text("center_y", str(y))
            meta.add_text("radius", str(radius))

            with PIL.Image.open(str(patch_file_name)) as im_pill:
                im_pill.save(str(patch_file_name), pnginfo=meta)

    @staticmethod
    def get_output_folder(directory):
        """
            TODO can this be done cooler?
            finds the parent folder of obj_train_data. In this folder new folders for various output are created.
            This assumes we have yolo output
        """
        # create output path for yolo input
        for parent_folder in Path(directory).parents:
            if parent_folder.name == "obj_train_data":
                return parent_folder.parent

        # create output path for coco input structure
        for parent_folder in Path(directory).parents:
            if parent_folder.parent.name == "COCO_1.0":
                return parent_folder

        print("arg")
        sys.exit()

    def execute(self, directories):
        for d in sorted(directories):
            print("working in", d)
            self.output_folder = self.get_output_folder(d)
            frames = self.get_frames_for_dir(d)

            for f in frames:
                self.set_current_frame(f)
                self.sim.executeFrame()
                # self.export_debug_images(f)
                self.export_patches2(f)
