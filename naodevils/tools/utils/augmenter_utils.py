import imgaug
import numpy as np

from utils.camera_matrix_utils import calculate_horizon
from bisect import bisect_left

class ResizeWithCameraMatrix(object):
    def __init__(self, size):
        self.size, self.size_order = self._handle_size_arg(size, False)
        self.interpolation = imgaug.parameters.Deterministic("nearest")

    @classmethod
    def _handle_size_arg(cls, size, subcall):
        def _dict_to_size_tuple(v1, v2):
            kaa = "keep-aspect-ratio"
            not_both_kaa = (v1 != kaa or v2 != kaa)
            assert not_both_kaa, (
                "Expected at least one value to not be \"keep-aspect-ratio\", "
                "but got it two times.")

            size_tuple = []
            for k in [v1, v2]:
                if k in ["keep-aspect-ratio", "keep"]:
                    entry = imgaug.parameters.Deterministic(k)
                else:
                    entry = cls._handle_size_arg(k, True)
                size_tuple.append(entry)
            return tuple(size_tuple)

        def _contains_any_key(dict_, keys):
            return any([key in dict_ for key in keys])

        # HW = height, width
        # SL = shorter, longer
        size_order = "HW"

        if size == "keep":
            result = imgaug.parameters.Deterministic("keep")
        elif imgaug.is_single_number(size):
            assert size > 0, "Expected only values > 0, got %s" % (size,)
            result = imgaug.parameters.Deterministic(size)
        elif not subcall and isinstance(size, dict):
            if len(size.keys()) == 0:
                result = imgaug.parameters.Deterministic("keep")
            elif _contains_any_key(size, ["height", "width"]):
                height = size.get("height", "keep")
                width = size.get("width", "keep")
                result = _dict_to_size_tuple(height, width)
            elif _contains_any_key(size, ["shorter-side", "longer-side"]):
                shorter = size.get("shorter-side", "keep")
                longer = size.get("longer-side", "keep")
                result = _dict_to_size_tuple(shorter, longer)
                size_order = "SL"
            else:
                raise ValueError(
                    "Expected dictionary containing no keys, "
                    "the keys \"height\" and/or \"width\", "
                    "or the keys \"shorter-side\" and/or \"longer-side\". "
                    "Got keys: %s." % (str(size.keys()),))
        elif isinstance(size, tuple):
            assert len(size) == 2, (
                "Expected size tuple to contain exactly 2 values, "
                "got %d." % (len(size),))
            assert size[0] > 0 and size[1] > 0, (
                "Expected size tuple to only contain values >0, "
                "got %d and %d." % (size[0], size[1]))
            if imgaug.is_single_float(size[0]) or imgaug.is_single_float(size[1]):
                result = imgaug.parameters.Uniform(size[0], size[1])
            else:
                result = imgaug.parameters.DiscreteUniform(size[0], size[1])
        elif isinstance(size, list):
            if len(size) == 0:
                result = imgaug.parameters.Deterministic("keep")
            else:
                all_int = all([imgaug.is_single_integer(v) for v in size])
                all_float = all([imgaug.is_single_float(v) for v in size])
                assert all_int or all_float, (
                    "Expected to get only integers or floats.")
                assert all([v > 0 for v in size]), (
                    "Expected all values to be >0.")
                result = imgaug.parameters.Choice(size)
        elif isinstance(size, imgaug.parameters.StochasticParameter):
            result = size
        else:
            raise ValueError(
                "Expected number, tuple of two numbers, list of numbers, "
                "dictionary of form "
                "{'height': number/tuple/list/'keep-aspect-ratio'/'keep', "
                "'width': <analogous>}, dictionary of form "
                "{'shorter-side': number/tuple/list/'keep-aspect-ratio'/"
                "'keep', 'longer-side': <analogous>} "
                "or StochasticParameter, got %s." % (type(size),)
            )

        if subcall:
            return result
        return result, size_order

    def to_deterministic(self, n=None):
        return self

    def __call__(self, *args, **kwargs):
        image = kwargs["image"]
        bboxes = kwargs["bounding_boxes"]

        camera_intrinsics = kwargs["camera_intrinsics"]

        rotation = camera_intrinsics["rotation"]
        translation = camera_intrinsics["translation"]
        focal_length = camera_intrinsics["focal_length"]
        optical_center = camera_intrinsics["optical_center"]

        y = calculate_horizon(rotation, translation, focal_length, optical_center)
        horizon_y = y if y > 0 else 0

        lower = horizon_y + 4 if horizon_y > 1 else horizon_y
        upper = image.shape[0]
        length = self.size[0].value
        downscaled_height_list = [int(round(lower + x*(upper-lower)/length)) for x in range(length)]
        quarter = int(len(downscaled_height_list) / 6)
        downscaled_height_list_temp = downscaled_height_list.copy()

        for i in range(quarter):
            first = downscaled_height_list_temp[i]
            second = downscaled_height_list_temp[i+1]
            new_value = int(np.floor((first + second) / 2))
            downscaled_height_list.pop(len(downscaled_height_list) - 1 - i)
            downscaled_height_list.insert(i+i+1, new_value)

        scaled_image = np.zeros((self.size[0].value, self.size[1].value, 3), np.uint8)
        width_factor = int(image.shape[1] / self.size[1].value)
        for i, r in enumerate(downscaled_height_list):
            scaled_image[i, :, :] = image[r, 0:image.shape[1]-1:width_factor, :]

        new_bboxes = []
        for bbox in bboxes.bounding_boxes:
            new_x1 = bbox.x1 / width_factor
            new_x2 = bbox.x2 / width_factor
            new_y1 = downscaled_height_list.index(self.take_closest(downscaled_height_list, bbox.y1))
            new_y2 = downscaled_height_list.index(self.take_closest(downscaled_height_list, bbox.y2))
            new_y2_orig = bbox.y2 / width_factor
            new_bboxes.append(bbox.copy(x1=new_x1, x2=new_x2, y1=new_y1, y2=new_y2, label=bbox.label.update({"new_y2": new_y2_orig})))

        bboxesOI = imgaug.BoundingBoxesOnImage(new_bboxes, (self.size[0].value, self.size[1].value))

        return scaled_image, bboxesOI

    def take_closest(self, myList, myNumber):
        """
        Assumes myList is sorted. Returns closest value to myNumber.

        If two numbers are equally close, return the smallest number.
        """
        pos = bisect_left(myList, myNumber)
        if pos == 0:
            return myList[0]
        if pos == len(myList):
            return myList[-1]
        before = myList[pos - 1]
        after = myList[pos]
        if after - myNumber < myNumber - before:
            return after
        else:
            return before

    def get_size_by_distance(self, sizeInReality, distance, focal_length):
        return sizeInReality / distance * focal_length