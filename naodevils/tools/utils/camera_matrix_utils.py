import numpy as np
from utils import *

MAX_DIST = 15000


def get_focal_lenght(raw_image_width, opening_angle_width):
    focal_length = raw_image_width / (2.0 * np.tan(np.radians(opening_angle_width) / 2.0))
    focal_length_inv = 1.0 / focal_length
    return focal_length_inv


def imageToRobot(x, y, z, camera_rotation, camera_translation, opening_angle_width, raw_image_width, optical_center):
    inv_focal_length = get_focal_lenght(raw_image_width=raw_image_width, opening_angle_width=opening_angle_width)
    return imageToRobot(x, y, z, camera_rotation, camera_translation, optical_center, inv_focal_length)


def imageToRobot(x, y, z, camera_rotation, camera_translation, optical_center, inv_focal_length):
    # if not isinstance(camera_rotation, np.ndarray):
    #     camera_rotation = np.array(camera_rotation)
    # if not isinstance(camera_translation, np.ndarray):
    #     camera_translation = np.array(camera_translation)
    # if not isinstance(optical_center, np.ndarray):
    #     optical_center = np.array(optical_center)

    assert inv_focal_length < 5.0

    vectorToCenter = np.array([1.0, (optical_center[0] - x) * inv_focal_length, (optical_center[1] - y) * inv_focal_length])
    vectorToCenterWorld = np.dot(camera_rotation, vectorToCenter)

    f = (camera_translation[2] - z) / (vectorToCenterWorld[2] + 1e-10)
    rel_x = camera_translation[0] - f * vectorToCenterWorld[0]
    rel_y = camera_translation[1] - f * vectorToCenterWorld[1]

    # if vectorToCenterWorld[2] > (-2 * inv_focal_length):
    #     return np.array([0, 0])

    return np.array([rel_x, rel_y])


def getSizeByDistance(sizeInReality, distance, focal_length):
    xFactor = focal_length
    return sizeInReality / distance * xFactor


def calculate_horizon(rotation, translation, focal_length, optical_center):
    r31 = rotation[2, 0]
    r32 = rotation[2, 1]
    r33 = max(rotation[2, 2], 1e-8)

    v1 = focal_length
    v2 = optical_center[0]
    v3 = optical_center[1]

    y1 = (v3 * r33 + r31 * v1 + r32 * v2) / r33
    y2 = (v3 * r33 + r31 * v1 - r32 * v2) / r33

    return (y1 + y2) / 2.0