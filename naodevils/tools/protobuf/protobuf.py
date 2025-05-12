import png
import cv2
from protobuf.imageLabelData_pb2 import ImageLabelData

def read_label_chunk(image_data):
    image_data.seek(0)
    p = png.Reader(image_data)
    for chunk_name, chunk_data in p.chunks():
        if chunk_name == 'laBl' or chunk_name == b'laBl':
            return chunk_data

def write_label_chunk(image_path, encoded_protobuf):
    with open(image_path, "rb") as image_data:
        image_data.seek(0)
        p = png.Reader(image_data)

        chunk_list = []

        index = None

        for i, (chunk_name, chunk_data) in enumerate(p.chunks()):
            chunk_list.append([chunk_name, chunk_data])
            if chunk_name == 'laBl' or chunk_name == b'laBl':
                index = i

        if index is not None:
            chunk_list[index][1] = encoded_protobuf
        else:
            chunk_list.insert(-1, [b'laBl', encoded_protobuf])

    with open(image_path, 'wb') as f:
        png.write_chunks(f, chunk_list)

def decode_data(image_path):
    encoded_data = read_label_chunk(image_path)
    decoded_data = ImageLabelData()
    if encoded_data is not None:
        decoded_data.ParseFromString(encoded_data)
    return decoded_data

def encode_data(protobuf):
    return protobuf.SerializeToString()

def copy_encoded_protobuf(image_source, image_target):
    protobuf_data = read_label_chunk(image_source)

    image_target.seek(0)
    p = png.Reader(image_target)

    chunk_list = []

    index = None

    for i, (chunk_name, chunk_data) in enumerate(p.chunks()):
        chunk_list.append([chunk_name, chunk_data])
        if chunk_name == 'laBl':
            index = i

    if index is not None:
        chunk_list[index][1] = protobuf_data
    else:
        chunk_list.insert(-1, ['laBl', protobuf_data])

    image_target.seek(0)
    png.write_chunks(image_target, chunk_list)

    return image_target

def read_bboxes(name, objects, scale=1.0, filename=None, width=None, height=None):
    bboxes = []
    updated = False
    for object in objects:
        obj = {}
        obj["name"] = name
        try:
            try:
                obj["teamColor"] = object.teamcolor
            except:
                obj["teamColor"] = 0

            if object.label != None:
                object = object.label
        except:
            print("Old Annotation Format!")

        try:
            obj["xmin"] = object.boundingBox.upperLeft.x
            obj["xmax"] = object.boundingBox.lowerRight.x
            obj["ymin"] = object.boundingBox.upperLeft.y
            obj["ymax"] = object.boundingBox.lowerRight.y

            if not scale == 1.0:
                w = obj["xmax"] - obj["xmin"]
                h = obj["ymax"] - obj["ymin"]
                x = (obj["xmax"] + obj["xmin"]) / 2
                y = (obj["ymax"] + obj["ymin"]) / 2
                w *= scale
                h *= scale
                obj["xmin"] = round(x - (w/2))
                obj["xmax"] = round(x + (w/2))
                obj["ymin"] = round(y - (h / 2))
                obj["ymax"] = round(y + (h / 2))
        except:
            obj["xmin"] = -1
            obj["xmax"] = -1
            obj["ymin"] = -1
            obj["ymax"] = -1

        try:
            obj["concealed"] = object.concealed
        except:
            obj["concealed"] = False

        try:
            obj["blurriness"] = object.blurriness
        except:
            obj["blurriness"] = -1

        try:
            obj["visibilityLevel"] = object.visibilityLevel
        except:
            obj["visibilityLevel"] = 0

        try:
            obj["id"] = object.id
        except:
            obj["id"] = 0

        if filename and obj['blurriness'] <= 0:
            image = cv2.imread(filename)
            cropped_image = image[
                            int(max(0.0, obj['xmin'])): int(min(image.shape[1], obj['xmax'])),
                            int(max(0.0, obj['ymin'])): int(min(image.shape[0], obj['ymax']))
                            ]
            if cropped_image.shape[0] > 2 and cropped_image.shape[1] > 2:
                gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
                blurriness = cv2.Laplacian(gray, cv2.CV_64F).var() / (len(image) / len(cropped_image))
                obj['blurriness'] = int(round(blurriness))
            else:
                obj['blurriness'] = 1001

            object.blurriness = obj['blurriness']
            updated = True

        if width and height and not obj['concealed'] and \
                (obj['xmin'] < 0.0 or obj['ymin'] < 0.0 or obj['xmax'] > width or obj['ymax'] > height):
            bbox_width = obj['xmax'] - obj['xmin']
            bbox_height = obj['ymax'] - obj['ymin']

            outside_width = 0
            outside_height = 0
            if obj['xmin'] < 0.0:
                outside_width += -obj['xmin']
            if obj['ymin'] < 0.0:
                outside_height += -obj['ymin']
            if obj['xmax'] > width:
                outside_width += obj['xmax'] - width
            if obj['ymax'] > height:
                outside_height += obj['ymax'] - height

            area_percent = ((bbox_width - outside_width) * (bbox_height - outside_height)) / (bbox_width * bbox_height)
            if area_percent >= 0.999:
                if obj['visibilityLevel'] != 0:
                    obj['visibilityLevel'] = 0
            elif area_percent >= 0.75:
                if obj['visibilityLevel'] != 1:
                    obj['visibilityLevel'] = 1
            elif area_percent >= 0.50:
                if obj['visibilityLevel'] != 2:
                    obj['visibilityLevel'] = 2
            elif area_percent >= 0.25:
                if obj['visibilityLevel'] != 3:
                    obj['visibilityLevel'] = 3
            elif area_percent > 0.0:
                if obj['visibilityLevel'] != 4:
                    obj['visibilityLevel'] = 4
            else:
                if obj['visibilityLevel'] != 5:
                    obj['visibilityLevel'] = 5

            if object.visibilityLevel != obj['visibilityLevel']:
                object.visibilityLevel = obj['visibilityLevel']
                updated = True

        if not (obj["xmin"] == -1 and obj["xmax"] == -1 and obj["ymin"] == -1 and obj["ymax"] == -1):
            bboxes.append(obj)

    return updated, bboxes

def clearProtobufAnotations(decoded_protobuf):
    while len(decoded_protobuf.robots) > 0:
        del decoded_protobuf.robots[-1]

    while len(decoded_protobuf.obstacles) > 0:
        del decoded_protobuf.obstacles[-1]

    while len(decoded_protobuf.balls) > 0:
        del decoded_protobuf.balls[-1]

    while len(decoded_protobuf.goalposts) > 0:
        del decoded_protobuf.goalposts[-1]

    while len(decoded_protobuf.penaltyCrosses) > 0:
        del decoded_protobuf.penaltyCrosses[-1]

    while len(decoded_protobuf.DEPRECATED_penaltyCrosses) > 0:
        del decoded_protobuf.DEPRECATED_penaltyCrosses[-1]

    while len(decoded_protobuf.lineCrossing) > 0:
        del decoded_protobuf.lineCrossing[-1]

    while len(decoded_protobuf.centerCircle) > 0:
        del decoded_protobuf.centerCircle[-1]

    return decoded_protobuf