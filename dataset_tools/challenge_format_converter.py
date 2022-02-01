"""
    Converts cvat image xml files to the format specified in the 2022 label challenge
"""
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

def main():
    xml_label_file = "cvat_images_with_attributes_annotations.xml"

    with open('naoth_annotations.csv', 'w') as f:
        # print('This message will be written to a file.', file=f)

        tree = ET.parse(xml_label_file)
        annotation_root = tree.getroot()

        for image in annotation_root:
            # ignore meta and version childs in the xml structure
            if image.tag != "image":
                continue
            print(image.attrib)
            image_name = Path(image.attrib["name"]).name  # TODO parse only the filename not the whole path
            for bbox in image:
                if bbox.attrib["occluded"] != "0":
                    # don't export occluded bounding boxes
                    continue
                
                if bbox.attrib["label"] == "nao":
                    label = 0
                    for item in bbox.findall("attribute"):
                        if item.attrib["name"] == "color":
                            color = item.text
                        if item.attrib["name"] == "number":
                            number = item.text                    
                elif bbox.attrib["label"] == "ball":
                    label = 1
                    color = -1
                    number = -1                    
                else:
                    print("ERROR: there was an unexpected label")
                    sys.exit(-1)

                xtl = bbox.attrib["xtl"]
                ytl = bbox.attrib["ytl"]
                xbr = bbox.attrib["xbr"]
                ybr = bbox.attrib["ybr"]
                print(image_name, label, xtl, ytl, xbr, ybr, color, number, file=f, sep=",")
                

if __name__ == "__main__":
    main()