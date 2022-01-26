import cv2
from pathlib import Path

if __name__ == '__main__' :
    image_folder = "/home/stella/Documents/datasets/label_challenge/images"
    image_files = list(Path(image_folder).rglob('*.jpg'))

    tracker = cv2.TrackerCSRT_create()

    for idx, image in enumerate(sorted(image_files)):
        img = cv2.imread(str(image))
        # reinit the tracker every 200 frames
        if idx % 200 == 0:
            print(idx)
            
            # manually select a bounding box
            bbox = cv2.selectROI(img, False)
            # Initialize tracker with first frame and bounding box
            ok = tracker.init(img, bbox)
        else:
            # Update tracker
            ok, bbox = tracker.update(img)

        if ok:
            # Tracking success
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(img, p1, p2, (255,0,0), 2, 1)

        #print(idx, image)
        
        cv2.imshow("Tracking", img)

        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27 : break