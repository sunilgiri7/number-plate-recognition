from ultralytics import YOLO
import cv2
import util
from sort.sort import *
from util import get_car, read_license_plate, write_csv
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

result = {}
mot_tracker = Sort()

#load models
coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO('./best.pt')
vechicles = [2,3,5,7]

cap = cv2.VideoCapture('./sample.mp4')
frame_nmr = -1
ret = True
while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    if ret:
        result[frame_nmr] = {}
        detect = coco_model(frame)[0]
        detections_ = []
        for detection in detect.boxes.data.tolist():
            x1,y1,x2,y2,score,class_id = detection
            if int(class_id) in vechicles:
                detections_.append([x1,y1,x2,y2,score])

        track_ids = mot_tracker.update(np.asarray(detections_))
        license_plates = license_plate_detector(frame)[0]

        for license_plate in license_plates.boxes.data.tolist():
            x1,y1,x2,y2,score,class_id = license_plate

            xcar1, ycar1, xcar2, ycar2, car_id= get_car(license_plate, track_ids)

            if car_id != -1:
                license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]

                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

                license_plate_test, license_plate_test_score = read_license_plate(license_plate_crop_thresh)
                if license_plate_test is not None:
                    result[frame_nmr][car_id] = {'car':{'bbox': [xcar1, ycar1, xcar2, ycar2]}, 
                                                'license_plate':{'bbox': [x1,y1,x2,y2],
                                                                'text': license_plate_test,
                                                                    'bbox_score': score,
                                                                    'text_score': license_plate_test_score}}
                # cv2.imshow('orginal_crop', license_plate_crop)
                # cv2.imshow('threshold', license_plate_crop_thresh)

                # cv2.waitKey(0)
write_csv(result, './test.csv')