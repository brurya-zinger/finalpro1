import time

import cv2
import numpy as np
net = cv2.dnn.readNetFromDarknet('yolov3.cfg','yolov3.weights')
ln = net.getLayerNames()
output_layers_names = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
# net = cv2.dnn.readNet('yolov3.weights','yolov3.cfg')
classes = []
with open('coco.txt','r') as f:
    classes = f.read().splitlines()

pathOut = 'roads_v2.mp4'
cap = cv2.VideoCapture('video1.mp4')
# cap = cv2.VideoCapture(0)

# img = cv2.imread('image.jpg')

# from _collections import deque
# pts = [deque(maxlen=30) for _ in range(1000)]
frame_list = []
out = None
fps = 30.0

object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)
while True:
    _, img = cap.read()
    # roi = img[340: 720, 500: 800]
    # mask = object_detector.apply(roi)
    # _, mask = cv2.threshold(roi, 254, 255, cv2.THRESH_BINARY)
    # contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


    height, width, _ = img.shape
    print(width,height)
    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (256, 256),
                                 swapRB=True, crop=False)
    # blob = cv2.dnn.blobFromImage(img, 1/255, (224,224), (0,0,0), swapRB=True, crop=False)

    net.setInput(blob)
    layerOutputs = net.forward(output_layers_names)


    # output_layers_names = net.getUnconnectedOutLayersNames()

    # layerOutputs = net.forward()

    # layerOutputs = net.forward(output_layers_names)


    boxes = []
    confidences = []
    class_ids = []
    counter = int(0)

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]

            class_id = np.argmax(scores)

            confidence = scores[class_id]
            if confidence > 0.5:

                center_x = int(detection[0]*width)
                center_y =  int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)
                x = int(center_x - w/2)
                y = int(center_y - h/2)
                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)


    cv2.line(img, (0, int(3 * height / 6 - height / 20)), (width, int(3 * height / 6 - height / 20)), (0, 255, 255), thickness=3)  # draw line
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0,255, size=(len(boxes), 3))
    if len(indexes) != 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i], 2))
            color = colors[i]
            cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
            cv2.putText(img, label + " " + confidence, (x,y+20), font, 2, (255,255,255),2 )
        if y <= int(3 * height / 6 + height / 20) and y >= int(3 * height / 6 - height / 20):
            if label == 'car' or label == 'truck':
                print('in if',label,y)
                counter=+1

    cv2.putText(img, 'Total Vehicle Count: ' + str(counter), (0, 80), 0, 1, (0, 0, 255), 5)
    frame_list.append(img)
    cv2.imshow('Image', img)
    size = (width, height)
    key = cv2.waitKey(1)
    if key==27:
        break
out = cv2.VideoWriter(pathOut, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
for i in range(len(frame_list)):
        # writing to a image array
        out.write(frame_list[i])
cap.release()
out.release()
cv2.destroyAllWindows()



















# import cv2
# # from absl import app, flags, logging
# import xlwt
# from xlwt import Workbook

# import numpy as np
# #
# #
# # wb = Workbook()
# # sheet1 = wb.add_sheet('Sheet 1')
#
# net = cv2.dnn.readNet('yolov3.weights','yolov3.cfg')
#
# classes = []
# with open('coco.txt','r') as f:
#     classes = f.read().splitlines()
#
#
# cap = cv2.VideoCapture('video.mp4')
#
#
# # cap = cv2.VideoCapture(0)
#
# # img = cv2.imread('image.jpg')
#
#
# # output path to save the video
# pathOut = 'roads_v2.mp4'
#
# # specify frames per second
# fps = 30.0
#
# frame_list = []
# while True:
#     _, img = cap.read()
#
#
#     height, width, _ = img.shape
#
#
#
#     # resized = cv2.resize(img, (224, 224))
#     # blob = cv2.dnn.blobFromImage(resized, 1, (224, 224), (100, 100, 100), swapRB=True, crop=False)
#     blob = cv2.dnn.blobFromImage(img, 1/255, (224,224), (0,0,0), swapRB=True, crop=False)
#
#     net.setInput(blob)
#
#     output_layers_names = net.getUnconnectedOutLayersNames()
#     layerOutputs = net.forward(output_layers_names)
#
#
#     # out = None
#
#     boxes = []
#     confidences = []
#     class_ids = []
#
#     current_count = int(0)
#     print('gggg')
#
#
#     for output in layerOutputs:
#         for detection in output:
#             scores = detection[5:]
#
#             class_id = np.argmax(scores)
#
#             confidence = scores[class_id]
#             # print('classId0', class_id)
#
#             if confidence > 0.5:
#                 print('in if')
#
#                 center_x = int(detection[0]*width)
#                 center_y = int(detection[1]*height)
#                 w = int(detection[2]*width)
#                 h = int(detection[3]*height)
#                 x = int(center_x - w/2)
#                 y = int(center_y - h/2)
#                 boxes.append([x, y, w, h])
#                 print('boxes', boxes)
#                 confidences.append((float(confidence)))
#                 print('classId', class_id)
#                 class_ids.append(class_id)
#
#             # print('classIds',class_ids)
#             cv2.line(img, (0, int(3 * height / 6 + height / 20)), (width, int(3 * height / 6 + height / 20)),(0, 255, 255),thickness=3)  # draw line
#             cv2.line(img, (0, int(3 * height / 6 - height / 20)), (width, int(3 * height / 6 - height / 20)),(0, 255, 255),thickness=3)  # draw line
#             center_y = int(detection[1] * height)
#             h = int(detection[3] * height)
#             y = int(center_y - h / 2)
#
#
#
#
#     indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
#
#     font = cv2.FONT_HERSHEY_PLAIN
#     colors = np.random.uniform(0,255, size=(len(boxes), 3))
#
#
#
#
#
#
#
#
#     for i in indexes.flatten():
#         x, y, w, h = boxes[i]
#         label = str(classes[class_ids[i]])
#         confidence = str(round(confidences[i], 2))
#         color = colors[i]
#         img = cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
#         cv2.putText(img, label + " " + confidence, (x,y+20), font, 2, (255,255,255),2 )
#         if y <= int(3 * height / 6 + height / 20) and y >= int(3 * height / 6 - height / 20):  # range
#             if label == 'car' or label == 'truck':
#                 # current_count[class_id] = current_count.get(classes[class_ids[0]], 0) + 1
#                 # sheet1.write(i, 0, 'car/truck'
#                 # sheet1.write(i, 0, 'ISBT DEHRADUN')
#
#                 current_count += 1
#         # result = np.asarray(img)
#
#         # print(boxes[i])
#
#     cv2.putText(img, 'Total current_count: ' + str(current_count), (0, 130), 0, 1, (0, 0, 255), 5)
#     # total_count = len(set(counter))
#     # cv2.putText(img, 'Total Vehicle Count: ' + str(counter), (0,80), 0, 1, (0,0,255),5)
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#     cv2.imshow('Image', img)
#     height, width, layers = img.shape
#     size = (width, height)
#     frame_list.append(img)
#
#
#
#
#
#
#     key = cv2.waitKey(30)
#
#     if key==27:
#         break
# print("hhh")
# # out = cv2.VideoWriter(pathOut, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
# # for i in range(len(frame_list)):
# #         # writing to a image array
# #         out.write(frame_list[i])
#
#
#
#
# # sheet1.write(3, 0, current_count)
# # wb.save('xlwt example.xls')
# # out.release()
# cap.release()
# cv2.destroyAllWindows()

