import cv2
import cvzone
from yolo_segmentation import YOLOSEG
from time import sleep

vid = cv2.VideoCapture(0)
vid.set(3, 640) # width
vid.set(4, 640) # heigth
model = YOLOSEG("runs/segment/train-small/weights/best.pt")
labels = open("labels.txt", "r")
label = labels.read()
class_list = label.split("\n")



# def RGB(event, x, y, flags, param):
#     if event == cv2.EVENT_MOUSEMOVE :
#         point = [x, y]
#         print(point)

# cv2.namedWindow('RGB')
# cv2.setMouseCallback('RGB', RGB)


while True:
    if not vid.isOpened():
        print("sorry, kamera kamu tidak terdeteksi")
        sleep(5)

    ret, frame = vid.read()
    frame = cv2.resize(frame, (640, 640))
    overlay = frame.copy()
    alpha = 0.8

    bboxes, classes, segmentation, score = model.detect(frame)
    for bbox, class_id, seg, score in zip(bboxes, classes, segmentation, score):
        (x, y, x2, y2) = bbox
        classes = class_list[class_id]
        conf = score * 100
        
        if conf > 70:
            cv2.rectangle(frame, (x, y), (x2, y2), (255, 0, 0), 2)
            cv2.polylines(frame, [seg], True, (0, 0, 255), 4)
            cv2.fillPoly(overlay, [seg], (0, 0, 255))
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 2, frame)
            cvzone.putTextRect(frame, f'{classes} {conf:.1f}%', (x, y), 1, 1)


    cv2.imshow("frane", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

vid.release()
cv2.destroyAllWindows()


