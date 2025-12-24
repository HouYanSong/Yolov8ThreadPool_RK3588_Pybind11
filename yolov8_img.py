import cv2
from build import yolov8_pybind11

yolov5 = yolov8_pybind11.Yolov8Detector("./weights/yolov8s.int.rknn", 640, 640, 80, [[80, 80], [40, 40], [20, 20]])

img = cv2.imread("./media/street.jpg")

detections = yolov5.detect(img.copy(), 0.45, 0.55, "opencv")

for i, det in enumerate(detections):
    cv2.rectangle(img, (det.box.x, det.box.y), (det.box.x + det.box.width, det.box.y + det.box.height), (0, 255, 0), 2)
    cv2.putText(img, f"{det.class_id} {det.confidence:.2f}", (det.box.x, det.box.y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

cv2.imwrite("result.jpg", img)