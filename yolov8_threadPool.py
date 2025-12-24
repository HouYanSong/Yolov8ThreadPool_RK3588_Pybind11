import cv2
import time
import threading
from build import yolov8_threadPool_pybind11 


class YoloV8ThreadPool:
    def __init__(self, model_file, input_w, input_h, class_num, mapSize, num_threads, video_url):
        self.end = False
        self.video_url = video_url
        self.read_frame_id = 0
        self.get_result_id = 0
        self.yolov8_threadPool = yolov8_threadPool_pybind11.Yolov8ThreadPool(model_file, input_w, input_h, class_num, mapSize, num_threads)
        self.read_frame_thread = threading.Thread(target=self.read_frame, daemon=True)
        self.get_result_thread = threading.Thread(target=self.get_result, daemon=True)
        self.read_frame_thread.start()
        self.get_result_thread.start()
        self.read_frame_thread.join()
        self.get_result_thread.join()
        

    def read_frame(self):
        cap = cv2.VideoCapture(self.video_url)
        while cap.isOpened() and not self.end:
            ret, frame = cap.read()
            if not ret:
                break
            if self.read_frame_id - self.get_result_id < 10:
                self.yolov8_threadPool.submitImg(frame, 0.25, 0.45, "opencv")
                self.read_frame_id += 1
            else:
                time.sleep(0.01)
        self.end = True
        cap.release()
            

    def get_result(self):
        FPS = None
        frame_count = 0
        start_time = time.time()     
        while not self.end:
            if self.get_result_id < self.read_frame_id:
                self.get_result_id += 1
                frame, detections = self.yolov8_threadPool.getResult()
                for det in detections:
                    cv2.rectangle(frame, (det.box.x, det.box.y), (det.box.x + det.box.width, det.box.y + det.box.height), (0, 255, 0), 2)
                    cv2.putText(frame, f"{det.class_id} {det.confidence:.2f}", (det.box.x, det.box.y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                frame_count += 1
                if frame_count > 100:
                    fps = frame_count / (time.time() - start_time)
                    FPS = f"{fps:.2f} FPS"
                    print(FPS)
                    frame_count = 0
                    start_time = time.time()
                if FPS:
                    cv2.putText(frame, FPS, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)  
                # cv2.imshow("YOLOv8 Thread Pool", frame)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break
            else:
                time.sleep(0.01)
        self.end = True
        

if __name__ == "__main__":
    model_file = "./weights/yolov8s.int.rknn"
    input_w = 640
    input_h = 640
    class_num = 80
    mapSize = [[80, 80], [40, 40], [20, 20]]
    num_threads = 9
    video_url = "./media/bj_short.mp4"
    yolov8_threadPool = YoloV8ThreadPool(model_file, input_w, input_h, class_num, mapSize, num_threads, video_url)