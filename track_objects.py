import os, sys
import cv2, time as t, numpy as np

tracker_types = ['MIL', 'KCF', 'CSRT', 'BOOSTING', 'TLD', 'GOTURN']


class Tracker_base():
    def init(self) -> bool: # ok/not ok
        self.a = 0
        self.b = []
        ... # other needed vars

    #@classmethod
    def update(self, frame: np.ndarray) -> tuple[bool, tuple]:
        assert isinstance(frame, np.ndarray), 'frame should be of type np.ndarray'
        # self.bbox = self.prev ... frame # using some formula with prev and curr frames and do update
        # self.prev = self.bbox
        #returns ok, self.bbox


# http://grabner.family/helmut/papers/Grabner2006On-lineBoostingand.pdf
class TrackerBoosting_create(Tracker_base): 
    def __init__(self, *values):
        # values - some values in order to do tracking
        super().__init__(values) 
        self.a = ...
        ...

## and then in while loop down below call tracker(*[x,y,v,...])
# http://vision.stanford.edu/teaching/cs231b_spring1415/papers/Kalal-PAMI.pdf
# https://www.cs.vsu.ru/ipmt-conf/conf/2021/works/3.%20%D0%A2%D0%B5%D1%85%D0%BD%D0%BE%D0%BB%D0%BE%D0%B3%D0%B8%D0%B8%20%D0%BE%D0%B1%D1%80%D0%B0%D0%B1%D0%BE%D1%82%D0%BA%D0%B8%20%D0%B8%20%D0%B7%D0%B0%D1%89%D0%B8%D1%82%D1%8B%20%D0%B8%D0%BD%D1%84%D0%BE%D1%80%D0%BC%D0%B0%D1%86%D0%B8%D0%B8/1778.dokl.pdf
class TrackerTLD_create(): 
    def __init__(self):
        super().__init__()

tracker_constructors = {
        1: cv2.TrackerMIL_create,
        2: cv2.TrackerKCF_create,
        3: cv2.TrackerCSRT_create,
        4: TrackerBoosting_create,
        5: TrackerTLD_create,
        }    


cv2.ocl.setUseOpenCL(False)
def track_update_loop(frame_queue, result_queue, tracker_type):
    tracker = tracker_constructors[tracker_type]()
    initialized = False
    while True:
        item = frame_queue.get()
        if item is None: break
        frame, bbox = item
        if bbox is not None:
            tracker = tracker_constructors[tracker_type]()
            tracker.init(frame, bbox)
            initialized = True
            result_queue.put((True, bbox))  # send initial bbox
            continue
        if not initialized:
            continue
        ok, bbox = tracker.update(frame)
        result_queue.put((ok, bbox))


if __name__ == '__main__':    
    from threading import Thread
    from queue import Queue

    frame_queue = Queue(maxsize=1)
    result_queue = Queue(maxsize=1)
    tracker_type, do_mask, video_stream_path = int(sys.argv[1]), bool(int(sys.argv[2])), sys.argv[3]
    print(cv2.getBuildInformation())
    if tracker_type not in [*range(1, len(tracker_types) + 1)]:
        raise ValueError('tracker type should be number 1-7')
    
    if not os.path.isfile(video_stream_path): video_stream_path = int(video_stream_path)
    else: video_stream_path = os.path.abspath(video_stream_path) 
    video = cv2.VideoCapture(video_stream_path)

    updating_process = Thread(
        target=track_update_loop,
        args=(frame_queue, result_queue, tracker_type),
        daemon=True
    )
    updating_process.start()
    if not video.isOpened():
        print("Couldn't open video")
        sys.exit(0)
    ok, frame = video.read()
    if not ok:
        print('Cannot read video file')
        sys.exit(0)
    bbox = cv2.selectROI(frame, False) 
    frame_queue.put((frame.copy(), bbox))

    path, fps = [], 0
    process_time = 0.0
    choose_roi_mode = False
    frame_frozen = None
    last_ok, last_bbox = True, bbox  # do caching of the last known tracking state
    while True:
        st = t.time()
        ok, frame = video.read()
        if not ok:
            frame_queue.put(None)
            break
        key_pressed = cv2.waitKey(30) & 0xFF
        if key_pressed == ord('q'):
            frame_queue.put(None)
            sys.exit(0)
        elif key_pressed == ord('r') and not choose_roi_mode:     
            choose_roi_mode = True
            frame_frozen = frame.copy()
            bbox = cv2.selectROI(frame_frozen, False) 
            frame_queue.put((frame_frozen.copy(), bbox))
            path = []
            choose_roi_mode = False
            process_time = 0
            last_ok, last_bbox = True, bbox
        else:
            if frame_queue.empty():
                frame_queue.put((frame.copy(), None))
            if not result_queue.empty():
                last_ok, last_bbox = result_queue.get()
                process_time = t.time()-st
                fps = 1.0 / process_time if process_time > 0 else fps

            if last_ok:
                x, y, w, h = map(int, last_bbox)
                if not do_mask:
                    cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
                else:
                    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                    cv2.rectangle(mask, (x,y), (x+w,y+h),  255, -1)
                    cv2.bitwise_and(frame, frame, mask)
                    frame_black = np.zeros_like(frame)
                    frame_black[mask == 255] = frame[mask == 255]
                    frame = frame_black
                path.append((x+w//2, y+h//2))
                for center in path:
                    cv2.circle(frame, center=center, radius=5, color=(0,0,255), thickness=2)
            else:
                cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)

            cv2.putText(frame, tracker_types[tracker_type-1] + " Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2)
            cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)
            cv2.putText(frame, "frame processing time " + str(f"{process_time:.2f}"), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)
            cv2.imshow("Tracking", frame)

    updating_process.join()
    cv2.destroyAllWindows()