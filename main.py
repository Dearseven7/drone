from djitellopy import Tello

import os
from pathlib import Path

#增加yolov7到python路径
yolov7_path = Path("yolov7")  # 替换为你的实际路径
import sys
sys.path.append(str(yolov7_path))

import cv2
import time
from yolo.detector import YoloDetector

# 初始化 Tello
tello = Tello()
tello.connect()
print(f"[INFO] 电池电量: {tello.get_battery()}%")
tello.streamon()

# 初始化 YOLO 检测器
detector = YoloDetector("yolov7/yolov7-tiny.pt")

# 设置视频写入器（MP4 格式）
timestamp = time.strftime("%Y%m%d_%H%M%S")
video_filename = f"tello_output_{timestamp}.avi"
fourcc = cv2.VideoWriter_fourcc(*'XVID')
# video_filename = f"tello_output_{timestamp}.mp4"
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # mp4 编码器
out = cv2.VideoWriter(video_filename, fourcc, 20.0, (640, 480))
print(f"[INFO] 视频将保存为: {video_filename}")

frame_reader = tello.get_frame_read()
frame_count = 0

print("[INFO] 正在检测人体，按 'q' 键退出")
try:
    while True:
        frame = frame_reader.frame

        # if frame is None or frame.shape[0] == 0:
        #     print("⚠️ 跳过空帧")
        #     continue

        # 确保尺寸匹配
        frame = cv2.resize(frame, (640, 480))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # 检测
        results = detector.detect(frame)
        for res in results:
            if res["label"] != "person":
                continue
            x1, y1, x2, y2 = res["box"]
            conf = res["conf"]
            label = f'{res["label"]} {conf:.2f}'
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # 显示

        # 写入视频
        out.write(frame)
        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    tello.streamoff()
    out.release()
    cv2.destroyAllWindows()
    print(f"[INFO] 共写入帧数: {frame_count}")
    print(f"[✅] 视频保存成功: {video_filename}")
