import cv2
import numpy as np

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('test_output.mp4', fourcc, 20.0, (640, 480))

for i in range(100):  # 5秒的视频
    frame = np.full((480, 640, 3), (i * 2) % 255, dtype=np.uint8)
    cv2.putText(frame, f"Frame {i}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    out.write(frame)

out.release()
print("✅ test_output.mp4 写入完成")