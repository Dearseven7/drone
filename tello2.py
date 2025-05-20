from djitellopy import Tello
import cv2

# 初始化 Tello
tello = Tello()
tello.connect()
print(f"[INFO] 电池电量: {tello.get_battery()}%")

# 开启视频流
tello.streamon()
frame_reader = tello.get_frame_read()

print("[INFO] 正在显示 Tello 摄像头画面，按 'q' 退出")
while True:
    frame = frame_reader.frame
    frame = cv2.resize(frame, (640, 480))
    
    cv2.imshow("Tello Live Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 清理资源
tello.streamoff()
cv2.destroyAllWindows()