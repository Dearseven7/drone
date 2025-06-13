from djitellopy import Tello
import os
from pathlib import Path
import sys
import cv2
import time
# 增加yolov7到Python路径
yolov7_path = Path("yolov7")
sys.path.append(str(yolov7_path))
from yolov7.detector import YoloDetector

class PIDController:
    def __init__(self, Kp, Ki, Kd, max_output=100):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.max_output = max_output
        self.prev_error = 0
        self.integral = 0

    def compute(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0
        
        # 积分限幅防止windup
        self.integral = max(-self.max_output, min(self.max_output, self.integral))
        
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        output = max(-self.max_output, min(self.max_output, output))
        self.prev_error = error
        return int(output)

# 初始化 Tello
tello = Tello()
tello.connect()
print(f"[INFO] 电池电量: {tello.get_battery()}%")
tello.streamon()
tello.takeoff()  # 无人机起飞

# 初始化 YOLO 检测器
detector = YoloDetector("yolov7/yolov7-tiny.pt")

# 设置视频写入器
timestamp = time.strftime("%Y%m%d_%H%M%S")
video_filename = f"tello_output_{timestamp}.avi"
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(video_filename, fourcc, 20.0, (640, 480))

frame_reader = tello.get_frame_read()
frame_count = 0

# PID参数（需要实际调试）
pid_yaw = PIDController(Kp=0.5, Ki=0.01, Kd=0.2, max_output=50)    # 偏航控制
pid_throttle = PIDController(Kp=0.4, Ki=0.01, Kd=0.1, max_output=50) # 高度控制

last_time = time.time()
print("[INFO] 正在检测人体，按 'q' 键退出")

try:
    yaw_speed = 0
    throttle_speed = 0
    while True:
        current_time = time.time()
        dt = current_time - last_time
        last_time = current_time

        frame = frame_reader.frame
        frame = cv2.resize(frame, (640, 480))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # 检测目标
        results = detector.detect(frame)
        selected_person = None
        max_area = 0

        for res in results:
            if res["label"] == "person":
                x1, y1, x2, y2 = res["box"]
                area = (x2 - x1) * (y2 - y1)
                if area > max_area:
                    max_area = area
                    selected_person = res

        # 控制逻辑
        if selected_person:
            x1, y1, x2, y2 = selected_person["box"]
            current_x = (x1 + x2) // 2
            current_y = (y1 + y2) // 2

            # 计算误差（中心点320x240）
            error_yaw = 320 - current_x   # 水平方向误差
            error_throttle = 240 - current_y  # 垂直方向误差

            # 计算控制量
            yaw_speed = pid_yaw.compute(error_yaw, dt)
            throttle_speed = pid_throttle.compute(error_throttle, dt)

            """
            控制方向说明：
            - 当目标偏左（error_yaw>0），应左转（负速度）
            - 当目标偏右（error_yaw<0），应右转（正速度）
            - 当目标偏下（error_throttle>0），应下降（负速度）
            - 当目标偏上（error_throttle<0），应上升（正速度）
            """
            # 发送控制指令（左右/前后速度为0，只控制旋转和高度）
            tello.send_rc_control(
                0,  # 左右速度
                0,  # 前后速度 
                -throttle_speed,  # 上下速度（注意符号方向）
                -yaw_speed       # 旋转速度（注意符号方向）
            )

            # 可视化
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (current_x, current_y), 5, (0, 255, 0), -1)
        else:
            # 丢失目标时停止运动
            tello.send_rc_control(0, 0, 0, 0)
            pid_yaw.integral = 0
            pid_throttle.integral = 0

        # 绘制中心线和状态信息
        cv2.circle(frame, (320, 240), 5, (0, 0, 255), -1)
        cv2.putText(frame, f"YAW: {yaw_speed} THR: {throttle_speed}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
        
        out.write(frame)
        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    tello.send_rc_control(0, 0, 0, 0)
    tello.land()
    tello.streamoff()
    out.release()
    cv2.destroyAllWindows()
    print(f"[INFO] 共写入帧数: {frame_count}")
    print(f"[✅] 视频保存成功: {video_filename}")
