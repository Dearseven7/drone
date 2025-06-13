from djitellopy import Tello
import os
from pathlib import Path
import sys
import cv2
import time
import numpy as np
import sys
# 增加yolov7到Python路径
yolov7_path = Path("yolov7")
sys.path.append(str(yolov7_path))
from yolov7.detector import YoloDetector

# 设置FFmpeg参数以改善视频流解码
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"

class PIDController:
    def __init__(self, Kp, Ki, Kd, max_output=100, setpoint=0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.max_output = max_output
        self.setpoint = setpoint
        self.prev_error = 0
        self.integral = 0
        self.last_time = time.time()

    def compute(self, measurement):
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time
        
        error = self.setpoint - measurement
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0

        # 积分限幅防止windup
        self.integral = max(-self.max_output, min(self.max_output, self.integral))

        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        output = max(-self.max_output, min(self.max_output, output))
        self.prev_error = error
        return output

def main():
    # 初始化 Tello
    print("[STATUS] 正在连接Tello的无人机...")
    tello = Tello()

    try:
        tello.connect()
        print(f"[STATUS] 连接成功! 电池电量: {tello.get_battery()}%")
        tello.TIME_BTW_RC_CONTROL_COMMANDS = 0.01
        
        # 启动视频流
        print("[VIDEO] 启动视频流")
        tello.streamon()
        time.sleep(1)  # 给视频流启动时间
        frame_reader = tello.get_frame_read()

        # 初始化 YOLO 检测器
        print("[STATUS] 正在加载YOLOv7模型...")
        detector = YoloDetector("yolov7/yolov7-tiny.pt")
        print("[STATUS] 模型加载完成")

        # 设置视频写入器
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        video_filename = f"tello_output_{timestamp}.avi"
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # 使用MJPG编码器
        out = cv2.VideoWriter(video_filename, fourcc, 20.0, (640, 480))
        print(f"[STATUS] 视频录制已启动: {video_filename}")

        frame_count = 0

        # PID控制器初始化
        pid_yaw = PIDController(Kp=0.2, Ki=0.01, Kd=0.001, max_output=50, setpoint=320)  # 偏航控制
        pid_ud = PIDController(Kp=0.2, Ki=0.01, Kd=0.001, max_output=50, setpoint=240)   # 上下控制
        pid_fb = PIDController(Kp=0.002, Ki=0.0, Kd=0.0002, max_output=50, setpoint=50000)  # 前后控制
        
        last_time = time.time()
        last_print_time = time.time()
        print("[STATUS] 开始人体检测循环 (按 'q' 键退出)")

        print("[STATUS] 所有初始化完成，准备起飞")
        input("[INPUT] 按 Enter 键起飞...")

        tello.takeoff()
        print("[STATUS] 起飞指令已发送")

        # 控制变量
        yaw_speed = 0  # 旋转速度
        ud_speed = 0   # 上下速度
        fb_speed = 0   # 前后速度

        while True:
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time
            frame_count += 1

            # 获取帧
            frame = frame_reader.frame
            if frame is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frame = cv2.resize(frame, (640, 480))
            else:
                print("[WARNING] 接收到空帧，跳过处理")
                time.sleep(0.1)
                continue

            # 检测目标
            results = detector.detect(frame)
            selected_person = None
            max_area = 0
            person_count = 0

            for res in results:
                if res["label"] == "person":
                    person_count += 1
                    x1, y1, x2, y2 = res["box"]
                    area = (x2 - x1) * (y2 - y1)
                    if area > max_area:
                        max_area = area
                        selected_person = res

            # 每2秒打印一次状态
            if current_time - last_print_time > 2.0:
                print(f"[STATUS] 帧率: {frame_count/(current_time-last_print_time):.1f}fps | 检测到人数: {person_count}")
                last_print_time = current_time
                frame_count = 0

            # 控制逻辑
            if selected_person:
                x1, y1, x2, y2 = selected_person["box"]
                current_x = (x1 + x2) // 2
                current_y = (y1 + y2) // 2
                current_area = (x2 - x1) * (y2 - y1)
                targety=y2 - (y2 - y1) * 0.575
                # 计算控制量
                yaw_speed = -int(pid_yaw.compute(current_x))  # 偏航控制
                ud_speed = int(pid_ud.compute(targety))   # 上下控制（注意方向）
                fb_speed = int(pid_fb.compute(current_area))         # 前后控制

                # 发送控制指令
                tello.send_rc_control(
                    0,         # 左右速度（禁用）
                    fb_speed,  # 前后速度
                    ud_speed,  # 上下速度
                    yaw_speed  # 旋转速度
                )

                # 可视化
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, (current_x, current_y), 5, (0, 255, 0), -1)
                cv2.putText(frame, f"YAW: {yaw_speed} UD: {ud_speed} FB: {fb_speed}",
                           (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            else:
                # 丢失目标时停止运动
                tello.send_rc_control(0, 0, 0, 0)
                pid_yaw.integral = 0
                pid_ud.integral = 0
                pid_fb.integral = 0
                if person_count == 0:
                    cv2.putText(frame, "NO PERSON DETECTED", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

            # 绘制中心线和状态信息
            cv2.circle(frame, (320, 240), 5, (0, 0, 255), -1)
            cv2.line(frame, (0, 240), (640, 240), (255, 0, 0), 1)  # 水平中线
            cv2.line(frame, (320, 0), (320, 480), (255, 0, 0), 1)  # 垂直中线
            cv2.putText(frame, f"YAW: {yaw_speed} UD: {ud_speed} FB: {fb_speed}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
            cv2.putText(frame, f"Battery: {tello.get_battery()}% | FPS: {1/dt:.1f}",
                       (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,100), 1)

            out.write(frame)

    except KeyboardInterrupt:
        print("\n[STATUS] 检测到键盘中断，正在清理...")
    except Exception as e:
        print(f"[ERROR] 发生异常: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        print("[STATUS] 清理资源...")
        try:
            # 发送停止指令
            tello.send_rc_control(0, 0, 0, 0)
            time.sleep(0.5)

            # 尝试降落
            print("[STATUS] 尝试降落无人机...")
            for _ in range(3):
                try:
                    tello.land()
                    print("[STATUS] 降落指令已发送")
                    time.sleep(1)
                    break
                except Exception as e:
                    print(f"[WARNING] 降落失败: {str(e)}")
                    time.sleep(1)

            # 关闭视频流
            tello.streamoff()

            # 释放视频写入器
            if 'out' in locals() and out is not None:
                out.release()

        except Exception as e:
            print(f"[ERROR] 清理过程中发生错误: {e}")

        print(f"[STATUS] 共处理帧数: {frame_count}")

        if 'video_filename' in locals():
            print(f"[SUCCESS] 视频保存成功: {video_filename}")

        print("[STATUS] 程序安全退出")
        sys.exit(0)
if __name__ == "__main__":
    main()
