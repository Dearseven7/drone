from djitellopy import Tello
import os
import cv2
import time
import numpy as np
import mediapipe as mp

# 设置FFmpeg参数以改善视频流解码（可选）
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"

# MediaPipe Pose 初始化
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False,
                    model_complexity=1,
                    enable_segmentation=False,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils


def main():
    # 初始化 Tello
    print("[STATUS] 正在连接Tello无人机...")
    tello = Tello()

    try:
        # 连接无人机
        tello.connect()
        print(f"[STATUS] 连接成功! 电池电量: {tello.get_battery()}%")

        # 启动视频流
        print("[VIDEO] 启动视频流")
        tello.streamon()
        time.sleep(1)
        frame_reader = tello.get_frame_read()

        # 视频写入器
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        video_filename = f"tello_blazepose_{timestamp}.avi"
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(video_filename, fourcc, 20.0, (640, 480))
        print(f"[STATUS] 视频录制已启动: {video_filename}")

        last_print = time.time()
        frame_count = 0
        print("[STATUS] 开始Pose检测循环 (按 'q' 键退出)")

        while True:
            start = time.time()
            frame_count += 1

            # 获取并预处理帧
            frame = frame_reader.frame
            if frame is None:
                time.sleep(0.01)
                continue
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img.flags.writeable = False
            img = cv2.resize(img, (640, 480))

            # MediaPipe Pose 处理
            results = pose.process(img)

            # 绘制检测结果
            img.flags.writeable = True
            vis = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            if results.pose_landmarks:
                # 绘制关键点骨架
                mp_drawing.draw_landmarks(
                    vis, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # 状态信息
            fps = 1.0 / (time.time() - start + 1e-9)
            cv2.putText(vis, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.putText(vis, f"Battery: {tello.get_battery()}%", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 1)

            # 显示并写入
            cv2.imshow("Tello BlazePose", vis)
            out.write(vis)

            # 打印日志
            if time.time() - last_print > 2.0:
                print(f"[STATUS] 处理中... 帧数: {frame_count} | FPS: {fps:.1f}")
                last_print = time.time()

            # 按 q 退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"[ERROR] 发生异常: {e}")
    finally:
        print("[STATUS] 清理资源...")
        tello.streamoff()
        out.release()
        cv2.destroyAllWindows()
        print(f"[SUCCESS] 视频保存成功: {video_filename}")
        print(f"[STATUS] 共处理帧数: {frame_count}")
        print("[STATUS] 程序安全退出")


if __name__ == "__main__":
    main()

