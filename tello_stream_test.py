from djitellopy import Tello
import os
import cv2
import time
import numpy as np
import sys
from pathlib import Path

# 增加yolov7到Python路径
yolov7_path = Path("yolov7")
sys.path.append(str(yolov7_path))
from yolov7.detector import YoloDetector

# 设置FFmpeg参数以改善视频流解码
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"

def main():
    # 初始化 Tello
    print("[STATUS] 正在连接Tello无人机...")
    tello = Tello()

    try:
        tello.connect()
        print(f"[STATUS] 连接成功! 电池电量: {tello.get_battery()}%")

        # 启动视频流
        print("[VIDEO] 启动视频流")
        tello.streamon()
        time.sleep(2)  # 给视频流启动时间
        frame_reader = tello.get_frame_read()

        # 初始化 YOLO 检测器
        print("[STATUS] 正在加载YOLOv7模型...")
        detector = YoloDetector("yolov7/yolov7-tiny.pt")
        print("[STATUS] 模型加载完成")

        # 设置视频写入器
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        video_filename = f"tello_human_detection_{timestamp}.avi"
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(video_filename, fourcc, 20.0, (960, 720))  # 使用更高分辨率
        print(f"[STATUS] 视频录制已启动: {video_filename}")

        last_print_time = time.time()
        frame_count = 0
        print("[STATUS] 开始人体检测循环 (按 'q' 键退出)")

        while True:
            start_time = time.time()
            frame_count += 1

            # 获取帧
            frame = frame_reader.frame
            if frame is None:
                print("[WARNING] 接收到空帧，跳过处理")
                time.sleep(0.1)
                continue
                
            # 调整帧大小
            frame = cv2.resize(frame, (960, 720))
            
            # 检测人体
            results = detector.detect(frame)
            person_count = 0

            for res in results:
                if res["label"] == "person":
                    person_count += 1
                    x1, y1, x2, y2 = res["box"]
                    # 绘制人体框和标签
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"Person {person_count}", (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            # 显示状态信息
            fps = 1.0 / (time.time() - start_time + 1e-9)
            cv2.putText(frame, f"Persons: {person_count}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
            cv2.putText(frame, f"Battery: {tello.get_battery()}%", (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

            # 写入视频帧
            out.write(frame)
            
            
            # 定期打印状态
            current_time = time.time()
            if current_time - last_print_time > 2.0:
                print(f"[STATUS] 处理中... 检测到人数: {person_count} | FPS: {fps:.1f}")
                last_print_time = current_time

            # 退出检测
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"[ERROR] 发生异常: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        print("[STATUS] 清理资源...")
        # 关闭视频流
        tello.streamoff()
        cv2.destroyAllWindows()
        
        # 释放视频写入器
        if 'out' in locals():
            out.release()
            print(f"[SUCCESS] 视频保存成功: {video_filename}")
            
        print(f"[STATUS] 共处理帧数: {frame_count}")
        print("[STATUS] 程序安全退出")

if __name__ == "__main__":
    main()
