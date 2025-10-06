#!/usr/bin/env python3
"""
screen_person_detection.py
"""

import os
import sys
# 添加当前文件所在目录到系统路径，解决无法索引到其他关联文件的问题
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
from ultralytics import YOLO
import time
import torch
import pyautogui
import mss
import tkinter as tk
from PIL import Image, ImageTk
import threading
import gc
import signal

class ScreenPersonDetector:
    def __init__(self, model_path='yolov8x.pt', confidence_threshold=0.3, verbose=False):
        # 设置是否输出详细信息
        self.verbose = verbose
        # 加载YOLOv8模型，使用GPU进行推理
        # 确保使用本地已有的模型文件
        if os.path.exists(model_path):
            if self.verbose:
                print(f"使用本地模型文件: {model_path}")
            self.model = YOLO(model_path, task='detect')
        else:
            raise FileNotFoundError(f"模型文件 {model_path} 不存在")
        self.confidence_threshold = confidence_threshold
        # 获取屏幕尺寸
        self.screen_width, self.screen_height = pyautogui.size()
        # 控制标志
        self.running = False
        # 创建透明覆盖窗口
        self.overlay_window = None
        # 性能控制
        self.frame_skip = 1  # 不跳过帧以提高检测效果
        self.frame_count = 0
        # 间隔输出控制
        self.last_output_time = time.time()
        self.output_interval = 5  # 每5秒输出一次状态

    def create_overlay_window(self):
        """创建透明覆盖窗口"""
        # 如果窗口已存在，先销毁它
        if self.overlay_window is not None:
            try:
                self.overlay_window.destroy()
            except tk.TclError:
                pass  # 窗口可能已经关闭
            
        try:
            # 创建新的透明窗口
            self.overlay_window = tk.Tk()
            self.overlay_window.title("Person Detection Overlay")
            self.overlay_window.geometry(f"{self.screen_width}x{self.screen_height}+0+0")
            self.overlay_window.overrideredirect(True)  # 无边框窗口
            self.overlay_window.attributes("-topmost", True)  # 窗口置顶
            self.overlay_window.attributes("-transparentcolor", "white")  # 设置透明颜色
            self.overlay_window.lift()
            
            # 创建Canvas用于绘制
            self.canvas = tk.Canvas(self.overlay_window, width=self.screen_width, height=self.screen_height, 
                                   highlightthickness=0, bg='white')
            self.canvas.pack(fill=tk.BOTH, expand=True)
            
            # 绑定关闭事件
            self.overlay_window.protocol("WM_DELETE_WINDOW", self.stop_detection)
            
            # 绑定Esc键退出
            self.overlay_window.bind("<Escape>", lambda e: self.stop_detection())
        except Exception as e:
            if self.verbose:
                print(f"创建覆盖窗口时出现错误: {e}")
            self.overlay_window = None

    def update_overlay(self, detections):
        """更新覆盖窗口上的检测框"""
        if self.overlay_window is None:
            return
            
        try:
            # 清除之前的绘制内容
            self.canvas.delete("all")
            
            # 绘制每个检测框
            for detection in detections:
                x1, y1, x2, y2, confidence = detection[:5]  # 只取前5个值（不需要距离）
                
                # 绘制边界框（增加宽度以提高可见性）
                self.canvas.create_rectangle(x1, y1, x2, y2, outline='red', width=3)
                
                # 绘制置信度（改善字体清晰度和大小）
                label = f"Person: {confidence:.2f}"
                # 添加背景以提高文本可读性
                self.canvas.create_rectangle(x1-2, y1-25, x1+120, y1-5, fill='white', outline='red', width=1)
                self.canvas.create_text(x1, y1 - 10, text=label, fill='red', anchor='sw', 
                                      font=('Arial', 14, 'bold'))
                
            # 更新窗口
            self.overlay_window.update()
        except tk.TclError:
            # 窗口可能已被销毁
            self.overlay_window = None

    def process_screen(self):
        """处理屏幕捕获和检测"""
        try:
            with mss.mss() as sct:
                # 获取主屏幕信息
                monitor = sct.monitors[1]  # 通常monitor[1]是主屏幕
                
                # 初始化输出时间
                self.last_output_time = time.time()
                
                # 输出初始信息
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                if self.verbose:
                    print(f"开始屏幕监控: {self.screen_width}x{self.screen_height}")
                    print(f"使用设备: {device}")
                    print(f"每{self.output_interval}秒输出一次状态信息")
                
                while self.running:
                    try:
                        # 跳过帧以提高性能
                        self.frame_count += 1
                        if self.frame_count % self.frame_skip != 0:
                            time.sleep(0.01)  # 短暂休眠以减少CPU使用
                            continue
                        
                        # 捕获屏幕
                        screenshot = sct.grab(monitor)
                        # 转换为numpy数组
                        frame = np.array(screenshot)
                        # 转换为BGR格式（OpenCV）
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                        
                        # 使用YOLOv8进行检测
                        results = self.model(frame, conf=self.confidence_threshold, classes=[0], 
                                           device=device)
                        
                        # 处理检测结果
                        detections = []
                        for result in results:
                            boxes = result.boxes
                            for box in boxes:
                                # 获取边界框坐标
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                # 获取置信度
                                confidence = box.conf[0]
                                
                                # 添加到检测列表（不包含距离）
                                detections.append((x1, y1, x2, y2, confidence))
                        
                        # 更新覆盖窗口
                        self.update_overlay(detections)
                        
                        # 间隔输出状态信息
                        current_time = time.time()
                        if self.verbose and current_time - self.last_output_time >= self.output_interval:
                            print(f"[状态更新] 已处理 {self.frame_count} 帧, 检测到 {len(detections)} 个人物")
                            self.last_output_time = current_time
                        
                        # 内存管理
                        if self.frame_count % 30 == 0:
                            gc.collect()
                            
                    except Exception as e:
                        # 仅输出关键错误信息，避免过多输出
                        current_time = time.time()
                        if self.verbose and current_time - self.last_output_time >= 1:  # 避免连续输出相同错误
                            print(f"处理错误: {str(e)[:100]}...")  # 限制错误信息长度
                            self.last_output_time = current_time
                        time.sleep(0.1)
        except Exception as e:
            if self.verbose:
                print(f"屏幕捕获错误: {e}")
                     
    def start_detection(self):
        """开始检测"""
        try:
            self.running = True
            # 创建覆盖窗口
            self.create_overlay_window()
            # 在新线程中处理屏幕捕获和检测
            self.process_thread = threading.Thread(target=self.process_screen)
            self.process_thread.daemon = True  # 设置为守护线程，主线程结束时自动终止
            self.process_thread.start()
            
            # 启动Tkinter主循环
            if self.overlay_window:
                self.overlay_window.mainloop()
        except Exception as e:
            if self.verbose:
                print(f"启动检测错误: {e}")
            self.stop_detection()

    def stop_detection(self):
        """停止检测"""
        self.running = False
        if hasattr(self, 'process_thread') and self.process_thread.is_alive():
            self.process_thread.join(timeout=1.0)  # 等待线程结束，最多等待1秒
        if self.overlay_window:
            try:
                # 确保在主线程中销毁窗口
                if threading.current_thread() == threading.main_thread():
                    self.overlay_window.quit()
                    self.overlay_window.destroy()
            except tk.TclError:
                pass  # 窗口可能已经关闭
            self.overlay_window = None
        if self.verbose:
            print("检测已停止")

# 全局信号处理函数，解决无法退出终端的问题
def signal_handler(sig, frame):
    print("\n收到退出信号，正在停止检测...")
    if 'detector' in globals():
        detector.stop_detection()
    sys.exit(0)

if __name__ == "__main__":
    try:
        # 设置信号处理，捕获Ctrl+C等退出信号
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # 创建ScreenPersonDetector实例，设置verbose=False以减少输出
        detector = ScreenPersonDetector(verbose=False)
        
        try:
            # 开始检测
            print("按 Ctrl+C 或 Esc 键停止检测")
            detector.start_detection()
        except KeyboardInterrupt:
            detector.stop_detection()
            print("程序已退出")
    except Exception as e:
        print(f"程序启动失败: {e}")
        print("请确保已正确安装所有依赖项并配置好环境")