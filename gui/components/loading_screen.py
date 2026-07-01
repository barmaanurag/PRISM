from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QProgressBar, QHBoxLayout
from PyQt6.QtCore import pyqtSignal, Qt, QProcess, QTimer, QTime
from PyQt6.QtGui import QFont
import sys
import os
import re

class LoadingScreen(QWidget):
    processing_finished = pyqtSignal(str)
    processing_error = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.status_label = QLabel("Initializing...")
        self.status_label.setStyleSheet("font-size: 24px; font-weight: bold; margin-bottom: 10px; color: #FF8C00;")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Live timer
        self.time_label = QLabel("00:00")
        self.time_label.setStyleSheet("font-size: 36px; font-weight: bold; margin-bottom: 20px; color: #333;")
        self.time_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.log_label = QLabel("Starting models...")
        self.log_label.setStyleSheet("font-size: 14px; margin-bottom: 20px; color: #666;")
        self.log_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.log_label.setWordWrap(True)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setFixedSize(400, 20)
        
        # Stats layout
        self.stats_label = QLabel("")
        self.stats_label.setStyleSheet("font-size: 14px; font-family: monospace; color: #444; background: white; padding: 10px; border-radius: 8px;")
        self.stats_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.stats_label.hide()
        
        layout.addWidget(self.status_label)
        layout.addWidget(self.time_label)
        layout.addWidget(self.progress_bar, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.log_label)
        layout.addWidget(self.stats_label, alignment=Qt.AlignmentFlag.AlignCenter)
        
        self.setLayout(layout)
        
        self.process = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_time)
        self.elapsed_time = 0
        self.stats_text = []

    def start_processing(self, video_path):
        self.status_label.setText("Processing Video")
        self.log_label.setText("Running Pipeline...")
        self.stats_text = []
        self.stats_label.setText("")
        self.stats_label.hide()
        
        self.elapsed_time = 0
        self.time_label.setText("00:00")
        self.timer.start(1000)
        
        self.process = QProcess()
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.process.readyReadStandardOutput.connect(self.handle_stdout)
        self.process.finished.connect(self.handle_finished)
        
        script_path = os.path.join(os.path.dirname(__file__), '..', 'run_inference.py')
        python_exe = sys.executable
        
        self.process.start(python_exe, [script_path, video_path])

    def update_time(self):
        self.elapsed_time += 1
        m, s = divmod(self.elapsed_time, 60)
        self.time_label.setText(f"{m:02d}:{s:02d}")

    def handle_stdout(self):
        data = self.process.readAllStandardOutput().data().decode('utf-8', errors='replace')
        lines = data.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check for timing stats
            stat_keywords = ["YOLO Detection", "Tracking (ReID)", "MediaPipe (Pose)", 
                             "Action (EfficientGCN)", "Face (InsightFace)", "Drawing & Video Write"]
                             
            is_stat = False
            for kw in stat_keywords:
                if kw in line and ":" in line and "%" in line:
                    self.stats_text.append(line)
                    is_stat = True
                    break
            
            if is_stat:
                self.stats_label.setText("\n".join(self.stats_text))
                self.stats_label.show()
            else:
                if len(line) > 80:
                    line = line[:77] + "..."
                self.log_label.setText(line)
                
            if "INFERENCE_COMPLETE" in line:
                self.status_label.setText("Processing Complete!")
                self.timer.stop()

    def handle_finished(self, exit_code, exit_status):
        self.timer.stop()
        if exit_code == 0:
            self.processing_finished.emit(self.time_label.text())
        else:
            self.status_label.setText("Error during processing!")
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(100)
            self.progress_bar.setStyleSheet("QProgressBar::chunk { background-color: red; }")
            self.processing_error.emit("Process finished with error code: " + str(exit_code))
