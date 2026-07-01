from PyQt6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog
from PyQt6.QtCore import pyqtSignal, Qt

class UploadScreen(QWidget):
    video_selected = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.title_label = QLabel("P.R.I.S.M. Video Analysis")
        self.title_label.setStyleSheet("font-size: 32px; font-weight: bold; margin-bottom: 20px;")
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.subtitle_label = QLabel("Upload a video to start processing")
        self.subtitle_label.setStyleSheet("font-size: 16px; margin-bottom: 30px;")
        self.subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.upload_btn = QPushButton("Select Video (.mp4)")
        self.upload_btn.setFixedSize(250, 50)
        self.upload_btn.clicked.connect(self.browse_file)
        
        layout.addWidget(self.title_label)
        layout.addWidget(self.subtitle_label)
        layout.addWidget(self.upload_btn, alignment=Qt.AlignmentFlag.AlignCenter)
        
        self.setLayout(layout)

    def browse_file(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video File",
            "",
            "Video Files (*.mp4 *.avi *.mkv)"
        )
        if file_name:
            self.video_selected.emit(file_name)
