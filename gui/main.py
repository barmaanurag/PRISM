import sys
import os
from PyQt6.QtWidgets import QApplication, QMainWindow, QStackedWidget, QVBoxLayout, QWidget, QSplitter
from PyQt6.QtCore import Qt

# Import styles
from styles import GLOBAL_STYLES

# Import components
from components.upload_screen import UploadScreen
from components.loading_screen import LoadingScreen
from components.video_player import VideoPlayer
from components.chat_panel import ChatPanel

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("P.R.I.S.M. - Modern Analysis GUI")
        self.resize(1000, 800)
        
        # Apply global styles
        self.setStyleSheet(GLOBAL_STYLES)
        
        # Main stacked widget to switch between screens
        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)
        
        # 1. Upload Screen
        self.upload_screen = UploadScreen()
        self.upload_screen.video_selected.connect(self.start_processing)
        self.stacked_widget.addWidget(self.upload_screen)
        
        # 2. Loading Screen
        self.loading_screen = LoadingScreen()
        self.loading_screen.processing_finished.connect(self.show_results)
        self.stacked_widget.addWidget(self.loading_screen)
        
        # 3. Results Screen (Video + Chat)
        self.results_widget = QWidget()
        results_layout = QVBoxLayout()
        results_layout.setContentsMargins(20, 20, 20, 20)
        
        # We use a splitter so user can resize video vs chat
        self.splitter = QSplitter(Qt.Orientation.Vertical)
        
        self.video_player = VideoPlayer()
        self.chat_panel = ChatPanel()
        
        self.splitter.addWidget(self.video_player)
        self.splitter.addWidget(self.chat_panel)
        # Give equal initial space
        self.splitter.setSizes([400, 400])
        
        results_layout.addWidget(self.splitter)
        self.results_widget.setLayout(results_layout)
        self.stacked_widget.addWidget(self.results_widget)

    def start_processing(self, video_path):
        # Switch to loading screen
        self.stacked_widget.setCurrentWidget(self.loading_screen)
        # Start the background process
        self.loading_screen.start_processing(video_path)

    def show_results(self, processing_time_str=""):
        # The processing script outputs 'output_action.mp4' and 'output_action_action_log.json' in the workspace root
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        video_out = os.path.join(base_dir, 'output_action.mp4')
        json_log = os.path.join(base_dir, 'output_action_action_log.json')
        
        # Load the video
        self.video_player.load_video(video_out)
        
        # Set processing time
        self.chat_panel.set_processing_time(processing_time_str)
        
        # Start the chat agent
        self.chat_panel.start_chat(json_log)
        
        # Switch to results screen
        self.stacked_widget.setCurrentWidget(self.results_widget)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
