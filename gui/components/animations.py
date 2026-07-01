from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import QTimer, Qt, QRectF
from PyQt6.QtGui import QPainter, QColor, QBrush
import math

class BouncingDots(QWidget):
    def __init__(self, color="#FF8C00", parent=None):
        super().__init__(parent)
        self.setFixedSize(100, 40)
        self.color = QColor(color)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_animation)
        self.tick = 0
        
    def start(self):
        self.timer.start(50)
        self.show()
        
    def stop(self):
        self.timer.stop()
        self.hide()
        
    def update_animation(self):
        self.tick += 1
        self.update()
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(self.color))
        
        # Draw 3 animated dots
        for i in range(3):
            # Sine wave for smooth bouncing
            offset = math.sin(self.tick * 0.2 - i * 0.8) * 8
            y = 20 + offset
            x = 20 + i * 25
            
            # Pulse the size slightly too
            size_offset = math.sin(self.tick * 0.2 - i * 0.8) * 2
            size = 10 + size_offset
            
            painter.drawEllipse(QRectF(x - size/2, y - size/2, size, size))
