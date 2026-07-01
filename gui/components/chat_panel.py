from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QLineEdit, QPushButton, QLabel
from PyQt6.QtCore import pyqtSignal, Qt, QProcess, QTimer
import sys
import os
import re

class ChatPanel(QWidget):
    def __init__(self):
        super().__init__()
        
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.proc_time_label = QLabel("")
        self.proc_time_label.setStyleSheet("color: #888888; font-size: 13px; font-weight: bold;")
        self.proc_time_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        
        # Modern Loading Animation
        self.loading_container = QWidget()
        loading_layout = QVBoxLayout()
        loading_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.loading_label = QLabel("The LLM is thinking...")
        self.loading_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #FF8C00; font-style: italic;")
        self.loading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        from components.animations import BouncingDots
        self.loader_anim = BouncingDots(color="#FF8C00")
        
        loading_layout.addWidget(self.loader_anim, alignment=Qt.AlignmentFlag.AlignCenter)
        loading_layout.addWidget(self.loading_label)
        self.loading_container.setLayout(loading_layout)
        
        self.chat_history = QTextEdit()
        self.chat_history.setReadOnly(True)
        self.chat_history.setStyleSheet("background-color: #ffffff; border-radius: 8px; font-size: 14px; padding: 15px; border: 1px solid #F0EAD6;")
        self.chat_history.hide() # Hidden initially
        
        input_layout = QHBoxLayout()
        self.query_input = QLineEdit()
        self.query_input.setPlaceholderText("Ask a question about the video...")
        self.query_input.setMinimumHeight(45)
        self.query_input.returnPressed.connect(self.send_query)
        
        self.send_btn = QPushButton("Send")
        self.send_btn.setFixedSize(100, 45)
        self.send_btn.clicked.connect(self.send_query)
        
        self.chat_loader = BouncingDots(color="#FF8C00")
        self.chat_loader.setFixedSize(80, 45)
        self.chat_loader.hide()
        
        input_layout.addWidget(self.query_input)
        input_layout.addWidget(self.chat_loader)
        input_layout.addWidget(self.send_btn)
        
        layout.addWidget(self.proc_time_label)
        layout.addWidget(self.loading_container)
        layout.addWidget(self.chat_history)
        layout.addLayout(input_layout)
        
        self.setLayout(layout)
        
        self.process = None
        self.is_ready = False
        self.summary_generating = True
        self.summary_time = ""
        
        self.loading_timer = QTimer()
        self.loading_timer.timeout.connect(self.animate_loading)
        self.dot_count = 0

    def animate_loading(self):
        self.dot_count = (self.dot_count + 1) % 4
        self.loading_label.setText("The LLM is thinking" + "." * self.dot_count)

    def set_processing_time(self, time_str):
        if time_str:
            self.proc_time_label.setText(f"⏱️ Video Processed in {time_str}")
        else:
            self.proc_time_label.hide()

    def start_chat(self, json_log_path="output_action_action_log.json"):
        self.query_input.setEnabled(False)
        self.send_btn.setEnabled(False)
        
        self.loader_anim.start()
        self.loading_timer.start(500)
        
        self.process = QProcess()
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.process.readyReadStandardOutput.connect(self.handle_stdout)
        
        script_path = os.path.join(os.path.dirname(__file__), '..', '..', 'nest_rag.py')
        script_path = os.path.abspath(script_path)
        
        python_exe = sys.executable
        self.process.start(python_exe, [script_path, '--json', json_log_path, '--regenerate'])

    def handle_stdout(self):
        data = self.process.readAllStandardOutput().data().decode('utf-8', errors='replace')
        
        # During summary generation, suppress output but look for the completion markers
        if self.summary_generating:
            for line in data.split('\n'):
                if "⚡" in line and "wall" in line:
                    # Extract the wall time
                    match = re.search(r'([\d\.]+s wall)', line)
                    if match:
                        self.summary_time = match.group(1)
                
                if "✓ Summary saved →" in line:
                    self.summary_generating = False
                    self.loader_anim.stop()
                    self.loading_timer.stop()
                    self.loading_container.hide()
                    self.chat_history.show()
                    
                    # Read the summary file
                    summary_path = line.split("→")[1].strip()
                    if os.path.exists(summary_path):
                        with open(summary_path, 'r', encoding='utf-8') as f:
                            summary_text = f.read()
                            # Format for HTML
                            summary_html = summary_text.replace('\n', '<br>')
                            time_msg = f"⏱️ <i>Summary generated in {self.summary_time}</i>" if self.summary_time else ""
                            self.chat_history.append(f"<h3 style='color: #FF8C00;'>Video Summary</h3>{summary_html}<br>{time_msg}<hr>")
            return

        # Normal chat mode
        if not self.is_ready:
            if "❓ Question:" in data:
                self.is_ready = True
                self.chat_loader.stop()
                self.send_btn.show()
                self.query_input.setEnabled(True)
                self.send_btn.setEnabled(True)
                # Discard all boilerplate before the first prompt
                data = data.split("❓ Question:")[-1]
            else:
                # Ignore all boilerplate text while waiting for readiness
                return
        else:
            if "❓ Question:" in data:
                self.chat_loader.stop()
                self.send_btn.show()
                self.query_input.setEnabled(True)
                self.send_btn.setEnabled(True)
                data = data.replace("❓ Question:", "")
        if data:
            # Hide some terminal artifacts from Ollama stream formatting if needed
            data = re.sub(r'─+', '', data)
            data = re.sub(r'ANSWER:.*', '', data)
            data = data.replace('⚡', '\n⚡')
            
            if data:
                from PyQt6.QtGui import QTextBlockFormat, QTextCharFormat, QColor
                
                # Stop the loader as soon as we start receiving answer tokens
                if self.chat_loader.isVisible():
                    self.chat_loader.stop()
                    self.send_btn.show()
                    
                    cursor = self.chat_history.textCursor()
                    cursor.movePosition(cursor.MoveOperation.End)
                    
                    # Create left-aligned block for AI
                    block_fmt = QTextBlockFormat()
                    block_fmt.setAlignment(Qt.AlignmentFlag.AlignLeft)
                    block_fmt.setTopMargin(15)
                    cursor.insertBlock(block_fmt)
                    cursor.insertHtml("<b style='color: #FF8C00;'>P.R.I.S.M.:</b><br>")
                    self.chat_history.setTextCursor(cursor)
                    
                # Insert streamed text inline (preserves spaces!)
                cursor = self.chat_history.textCursor()
                cursor.movePosition(cursor.MoveOperation.End)
                
                char_fmt = QTextCharFormat()
                char_fmt.setForeground(QColor("#444444"))
                cursor.insertText(data, char_fmt)
                
                scrollbar = self.chat_history.verticalScrollBar()
                scrollbar.setValue(scrollbar.maximum())

    def send_query(self):
        query = self.query_input.text().strip()
        if not query or not self.process or not self.is_ready:
            return
            
        from PyQt6.QtGui import QTextBlockFormat
        cursor = self.chat_history.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        
        # Create right-aligned block for user
        block_fmt = QTextBlockFormat()
        block_fmt.setAlignment(Qt.AlignmentFlag.AlignRight)
        block_fmt.setTopMargin(15)
        cursor.insertBlock(block_fmt)
        cursor.insertHtml(f"<span style='color: #FF8C00;'><b>You:</b></span> <span style='color: #333;'>{query}</span>")
        self.chat_history.setTextCursor(cursor)
        
        self.query_input.clear()
        self.query_input.setEnabled(False)
        self.send_btn.hide()
        self.chat_loader.start()
        
        query_bytes = (query + "\n").encode('utf-8')
        self.process.write(query_bytes)
        
    def closeEvent(self, event):
        if self.process:
            self.process.kill()
        super().closeEvent(event)
