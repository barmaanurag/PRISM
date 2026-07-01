# gui/styles.py

CREAM = "#FDFBF7"
CREAM_DARKER = "#F0EAD6"
ORANGE = "#FF8C00"
ORANGE_HOVER = "#E07B00"
TEXT_COLOR = "#333333"

GLOBAL_STYLES = f"""
QWidget {{
    background-color: {CREAM};
    color: {TEXT_COLOR};
    font-family: "Segoe UI", Arial, sans-serif;
}}

QLabel {{
    background-color: transparent;
}}

QPushButton {{
    background-color: {ORANGE};
    color: white;
    border: none;
    border-radius: 8px;
    padding: 12px 24px;
    font-size: 15px;
    font-style: italic;
    font-weight: bold;
}}

QPushButton:hover {{
    background-color: {ORANGE_HOVER};
    margin-top: -2px; /* subtle lift effect */
    border-bottom: 2px solid #CC7000;
}}

QPushButton:pressed {{
    background-color: #CC7000;
    margin-top: 2px;
    border-bottom: none;
}}

QPushButton:disabled {{
    background-color: #cccccc;
    color: #666666;
    border-bottom: none;
}}

QTextEdit, QLineEdit {{
    background-color: white;
    border: 2px solid {CREAM_DARKER};
    border-radius: 8px;
    padding: 12px;
    font-size: 15px;
}}

QTextEdit:focus, QLineEdit:focus {{
    border: 2px solid {ORANGE};
}}

QProgressBar {{
    border: 2px solid {CREAM_DARKER};
    border-radius: 8px;
    text-align: center;
    color: {TEXT_COLOR};
    font-weight: bold;
    background-color: white;
    height: 24px;
}}

QProgressBar::chunk {{
    background-color: {ORANGE};
    border-radius: 6px;
}}
"""
