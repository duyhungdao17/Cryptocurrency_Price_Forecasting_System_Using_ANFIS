"""
sys.path fix cho imports từ các thư mục khác nhau
Thêm parent directory vào Python path để import modules
"""

import sys
import os

# Lấy đường dẫn của file hiện tại
current_dir = os.path.dirname(os.path.abspath(__file__))
# Lấy parent directory (Crypto_Forecasting_Model)
parent_dir = os.path.dirname(current_dir)

# Thêm vào sys.path nếu chưa có
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
