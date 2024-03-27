import sys
import os

# 获取当前文件所在的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from answer_searcher import *
from question_parser import *
