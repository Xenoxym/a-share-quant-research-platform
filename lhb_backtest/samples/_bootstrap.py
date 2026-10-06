"""samples 公共引导：把项目根目录加入 sys.path，并配置控制台输出。

每个 sample 脚本开头 ``import _bootstrap`` 即可直接 import src.* 模块。
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 保持控制台原生编码（GBK/UTF-8 均可正常显示中文），
# 仅将无法编码的字符替换为 '?'，避免 UnicodeEncodeError 崩溃
for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(errors="replace")
    except Exception:
        pass
