"""
Pytest configuration and fixture setups.
"""

import sys
from pathlib import Path

# Add project root directory to sys.path for test imports
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
