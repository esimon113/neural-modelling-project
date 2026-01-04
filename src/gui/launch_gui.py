import sys
import os

# Add src directory to path for imports
# This allows importing from gui and other src modules...
src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from gui.main_window import main

if __name__ == "__main__":
    main()
