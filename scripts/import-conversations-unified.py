#!/usr/bin/env python3
"""
Legacy import script for backward compatibility.
Redirects to the new smart incremental importer.
"""

import sys
import os

# Add the script directory to path to import the new script
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

# Import and run the new smart watcher
from import_watcher_smart import main

if __name__ == "__main__":
    print("Note: import-conversations-unified.py is deprecated.")
    print("Redirecting to import-watcher-smart.py...")
    main()