#!/usr/bin/env python3
"""
Quick setup script to use the enhanced mobile-friendly index.html
This script backs up your current index.html and replaces it with the enhanced version.
"""

import os
import shutil
from datetime import datetime

def main():
    webapp_dir = os.path.dirname(os.path.abspath(__file__))
    templates_dir = os.path.join(webapp_dir, 'templates')
    static_js_dir = os.path.join(webapp_dir, 'static', 'js')

    # Backup current files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Backup index.html
    current_index = os.path.join(templates_dir, 'index.html')
    backup_index = os.path.join(templates_dir, f'index_backup_{timestamp}.html')

    if os.path.exists(current_index):
        shutil.copy2(current_index, backup_index)
        print(f"✅ Backed up current index.html to: {backup_index}")

    # Backup main.js
    current_js = os.path.join(static_js_dir, 'main.js')
    backup_js = os.path.join(static_js_dir, f'main_backup_{timestamp}.js')

    if os.path.exists(current_js):
        shutil.copy2(current_js, backup_js)
        print(f"✅ Backed up current main.js to: {backup_js}")

    # Replace with enhanced versions
    enhanced_index = os.path.join(templates_dir, 'index_enhanced.html')
    enhanced_js = os.path.join(static_js_dir, 'main_enhanced.js')

    if os.path.exists(enhanced_index):
        shutil.copy2(enhanced_index, current_index)
        print(f"✅ Replaced index.html with enhanced version")
    else:
        print(f"❌ Enhanced index.html not found at: {enhanced_index}")

    if os.path.exists(enhanced_js):
        shutil.copy2(enhanced_js, current_js)
        print(f"✅ Replaced main.js with enhanced version")
    else:
        print(f"❌ Enhanced main.js not found at: {enhanced_js}")

    print("\n🎉 Setup complete! Your webapp now has mobile tap/hold functionality:")
    print("   • Tap the camera icon to select from gallery")
    print("   • Hold for 1.5 seconds to open camera")
    print("   • Progress animation shows hold duration")
    print("   • Haptic feedback on supported devices")
    print("\n📱 Test on mobile device for best experience!")
    print("\n🔄 To revert, use the backup files created above")

def revert():
    """Revert to backup files"""
    webapp_dir = os.path.dirname(os.path.abspath(__file__))
    templates_dir = os.path.join(webapp_dir, 'templates')
    static_js_dir = os.path.join(webapp_dir, 'static', 'js')

    # Find most recent backup files
    backup_files = []

    for file in os.listdir(templates_dir):
        if file.startswith('index_backup_'):
            backup_files.append(os.path.join(templates_dir, file))

    if backup_files:
        latest_backup = max(backup_files, key=os.path.getctime)
        current_index = os.path.join(templates_dir, 'index.html')
        shutil.copy2(latest_backup, current_index)
        print(f"✅ Reverted index.html from: {latest_backup}")

    backup_js_files = []
    for file in os.listdir(static_js_dir):
        if file.startswith('main_backup_'):
            backup_js_files.append(os.path.join(static_js_dir, file))

    if backup_js_files:
        latest_backup_js = max(backup_js_files, key=os.path.getctime)
        current_js = os.path.join(static_js_dir, 'main.js')
        shutil.copy2(latest_backup_js, current_js)
        print(f"✅ Reverted main.js from: {latest_backup_js}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'revert':
        revert()
    else:
        main()
