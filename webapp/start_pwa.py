"""
Quick Start Script for PWA Testing
Run this to start your Flask app and test PWA features
"""

import os
import sys

def main():
    print("=" * 70)
    print("  🚀 Starting Burn Classifier PWA")
    print("=" * 70)
    print()
    print("✅ PWA Features Enabled:")
    print("   • Service Worker for caching")
    print("   • App icons generated")
    print("   • Install prompt ready")
    print("   • Offline support enabled")
    print()
    print("📱 To test on mobile:")
    print("   1. Use ngrok: https://ngrok.com/download")
    print("   2. Run: ngrok http 5000")
    print("   3. Open the ngrok URL on your phone")
    print()
    print("💻 To test on desktop:")
    print("   1. Open http://localhost:5000 in Chrome")
    print("   2. Look for install icon in address bar")
    print("   3. Check DevTools → Application → Service Workers")
    print()
    print("=" * 70)
    print("Starting Flask server...")
    print("=" * 70)
    print()

    # Change to webapp directory
    webapp_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(webapp_dir)

    # Import and run Flask app
    try:
        from app import app
        app.run(debug=True, host='0.0.0.0', port=5000)
    except ImportError as e:
        print(f"❌ Error: Could not import Flask app: {e}")
        print("Make sure you're in the webapp directory and dependencies are installed.")
        sys.exit(1)

if __name__ == '__main__':
    main()
