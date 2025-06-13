#!/usr/bin/env python3
"""
Startup script for OpenAI Token Counter & Cost Calculator
"""

import sys
import subprocess
import os
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8 or higher is required.")
        print(f"Current version: {sys.version}")
        sys.exit(1)
    print(f"✅ Python version: {sys.version.split()[0]}")

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import streamlit
        print(f"✅ Streamlit version: {streamlit.__version__}")
    except ImportError:
        print("❌ Streamlit not found. Installing dependencies...")
        install_dependencies()
        return

    try:
        import tiktoken
        print(f"✅ tiktoken installed")
    except ImportError:
        print("❌ tiktoken not found. Installing dependencies...")
        install_dependencies()
        return

def install_dependencies():
    """Install required dependencies"""
    requirements_file = Path("requirements.txt")
    
    if not requirements_file.exists():
        print("❌ requirements.txt not found!")
        sys.exit(1)
    
    print("📦 Installing dependencies...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("✅ Dependencies installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        sys.exit(1)

def check_app_file():
    """Check if main application file exists"""
    app_file = Path("app.py")
    if not app_file.exists():
        print("❌ app.py not found!")
        print("Make sure you're running this script from the project directory.")
        sys.exit(1)
    print("✅ Main application file found")

def run_streamlit():
    """Run the Streamlit application"""
    print("\n🚀 Starting OpenAI Token Counter & Cost Calculator...")
    print("📊 Application will open in your default web browser")
    print("🔗 URL: http://localhost:8501")
    print("\n⏹️  Press Ctrl+C to stop the application\n")
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.address", "localhost",
            "--server.port", "8501",
            "--server.headless", "false",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n\n👋 Application stopped by user")
    except Exception as e:
        print(f"\n❌ Error running application: {e}")
        sys.exit(1)

def main():
    """Main startup function"""
    print("🧮 OpenAI Token Counter & Cost Calculator")
    print("=" * 50)
    
    # Check system requirements
    check_python_version()
    check_app_file()
    check_dependencies()
    
    print("\n" + "=" * 50)
    
    # Run the application
    run_streamlit()

if __name__ == "__main__":
    main() 