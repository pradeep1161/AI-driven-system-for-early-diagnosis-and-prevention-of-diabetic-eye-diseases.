#!/usr/bin/env python3
"""
Setup script for AI-DRIVE System
Initializes database and creates admin user
"""

from app import app, db, create_tables
from create_user import create_admin_user

def setup_system():
    """Initialize the system with database and admin user"""
    print("🚀 Setting up AI-DRIVE System...")
    
    # Create database tables
    print("📊 Creating database tables...")
    create_tables()
    
    # Create admin user
    print("👤 Creating admin user...")
    create_admin_user()
    
    print("✅ System setup complete!")
    print("\n🌐 You can now run: python app.py")

if __name__ == "__main__":
    setup_system()
