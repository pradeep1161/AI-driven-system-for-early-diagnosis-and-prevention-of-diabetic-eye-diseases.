#!/usr/bin/env python3
"""
User Management Script for AI-DRIVE System
Creates admin user for login access
"""

from app import app, db, User

def create_admin_user():
    """Create admin user if it doesn't exist"""
    with app.app_context():
        # Check if admin user already exists
        #existing_user = User.query.filter_by(username='admin').first()
        existing_user = None
        if existing_user:
            print("✅ Admin user already exists!")
            print(f"Username: {existing_user.username}")
            print(f"Password: {existing_user.password}")
        else:
            # Create new admin user
            admin_user = User(username='admin', password='Pradeep@116')
            db.session.add(admin_user)
            db.session.commit()
            print("✅ Admin user created successfully!")
            print("Username: admin")
            print("Password: Pradeep@116")
        
        print("\n🔐 Login Credentials:")
        print("Username: admin")
        print("Password: Pradeep@116")
        print("\n🌐 Access the login page at: http://127.0.0.1:8501/login")

if __name__ == "__main__":
    create_admin_user()
