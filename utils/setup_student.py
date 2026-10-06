import os
import mysql.connector
import bcrypt

# MySQL database configuration (same as in app.py)
db_config = {
    "host": os.environ.get("DB_HOST", "localhost"),
    "user": os.environ.get("DB_USER", "root"),
    "password": os.environ.get("DB_PASSWORD", ""),
    "database": os.environ.get("DB_NAME", "digitalhajir")
}

def setup_student_login():
    """
    Setup student login by adding password column and setting default passwords
    """
    try:
        # Connect to the database
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        
        # Check if password column exists in users table
        cursor.execute("SHOW COLUMNS FROM users LIKE 'password'")
        if not cursor.fetchone():
            print("Adding password column to users table...")
            cursor.execute("ALTER TABLE users ADD COLUMN password VARCHAR(255)")
            conn.commit()
            print("✓ Password column added successfully!")
        else:
            print("✓ Password column already exists")
        
        # Get all users
        cursor.execute("SELECT id, name, rollno, email FROM users")
        users = cursor.fetchall()
        
        if not users:
            print("\n⚠️ No users found in the database!")
            print("Please add users through the admin panel first.")
            cursor.close()
            conn.close()
            return
        
        print(f"\nFound {len(users)} users in the database")
        print("\nSetting default passwords (password = roll number)...\n")
        
        for user in users:
            user_id, name, rollno, email = user
            
            # Set password as roll number
            default_password = str(rollno)
            hashed_password = bcrypt.hashpw(default_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
            
            # Update password
            cursor.execute("UPDATE users SET password = %s WHERE id = %s", (hashed_password, user_id))
            
            print(f"✓ {name} (Roll: {rollno})")
            print(f"  Username: {rollno}")
            print(f"  Password: {rollno}")
            print()
        
        conn.commit()
        cursor.close()
        conn.close()
        
        print("=" * 50)
        print("✅ All student accounts are ready!")
        print("=" * 50)
        print("\nStudents can now login with:")
        print("  Username: Their Roll Number")
        print("  Password: Their Roll Number")
        print("\nExample: Roll 36 → Username: 36, Password: 36")
        
    except mysql.connector.Error as err:
        print(f"❌ Database Error: {err}")
        print("\nPlease make sure:")
        print("1. MySQL server is running")
        print("2. Database 'digitalhajir' exists")
        print("3. Users table exists with data")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("=" * 50)
    print("Student Login Setup Script")
    print("=" * 50)
    print()
    
    setup_student_login()
