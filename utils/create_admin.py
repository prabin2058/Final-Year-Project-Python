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

def create_admin_account(username, password):
    """
    Create an admin account with bcrypt hashed password
    """
    try:
        # Connect to the database
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        
        # Check if admin table exists
        cursor.execute("SHOW TABLES LIKE 'admin'")
        if not cursor.fetchone():
            print("Admin table does not exist. Creating admin table...")
            cursor.execute("""
                CREATE TABLE admin (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    username VARCHAR(255) UNIQUE NOT NULL,
                    password VARCHAR(255) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            print("Admin table created successfully.")
        
        # Check if username already exists
        cursor.execute("SELECT username FROM admin WHERE username = %s", (username,))
        existing_user = cursor.fetchone()
        
        if existing_user:
            print(f"Username '{username}' already exists. Updating password...")
            # Hash the password using bcrypt
            hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
            
            # Update existing user's password
            cursor.execute("UPDATE admin SET password = %s WHERE username = %s", (hashed_password, username))
            conn.commit()
            print(f"Password updated successfully for user '{username}'!")
        else:
            print(f"Creating new admin account for username '{username}'...")
            # Hash the password using bcrypt
            hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
            
            # Insert new admin account
            cursor.execute("INSERT INTO admin (username, password) VALUES (%s, %s)", (username, hashed_password))
            conn.commit()
            print(f"Admin account created successfully!")
        
        print(f"\nAdmin Credentials:")
        print(f"Username: {username}")
        print(f"Password: {password}")
        print(f"\nYou can now login to the admin panel with these credentials.")
        
        cursor.close()
        conn.close()
        
    except mysql.connector.Error as err:
        print(f"Database Error: {err}")
        print("\nPlease make sure:")
        print("1. MySQL server is running")
        print("2. Database 'digitalhajir' exists")
        print("3. Database credentials in db_config are correct")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Create admin account with username='admin' and password='admin123'
    username = "admin"
    password = "admin123"
    
    print("=" * 50)
    print("Admin Account Creation Script")
    print("=" * 50)
    print()
    
    create_admin_account(username, password)
