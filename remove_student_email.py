#!/usr/bin/env python3
"""
Script to remove a student email from the database.
This allows the student to re-register.
"""

from database import get_db

def remove_student_email(email):
    """Remove student with given email from database."""
    conn = get_db()
    cursor = conn.cursor()
    
    try:
        # 1. Check student_registration_requests table
        cursor.execute("SELECT id FROM student_registration_requests WHERE email = ?", (email,))
        reg_request = cursor.fetchone()
        
        if reg_request:
            print(f"✓ Found in registration_requests (ID: {reg_request[0]})")
            cursor.execute("DELETE FROM student_registration_requests WHERE email = ?", (email,))
            print(f"  → Deleted from student_registration_requests")
        
        # 2. Check users table
        cursor.execute("SELECT id FROM users WHERE email = ?", (email,))
        user = cursor.fetchone()
        
        if user:
            user_id = user[0]
            print(f"✓ Found in users (ID: {user_id})")
            
            # Delete from students table first (foreign key constraint)
            cursor.execute("DELETE FROM students WHERE user_id = ?", (user_id,))
            print(f"  → Deleted from students table")
            
            # Delete from users table
            cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
            print(f"  → Deleted from users table")
        
        conn.commit()
        
        if reg_request or user:
            print(f"\n✅ Successfully removed {email} from database!")
        else:
            print(f"\n⚠️  Email '{email}' not found in database.")
        
        return True
    
    except Exception as e:
        conn.rollback()
        print(f"\n❌ Error: {e}")
        return False
    
    finally:
        conn.close()


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python remove_student_email.py <email>")
        print("Example: python remove_student_email.py sv0002@srmist.edu.in")
        sys.exit(1)
    
    email = sys.argv[1]
    print(f"Removing email: {email}\n")
    remove_student_email(email)
