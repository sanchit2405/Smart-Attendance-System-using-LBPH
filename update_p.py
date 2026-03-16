import sqlite3
import os

DB_PATH = 'attendance.db'

def update_password():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Update for both correct email and possible typo
    emails = ['faculty@gmail.com', 'faculty@gamil.com']
    
    updated = False
    for email in emails:
        cursor.execute("UPDATE users SET password = 'fffff' WHERE email = ?", (email,))
        if cursor.rowcount > 0:
            print(f"Password updated for {email}")
            updated = True
    
    if not updated:
        # Fallback: find any user starting with faculty
        cursor.execute("UPDATE users SET password = 'fffff' WHERE email LIKE 'faculty%'")
        if cursor.rowcount > 0:
            print("Password updated for faculty user (matched by pattern)")
            updated = True
            
    conn.commit()
    conn.close()
    if updated:
        print("Success!")
    else:
        print("No faculty user found to update.")

if __name__ == '__main__':
    update_password()
