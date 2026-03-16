import sqlite3
conn = sqlite3.connect('attendance.db')
cursor = conn.cursor()
cursor.execute("SELECT email, password FROM users WHERE email LIKE 'faculty%'")
row = cursor.fetchone()
if row:
    print(f"Email: {row[0]}, Password: {row[1]}")
else:
    print("Faculty not found.")
conn.close()
