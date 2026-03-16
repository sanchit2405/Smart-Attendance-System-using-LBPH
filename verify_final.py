import sqlite3
with sqlite3.connect('attendance.db') as conn:
    p = conn.execute("SELECT password FROM users WHERE email='faculty@gmail.com'").fetchone()
    with open('status.txt', 'w') as f:
        f.write(p[0] if p else 'NOT FOUND')
