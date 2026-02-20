import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), 'attendance.db')


def get_db():
    """Get a database connection."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    """Create tables and seed default admin if not exists."""
    conn = get_db()
    cursor = conn.cursor()

    # ── Users table ──
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('admin', 'student')),
            name TEXT NOT NULL
        )
    ''')

    # ── Students table ──
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            roll_number TEXT UNIQUE NOT NULL,
            department TEXT NOT NULL,
            face_encoding BLOB,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')

    # ── Attendance table ──
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL,
            date TEXT NOT NULL,
            time TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'Present',
            FOREIGN KEY (student_id) REFERENCES students(id)
        )
    ''')

    # ── Seed default admin ──
    cursor.execute("SELECT id FROM users WHERE email = ?", ('admin@gmail.com',))
    if not cursor.fetchone():
        cursor.execute(
            "INSERT INTO users (email, password, role, name) VALUES (?, ?, ?, ?)",
            ('admin@gmail.com', 'admin123', 'admin', 'Admin')
        )

    conn.commit()
    conn.close()


if __name__ == '__main__':
    init_db()
    print("Database initialized successfully!")
