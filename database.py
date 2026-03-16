import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), 'attendance.db')


def get_db():
    """Get a database connection.

    We enable a longer timeout and set journal_mode to WAL so that the
    camera thread and admin actions can coexist without frequently hitting
    "database is locked" errors.  Foreign keys are enforced as before.
    """
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    # allow multiple writers/readers using WAL mode
    conn.execute("PRAGMA journal_mode = WAL")
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
            role TEXT NOT NULL CHECK(role IN ('admin', 'student', 'faculty')),
            name TEXT NOT NULL
        )
    ''')

    # ── Faculty table ──
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS faculty (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            faculty_id TEXT NOT NULL DEFAULT '',
            subject TEXT NOT NULL,
            department TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')

    # Migration: add faculty_id column to existing DBs (SQLite can't add UNIQUE via ALTER TABLE)
    try:
        cursor.execute("ALTER TABLE faculty ADD COLUMN faculty_id TEXT NOT NULL DEFAULT ''")
    except Exception:
        pass  # Column already exists

    # Enforce uniqueness via index (works on both new and migrated tables)
    cursor.execute('''
        CREATE UNIQUE INDEX IF NOT EXISTS uq_faculty_faculty_id ON faculty(faculty_id)
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

    # ── Sessions table (Attendance Sessions) ──
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            faculty_id INTEGER NOT NULL,
            subject TEXT NOT NULL,
            date TEXT NOT NULL,
            start_time TEXT NOT NULL,
            end_time TEXT,
            is_active INTEGER DEFAULT 1,
            FOREIGN KEY (faculty_id) REFERENCES faculty(id)
        )
    ''')

    # ── Attendance table (Modified for sessions) ──
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL,
            session_id INTEGER,
            date TEXT NOT NULL,
            time TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'Present',
            FOREIGN KEY (student_id) REFERENCES students(id),
            FOREIGN KEY (session_id) REFERENCES sessions(id),
            UNIQUE(student_id, session_id, date)
        )
    ''')

    # ── student_registration_requests table ──
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS student_registration_requests (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name      TEXT NOT NULL,
            reg_number     TEXT NOT NULL,
            email          TEXT NOT NULL,
            password       TEXT NOT NULL,
            department     TEXT NOT NULL,
            status         TEXT NOT NULL DEFAULT 'Pending'
                               CHECK(status IN ('Pending', 'Approved', 'Rejected')),
            otp_code       TEXT,
            otp_expiry     TEXT,
            otp_attempts   INTEGER NOT NULL DEFAULT 0,
            email_verified INTEGER NOT NULL DEFAULT 0,
            created_at     TEXT NOT NULL
        )
    ''')

    # Migration: add otp_attempts column to existing DBs that were created before this update
    try:
        cursor.execute("ALTER TABLE student_registration_requests ADD COLUMN otp_attempts INTEGER NOT NULL DEFAULT 0")
    except Exception:
        pass  # Column already exists

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