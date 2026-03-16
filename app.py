from flask import Flask, render_template, request, redirect, url_for, session, flash, Response, jsonify
from database import get_db, init_db
from datetime import date, datetime, timedelta
import cv2
import numpy as np
import os
import shutil
import threading
import sqlite3
import random
import traceback
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from werkzeug.security import generate_password_hash, check_password_hash

# ── Gmail SMTP config ──
# Replace the strings below with your Gmail and App Password, OR set them
# as environment variables GMAIL_USER and GMAIL_APP_PASSWORD.
GMAIL_USER     = os.environ.get("GMAIL_USER", "svashisth2405@gmail.com")
GMAIL_PASSWORD = os.environ.get("GMAIL_APP_PASSWORD", "haffdtudajobgivz")

app = Flask(__name__)
app.secret_key = 'smart-attendance-secret-key-2026'

# Initialize database on startup
init_db()


# ══════════════════════════════
# MIGRATE FACE FOLDERS (student_id → roll_number)
# ══════════════════════════════
def migrate_face_folders():
    """Rename face folders from student_id to roll_number (one-time migration)."""
    if not os.path.exists(FACES_DIR):
        return
    conn = get_db()
    for folder_name in os.listdir(FACES_DIR):
        folder_path = os.path.join(FACES_DIR, folder_name)
        if not os.path.isdir(folder_path):
            continue
        # Check if folder name is a numeric student_id
        try:
            student_id = int(folder_name)
        except ValueError:
            continue  # Already a roll_number or non-numeric
        # Look up roll_number for this student_id
        student = conn.execute(
            "SELECT roll_number FROM students WHERE id = ?", (student_id,)
        ).fetchone()
        if student:
            new_path = os.path.join(FACES_DIR, student['roll_number'])
            if not os.path.exists(new_path) and folder_path != new_path:
                os.rename(folder_path, new_path)
    conn.close()

# ══════════════════════════════
# PATHS
# ══════════════════════════════
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FACES_DIR = os.path.join(BASE_DIR, 'static', 'faces')
MODEL_DIR = os.path.join(BASE_DIR, 'model')
MODEL_PATH = os.path.join(MODEL_DIR, 'trained_model.yml')
LABELS_PATH = os.path.join(MODEL_DIR, 'labels.npy')

os.makedirs(FACES_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Run migration on startup
migrate_face_folders()


# ══════════════════════════════
# CAMERA SETUP
# ══════════════════════════════

# Load Face Detection Model (Haar Cascade)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Thread lock for camera access
camera_lock = threading.Lock()
# Thread lock for attendance marking to prevent race conditions
attendance_lock = threading.Lock()

# Global frame for capture
latest_frame = None
capture_camera_active = False

# Training state
training_status = {'running': False, 'message': '', 'success': False}

# Global attendance tracking to prevent duplicates
marked_today = set()
# Track records that were manually removed so the camera doesn't re-add them during
# the same live session. Deletion routes will add a key here; the set is reset
# when a new session starts (or on server restart).
blocked_today = set()


def _iou(boxA, boxB):
    """Intersection over Union between two (x,y,w,h) boxes."""
    ax, ay, aw, ah = boxA
    bx, by, bw, bh = boxB
    ix = max(ax, bx)
    iy = max(ay, by)
    iw = min(ax + aw, bx + bw) - ix
    ih = min(ay + ah, by + bh) - iy
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def generate_frames():
    """Generate live camera frames — multi-face detection + stable recognition."""
    from collections import deque, Counter
    global marked_today

    with camera_lock:
        cam = cv2.VideoCapture(0,cv2.CAP_DSHOW)                     # for laptop
        #cam = cv2.VideoCapture("http://10.3.10.87:4747/video")     # for cameras
        

    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 480)    # Raised for multi-face detection
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)   # Raised for multi-face detection
    cam.set(cv2.CAP_PROP_FPS, 15)             # Limit FPS to 15 for network stability

    if not cam.isOpened():
        cam.release()
        return

    recognizer = None
    label_map = {}

    if os.path.exists(MODEL_PATH) and os.path.exists(LABELS_PATH):
        try:
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            recognizer.read(MODEL_PATH)
            label_map = np.load(LABELS_PATH, allow_pickle=True).item()
        except AttributeError:
            recognizer = None

    student_name_cache = {}
    conn = get_db()
    rows = conn.execute(
        "SELECT s.id, u.name FROM students s JOIN users u ON s.user_id = u.id"
    ).fetchall()
    conn.close()

    for row in rows:
        student_name_cache[row['id']] = row['name']

    BUFFER_SIZE = 25
    CONFIDENCE_THRESH = 95
    AVG_CONF_THRESH = 95
    VOTE_MAJORITY = 0.6
    IOU_MATCH_THRESH = 0.25      # Lowered so adjacent face tracks don't collide
    MAX_MISSING_FRAMES = 10
    MIN_FRAMES_TO_DECIDE = 20
    RETRY_AFTER_UNKNOWN = 8       # ~1 s: 15 FPS ÷ FRAME_SKIP(2) ≈ 8 processed frames/s
    FRAME_SKIP = 2               # Process every 2nd frame to reduce lag

    clahe = cv2.createCLAHE(2.0, (8, 8))

    next_track_id = 0
    tracks = {}
    frame_count = 0

    def draw_overlays(img):
        """Draw cached track overlays onto img (in-place)."""
        h_img, w_img = img.shape[:2]
        for tid, tr in tracks.items():
            if tr['missing'] > 0:
                continue
            x, y, w, h = [int(v) for v in tr['box']]
            x = min(max(x, 0), w_img - 1)
            y = min(max(y, 0), h_img - 1)
            w = min(w, w_img - x)
            h = min(h, h_img - y)
            if w <= 0 or h <= 0:
                continue

            dname, _ = tr['confirmed']
            if 'Scanning' in dname:
                bclr = (0, 165, 255)
                pclr = (0, 140, 215)
                tr['scan_dots'] = (tr.get('scan_dots', 0) + 1) % 4
                lbl  = 'Scanning' + '.' * tr['scan_dots']
            elif dname == 'Unknown':
                bclr = (50, 50, 220)
                pclr = (40, 40, 200)
                lbl  = 'Unknown'
            else:
                bclr = (30, 200, 60)
                pclr = (20, 170, 50)
                lbl  = dname

            # corner-bracket rectangle
            crn = min(14, w // 5, h // 5)
            cv2.rectangle(img, (x, y), (x+w, y+h), bclr, 1)
            cv2.line(img, (x, y), (x+crn, y), bclr, 3)
            cv2.line(img, (x, y), (x, y+crn), bclr, 3)
            cv2.line(img, (x+w, y), (x+w-crn, y), bclr, 3)
            cv2.line(img, (x+w, y), (x+w, y+crn), bclr, 3)
            cv2.line(img, (x, y+h), (x+crn, y+h), bclr, 3)
            cv2.line(img, (x, y+h), (x, y+h-crn), bclr, 3)
            cv2.line(img, (x+w, y+h), (x+w-crn, y+h), bclr, 3)
            cv2.line(img, (x+w, y+h), (x+w, y+h-crn), bclr, 3)

            # pill label
            font = cv2.FONT_HERSHEY_SIMPLEX
            fs, ft = 0.46, 1
            (tw, th), baseline = cv2.getTextSize(lbl, font, fs, ft)
            px, py = 7, 4
            px1 = x
            py1 = max(0, y - th - 2*py - 2)
            px2 = x + tw + 2*px
            py2 = max(th + 2*py, y)
            cv2.rectangle(img, (px1, py1), (px2, py2), pclr, cv2.FILLED)
            cv2.putText(img, lbl, (px1+px, py2-py-baseline),
                        font, fs, (255, 255, 255), ft, cv2.LINE_AA)

    try:
        while True:
            success, frame = cam.read()
            frame_count += 1
            skip_processing = frame_count % FRAME_SKIP != 0
            if not success or frame is None:
                break

            h_frame, w_frame = frame.shape[:2]
            
            # Skip heavy processing on alternate frames — but still draw overlays
            if skip_processing:
                draw_overlays(frame)
                ret, buffer = cv2.imencode('.jpg', frame)
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' +
                           buffer.tobytes() + b'\r\n')
                continue
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            raw_faces = face_cascade.detectMultiScale(
                gray, 1.05, 3, minSize=(35, 35)
            )

            detected = list(raw_faces) if len(raw_faces) > 0 else []

            matched_tracks = set()
            matched_detects = set()

            for tid, track in tracks.items():
                best_iou = IOU_MATCH_THRESH
                best_didx = -1

                for didx, det_box in enumerate(detected):
                    if didx in matched_detects:
                        continue
                    score = _iou(track['box'], det_box)
                    if score > best_iou:
                        best_iou = score
                        best_didx = didx

                if best_didx >= 0:
                    matched_tracks.add(tid)
                    matched_detects.add(best_didx)
                    tracks[tid]['box'] = detected[best_didx]
                    tracks[tid]['missing'] = 0

            for tid in list(tracks.keys()):
                if tid not in matched_tracks:
                    tracks[tid]['missing'] += 1

            for tid in [t for t, v in tracks.items() if v['missing'] > MAX_MISSING_FRAMES]:
                del tracks[tid]

            for didx, det_box in enumerate(detected):
                if didx not in matched_detects:
                    tracks[next_track_id] = {
                        'box': det_box,
                        'buffer': deque(maxlen=BUFFER_SIZE),
                        'confirmed': ('Scanning...', None),
                        'missing': 0,
                        'retry_counter': 0,
                        'stable_count': 0,
                        'scan_dots': 0
                    }
                    next_track_id += 1

            for tid, track in tracks.items():
                if track['missing'] > 0:
                    continue

                x, y, w, h = track['box']

                # SAFE CLAMP (prevents crash)
                x = max(0, x)
                y = max(0, y)
                w = min(w_frame - x, w)
                h = min(h_frame - y, h)

                face_roi = gray[y:y+h, x:x+w]
                if face_roi.size == 0:
                    continue

                face_roi = cv2.resize(face_roi, (100, 100))
                face_roi = clahe.apply(face_roi)

                raw_label, raw_conf = -1, 999.0

                if recognizer is not None:
                    raw_label, raw_conf = recognizer.predict(face_roi)

                if raw_conf < CONFIDENCE_THRESH and raw_label in label_map:
                    track['buffer'].append((label_map[raw_label], raw_conf))
                else:
                    track['buffer'].append((None, raw_conf))

                buf = track['buffer']

                if track['confirmed'][0] == 'Unknown':
                    track['retry_counter'] += 1
                    if track['retry_counter'] >= RETRY_AFTER_UNKNOWN:
                        track['buffer'].clear()
                        track['confirmed'] = ('Scanning...', None)
                        track['retry_counter'] = 0

                elif len(buf) >= MIN_FRAMES_TO_DECIDE:
                    votes = [(sid, conf) for sid, conf in buf if sid is not None]
                    if votes:
                        vote_ids = [sid for sid, _ in votes]
                        top_id, top_count = Counter(vote_ids).most_common(1)[0]
                        top_confs = [conf for sid, conf in votes if sid == top_id]
                        avg_conf = sum(top_confs) / len(top_confs)

                        STABLE_CONFIRM_FRAMES = 8   # number of stable frames before confirming

                        if (top_count / len(buf) >= VOTE_MAJORITY
                                and avg_conf < AVG_CONF_THRESH):

                            # Increase stability counter
                            track['stable_count'] = track.get('stable_count', 0) + 1

                            # Only confirm after face is stable
                            if track['stable_count'] >= STABLE_CONFIRM_FRAMES:
                                name = student_name_cache.get(top_id, 'Unknown')
                                track['confirmed'] = (name, top_id)
                                track['retry_counter'] = 0

                        else:
                            # Reset stability if recognition not strong
                            track['stable_count'] = 0
                            track['confirmed'] = ('Unknown', None)
                            track['retry_counter'] = 0
                    else:
                        track['confirmed'] = ('Unknown', None)
                        track['retry_counter'] = 0

                display_name, confirmed_id = track['confirmed']

                if confirmed_id:
                    today = date.today().isoformat()
                    now = datetime.now().strftime('%H:%M:%S')

                    with attendance_lock:
                        conn = get_db()

                        # Get active session
                        active_session = conn.execute(
                            "SELECT id FROM sessions WHERE is_active = 1 AND date = ? ORDER BY id DESC LIMIT 1",
                            (today,)
                        ).fetchone()

                        session_id = active_session['id'] if active_session else None

                        # unique key = (student, session) — each session tracks independently
                        unique_key = (confirmed_id, session_id)

                        # skip if already marked or we have explicitly blocked this student
                        if unique_key not in marked_today and unique_key not in blocked_today:

                            # Check if this student was already marked in THIS specific session
                            already_marked = conn.execute(
                                "SELECT id FROM attendance WHERE student_id = ? AND date = ? AND session_id IS ?",
                                (confirmed_id, today, session_id)
                            ).fetchone()

                            if not already_marked:
                                try:
                                    conn.execute(
                                        "INSERT INTO attendance (student_id, session_id, date, time, status) VALUES (?, ?, ?, ?, 'Present')",
                                        (confirmed_id, session_id, today, now)
                                    )
                                    conn.commit()
                                    marked_today.add(unique_key)

                                except sqlite3.IntegrityError:
                                    # if another thread already inserted a duplicate, still mark
                                    marked_today.add(unique_key)
                            else:
                                marked_today.add(unique_key)

                        conn.close()

            # Draw all track overlays onto this (fully-processed) frame
            draw_overlays(frame)

            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' +
                   buffer.tobytes() + b'\r\n')

    finally:
        cam.release()


def generate_capture_frames():
    """Generate camera frames for face capture (student registration)."""
    global latest_frame, capture_camera_active
    capture_camera_active = True

    cam = cv2.VideoCapture(0,cv2.CAP_DSHOW)                     # for laptops
    #cam = cv2.VideoCapture("http://10.3.10.87:4747/video")     # for cameras
    

    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 320)    # Lowered from 640 for speed
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)   # Lowered from 480 for speed
    cam.set(cv2.CAP_PROP_FPS, 15)             # Limit FPS to 15 for network stability

    if not cam.isOpened():
        cam.release()
        capture_camera_active = False
        return

    try:
        while capture_camera_active:
            success, frame = cam.read()
            if not success or frame is None:
                break

            with camera_lock:
                latest_frame = frame.copy()

            # Draw face guide rectangle
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(30, 30))
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    finally:
        cam.release()
        capture_camera_active = False
        latest_frame = None


# ══════════════════════════════
# HELPER — Login Required
# ══════════════════════════════
def login_required(role=None):
    if 'user_id' not in session:
        return False
    if role and session.get('role') != role:
        return False
    return True


# ══════════════════════════════
# LOGIN PAGE
# ══════════════════════════════
@app.route('/')
def home():
    return render_template('login.html')


# ══════════════════════════════
# LOGIN LOGIC
# ══════════════════════════════
@app.route('/login', methods=['POST'])
def login():
    email = request.form.get('email')
    password = request.form.get('password')
    role = request.form.get('role')

    conn = get_db()
    # Fetch by email + role only (handle hashed vs plain passwords below)
    user = conn.execute(
        "SELECT * FROM users WHERE email = ? AND role = ?",
        (email, role)
    ).fetchone()

    if user:
        stored_pw = user['password']
        # Support both bcrypt-hashed (self-registered) and plain-text (admin-created)
        if stored_pw.startswith('pbkdf2:') or stored_pw.startswith('scrypt:'):
            pw_ok = check_password_hash(stored_pw, password)
        else:
            pw_ok = (stored_pw == password)

        if not pw_ok:
            conn.close()
            return render_template('login.html', error="Invalid Email or Password ❌")

        # Block student if their registration is still pending admin approval
        if user['role'] == 'student':
            pending_check = conn.execute(
                "SELECT id FROM student_registration_requests WHERE email = ? AND status = 'Pending' AND email_verified = 1",
                (email,)
            ).fetchone()
            if pending_check:
                conn.close()
                return render_template('login.html',
                    error="⏳ Your registration is pending admin approval. Please wait.")

        conn.close()
        session['user_id'] = user['id']
        session['role']    = user['role']
        session['name']    = user['name']

        if role == 'admin':
            return redirect(url_for('admin_dashboard'))
        elif role == 'faculty':
            return redirect(url_for('faculty_dashboard'))
        else:
            return redirect(url_for('student_dashboard'))
    else:
        conn.close()
        return render_template('login.html', error="Invalid Email or Password ❌")


# ══════════════════════════════
# LOGOUT
# ══════════════════════════════
@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('home'))


# ══════════════════════════════
# STUDENT DASHBOARD
# ══════════════════════════════
@app.route('/student_dashboard')
def student_dashboard():
    if not login_required(role='student'):
        return redirect(url_for('home'))

    user_id = session['user_id']
    conn = get_db()

    student = conn.execute(
        "SELECT s.*, u.name, u.email FROM students s JOIN users u ON s.user_id = u.id WHERE u.id = ?",
        (user_id,)
    ).fetchone()

    student_name = session.get('name', 'Student')

    attendance_count = 0
    face_registered = False
    today_status = None
    today_time = None
    student_data = {}

    if student:
        # Initialize default student data
        student_data = {
            'name': student['name'],
            'email': student['email'],
            'roll_number': student['roll_number'],
            'department': student['department'],
            'today_subject': None
        }

        attendance_count = conn.execute(
            "SELECT COUNT(*) as cnt FROM attendance WHERE student_id = ?",
            (student['id'],)
        ).fetchone()['cnt']

        # Check if face images exist (need at least 15 total images)
        student_face_dir = os.path.join(FACES_DIR, student['roll_number'])
        if os.path.exists(student_face_dir):
            face_count = len([f for f in os.listdir(student_face_dir) if f.endswith('.jpg')])
            face_registered = face_count >= 15

        # Check today's attendance status and session info
        today = date.today().isoformat()
        today_record = conn.execute('''
            SELECT a.time, a.status, s_sess.subject 
            FROM attendance a 
            LEFT JOIN sessions s_sess ON a.session_id = s_sess.id
            WHERE a.student_id = ? AND a.date = ?
            ORDER BY a.time DESC LIMIT 1
        ''', (student['id'], today)).fetchone()
        
        if today_record:
            today_status = today_record['status']
            today_time = today_record['time']
            # If there's a subject associated with the attendance, add it to session data
            if today_record['subject']:
                student_data['today_subject'] = today_record['subject']

    conn.close()

    return render_template('student_dashboard.html',
                           student_name=student_name,
                           attendance_count=attendance_count,
                           face_registered=face_registered,
                           today_status=today_status,
                           today_time=today_time,
                           student_data=student_data)


# ══════════════════════════════
# ADMIN DASHBOARD
# ══════════════════════════════
@app.route('/admin_dashboard')
def admin_dashboard():
    if not login_required(role='admin'):
        return redirect(url_for('home'))

    conn = get_db()

    total_students = conn.execute(
        "SELECT COUNT(*) as cnt FROM students").fetchone()['cnt']

    total_faculty = conn.execute(
        "SELECT COUNT(*) as cnt FROM faculty").fetchone()['cnt']

    today = date.today().isoformat()
    today_attendance = conn.execute(
        "SELECT COUNT(*) as cnt FROM attendance WHERE date = ?", (today,)
    ).fetchone()['cnt']

    pending_requests = conn.execute(
        "SELECT COUNT(*) as cnt FROM student_registration_requests WHERE status='Pending' AND email_verified=1"
    ).fetchone()['cnt']

    conn.close()

    return render_template('admin_dashboard.html',
                           total_students=total_students,
                           total_faculty=total_faculty,
                           today_attendance=today_attendance,
                           pending_requests=pending_requests)



# ══════════════════════════════
# LIVE ATTENDANCE CONTROL PANEL
# ══════════════════════════════
@app.route('/live_attendance')
def live_attendance():
    if not login_required(role='admin'):
        return redirect(url_for('home'))

    return render_template('live_attendance.html')


# ══════════════════════════════
# ADMIN STATS (JSON polling)
# ══════════════════════════════
@app.route('/admin/stats')
def admin_stats():
    if not login_required(role='admin'):
        return jsonify({}), 401
    conn = get_db()
    today = date.today().isoformat()
    data = {
        'total_students':  conn.execute("SELECT COUNT(*) as c FROM students").fetchone()['c'],
        'total_faculty':   conn.execute("SELECT COUNT(*) as c FROM faculty").fetchone()['c'],
        'today_attendance': conn.execute(
            "SELECT COUNT(*) as c FROM attendance WHERE date=?", (today,)).fetchone()['c'],
    }
    conn.close()
    return jsonify(data)


# ══════════════════════════════
# FACULTY MANAGEMENT (Admin)
# ══════════════════════════════
@app.route('/admin/faculty')
def admin_faculty_list():
    if not login_required(role='admin'):
        return redirect(url_for('home'))

    conn = get_db()
    rows = conn.execute('''
        SELECT u.id as uid, u.name, u.email,
               f.id as fid, f.faculty_id, f.subject, f.department
        FROM faculty f
        JOIN users u ON f.user_id = u.id
        ORDER BY u.name
    ''').fetchall()
    dept_count = conn.execute(
        "SELECT COUNT(DISTINCT department) as c FROM faculty"
    ).fetchone()['c']
    conn.close()

    return render_template('admin_faculty.html',
                           faculty_list=rows,
                           dept_count=dept_count)


@app.route('/admin/faculty/edit', methods=['POST'])
def admin_faculty_edit():
    if not login_required(role='admin'):
        return redirect(url_for('home'))

    user_id    = request.form.get('user_id', '').strip()
    name       = request.form.get('name', '').strip()
    faculty_id = request.form.get('faculty_id', '').strip()
    email      = request.form.get('email', '').strip()
    subject    = request.form.get('subject', '').strip()
    department = request.form.get('department', '').strip()

    if not all([user_id, name, faculty_id, email, subject, department]):
        flash("All fields are required ❌")
        return redirect(url_for('admin_faculty_list'))

    conn = get_db()

    # Check if email taken by someone else
    conflict_email = conn.execute(
        "SELECT id FROM users WHERE email=? AND id!=?", (email, user_id)
    ).fetchone()
    if conflict_email:
        conn.close()
        flash("Email already in use by another account ❌")
        return redirect(url_for('admin_faculty_list'))

    # Check if faculty_id taken by someone else
    conflict_fid = conn.execute(
        "SELECT f.id FROM faculty f WHERE f.faculty_id=? AND f.user_id!=?",
        (faculty_id, user_id)
    ).fetchone()
    if conflict_fid:
        conn.close()
        flash("Faculty ID already in use ❌")
        return redirect(url_for('admin_faculty_list'))

    conn.execute("UPDATE users SET name=?, email=? WHERE id=?",
                 (name, email, user_id))
    conn.execute("UPDATE faculty SET faculty_id=?, subject=?, department=? WHERE user_id=?",
                 (faculty_id, subject, department, user_id))
    conn.commit()
    conn.close()

    flash("Faculty updated successfully ✅")
    return redirect(url_for('admin_faculty_list'))


@app.route('/admin/faculty/delete/<int:uid>', methods=['POST'])
def admin_faculty_delete(uid):
    if not login_required(role='admin'):
        return redirect(url_for('home'))

    conn = get_db()
    try:
        # lookup faculty primary key so we can remove dependent rows
        faculty_row = conn.execute("SELECT id FROM faculty WHERE user_id = ?", (uid,)).fetchone()
        if faculty_row:
            fid = faculty_row['id']
            # delete any attendance tied to sessions for this faculty
            conn.execute("DELETE FROM attendance WHERE session_id IN (SELECT id FROM sessions WHERE faculty_id = ?)", (fid,))
            # delete sessions themselves
            conn.execute("DELETE FROM sessions WHERE faculty_id = ?", (fid,))
        # now remove faculty record + user account
        conn.execute("DELETE FROM faculty WHERE user_id=?", (uid,))
        conn.execute("DELETE FROM users WHERE id=?", (uid,))
        conn.commit()
    except sqlite3.IntegrityError:
        conn.rollback()
        flash("Unable to remove faculty – please clear their sessions first ⚠️")
        return redirect(url_for('admin_faculty_list'))
    except sqlite3.OperationalError:
        conn.rollback()
        flash("Database busy. Try again in a moment ⚠️")
        return redirect(url_for('admin_faculty_list'))
    finally:
        conn.close()

    # clear any cached attendance blocks for this faculty's students
    # (not strictly needed but keeps memory clean)
    blocked_today_copy = set(blocked_today)
    for key in blocked_today_copy:
        # keys are (student_id, session_id) – we don't know which belonged
        # to this faculty, so we simply leave them; they will expire next day
        pass

    flash("Faculty removed successfully ✅")
    return redirect(url_for('admin_faculty_list'))


# Removed to avoid duplication - logic handles in later part of file


# ══════════════════════════════
# VIDEO STREAM ROUTES
# ══════════════════════════════
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed_capture')
def video_feed_capture():
    return Response(generate_capture_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# ══════════════════════════════
# FACE CAPTURE (Student)
# ══════════════════════════════
@app.route('/capture_face')
def capture_face():
    if not login_required(role='student'):
        return redirect(url_for('home'))

    # Pass existing photo counts so the UI starts from there
    user_id = session['user_id']
    conn = get_db()
    student = conn.execute(
        "SELECT s.id, s.roll_number FROM students s WHERE s.user_id = ?", (user_id,)
    ).fetchone()
    conn.close()

    camera_count = 0
    upload_count = 0
    if student:
        student_dir = os.path.join(FACES_DIR, student['roll_number'])
        if os.path.exists(student_dir):
            for f in os.listdir(student_dir):
                if f.startswith('upload_') and f.endswith('.jpg'):
                    upload_count += 1
                elif f.endswith('.jpg'):
                    camera_count += 1

    return render_template('capture_face.html', camera_count=camera_count, upload_count=upload_count)


@app.route('/save_face', methods=['POST'])
def save_face():
    if not login_required(role='student'):
        return jsonify({'error': 'Not logged in'}), 401

    user_id = session['user_id']
    conn = get_db()
    student = conn.execute(
        "SELECT s.id, s.roll_number FROM students s WHERE s.user_id = ?", (user_id,)
    ).fetchone()
    conn.close()

    if not student:
        return jsonify({'error': 'Student not found'}), 404

    roll_number = student['roll_number']
    student_dir = os.path.join(FACES_DIR, roll_number)
    os.makedirs(student_dir, exist_ok=True)

    # Get current frame from camera
    with camera_lock:
        frame = latest_frame.copy() if latest_frame is not None else None

    if frame is None:
        return jsonify({'error': 'Camera not active. Please wait for camera to load.'}), 400

    # Detect face in frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(30, 30))

    if len(faces) == 0:
        return jsonify({'error': 'No face detected. Please position your face in the frame.'}), 400

    if len(faces) > 1:
        return jsonify({'error': 'Multiple faces detected. Only one person should be in the frame.'}), 400

    # Crop, resize, and apply CLAHE (much better than equalizeHist for uneven lighting)
    (x, y, w, h) = faces[0]
    face_crop = gray[y:y+h, x:x+w]
    face_resized = cv2.resize(face_crop, (100, 100))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    face_resized = clahe.apply(face_resized)

    # Count existing camera images (non-upload)
    existing = [f for f in os.listdir(student_dir) if f.endswith('.jpg') and not f.startswith('upload_')]
    count = len(existing) + 1

    if count > 30:
        return jsonify({'count': 30, 'done': True})

    filepath = os.path.join(student_dir, f'{count}.jpg')
    cv2.imwrite(filepath, face_resized)

    done = count >= 30
    return jsonify({'count': count, 'done': done})


# ══════════════════════════════
# UPLOAD FACE IMAGES (Student)
# ══════════════════════════════
@app.route('/upload_faces', methods=['POST'])
def upload_faces():
    if not login_required(role='student'):
        return jsonify({'error': 'Not logged in'}), 401

    user_id = session['user_id']
    conn = get_db()
    student = conn.execute(
        "SELECT s.id, s.roll_number FROM students s WHERE s.user_id = ?", (user_id,)
    ).fetchone()
    conn.close()

    if not student:
        return jsonify({'error': 'Student not found'}), 404

    roll_number = student['roll_number']
    student_dir = os.path.join(FACES_DIR, roll_number)
    os.makedirs(student_dir, exist_ok=True)

    # Count existing uploads
    existing_uploads = [f for f in os.listdir(student_dir) if f.startswith('upload_') and f.endswith('.jpg')]
    current_upload_count = len(existing_uploads)

    if current_upload_count >= 10:
        return jsonify({'error': 'Already uploaded 10 images', 'upload_count': 10, 'done': True})

    files = request.files.getlist('images')
    if not files:
        return jsonify({'error': 'No images selected'}), 400

    # Limit to remaining slots
    remaining = 10 - current_upload_count
    files = files[:remaining]

    saved = 0
    errors = []

    for i, file in enumerate(files):
        try:
            # Read image from upload
            file_bytes = np.frombuffer(file.read(), np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            if img is None:
                errors.append(f'Image {i+1}: Could not read file')
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(30, 30))

            if len(faces) == 0:
                errors.append(f'Image {i+1}: No face detected')
                continue

            if len(faces) > 1:
                errors.append(f'Image {i+1}: Multiple faces detected')
                continue

            # Crop, resize, CLAHE — same pipeline as camera capture
            (x, y, w, h) = faces[0]
            face_crop = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face_crop, (100, 100))
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            face_resized = clahe.apply(face_resized)

            current_upload_count += 1
            filepath = os.path.join(student_dir, f'upload_{current_upload_count}.jpg')
            cv2.imwrite(filepath, face_resized)
            saved += 1

        except Exception:
            errors.append(f'Image {i+1}: Processing error')
            continue

    done = current_upload_count >= 10
    return jsonify({
        'upload_count': current_upload_count,
        'saved': saved,
        'errors': errors,
        'done': done
    })


# ══════════════════════════════
# REGISTER STUDENT (Admin)
# ══════════════════════════════
@app.route('/register', methods=['GET', 'POST'])
def register_student():
    if not login_required(role='admin'):
        return redirect(url_for('home'))

    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')
        roll_number = request.form.get('roll_number')
        department = request.form.get('department')

        conn = get_db()

        existing = conn.execute(
            "SELECT id FROM users WHERE email = ?", (email,)
        ).fetchone()

        if existing:
            conn.close()
            return render_template('register_student.html',
                                   error="Email already registered ❌")

        existing_roll = conn.execute(
            "SELECT id FROM students WHERE roll_number = ?", (roll_number,)
        ).fetchone()

        if existing_roll:
            conn.close()
            return render_template('register_student.html',
                                   error="Roll number already exists ❌")

        cursor = conn.execute(
            "INSERT INTO users (email, password, role, name) VALUES (?, ?, 'student', ?)",
            (email, password, name)
        )
        user_id = cursor.lastrowid

        conn.execute(
            "INSERT INTO students (user_id, roll_number, department) VALUES (?, ?, ?)",
            (user_id, roll_number, department)
        )

        conn.commit()
        conn.close()

        flash("Student registered successfully ✅")
        return redirect(url_for('admin_dashboard'))

    return render_template('register_student.html')


# ══════════════════════════════
# REGISTER FACULTY (Admin)
# ══════════════════════════════
@app.route('/admin/register-faculty', methods=['GET', 'POST'])
def register_faculty():
    if not login_required(role='admin'):
        return redirect(url_for('home'))

    if request.method == 'POST':
        name       = request.form.get('name', '').strip()
        email      = request.form.get('email', '').strip()
        password   = request.form.get('password', '').strip()
        faculty_id = request.form.get('faculty_id', '').strip()
        subject    = request.form.get('subject', '').strip()
        department = request.form.get('department', '').strip()

        if not all([name, email, password, faculty_id, subject, department]):
            return render_template('register_faculty.html',
                                   error="All fields are required ❌")

        conn = get_db()

        existing_email = conn.execute(
            "SELECT id FROM users WHERE email = ?", (email,)
        ).fetchone()
        if existing_email:
            conn.close()
            return render_template('register_faculty.html',
                                   error="Email already registered ❌")

        existing_fid = conn.execute(
            "SELECT id FROM faculty WHERE faculty_id = ?", (faculty_id,)
        ).fetchone()
        if existing_fid:
            conn.close()
            return render_template('register_faculty.html',
                                   error="Faculty ID already exists ❌")

        cursor = conn.execute(
            "INSERT INTO users (email, password, role, name) VALUES (?, ?, 'faculty', ?)",
            (email, password, name)
        )
        user_id = cursor.lastrowid

        conn.execute(
            "INSERT INTO faculty (user_id, faculty_id, subject, department) VALUES (?, ?, ?, ?)",
            (user_id, faculty_id, subject, department)
        )

        conn.commit()
        conn.close()

        flash("Faculty registered successfully ✅")
        return redirect(url_for('admin_dashboard'))

    return render_template('register_faculty.html')


# ══════════════════════════════
# TRAIN FACE MODEL (Admin)
# ══════════════════════════════
def _run_training():
    """Background training thread."""
    global training_status
    try:
        faces_list = []
        labels_list = []
        label_map = {}
        current_label = 0

        conn = get_db()

        for roll_number_str in os.listdir(FACES_DIR):
            student_dir = os.path.join(FACES_DIR, roll_number_str)
            if not os.path.isdir(student_dir):
                continue

            # Look up student_id from roll_number
            student = conn.execute(
                "SELECT id FROM students WHERE roll_number = ?", (roll_number_str,)
            ).fetchone()
            if not student:
                continue

            student_id = student['id']
            label_map[current_label] = student_id

            for img_name in os.listdir(student_dir):
                if not img_name.endswith('.jpg'):
                    continue

                img_path = os.path.join(student_dir, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    # Images are saved as 100x100 + CLAHE from save_face/upload_faces
                    # DO NOT apply CLAHE again — just resize to ensure consistent dimensions
                    img = cv2.resize(img, (100, 100))
                    faces_list.append(img)
                    labels_list.append(current_label)

            current_label += 1

        conn.close()

        if len(faces_list) == 0:
            training_status = {'running': False, 'message': 'No face images found. Students need to register their faces first ❌', 'success': False}
            return

        try:
            recognizer = cv2.face.LBPHFaceRecognizer_create()
        except AttributeError:
            training_status = {'running': False, 'message': 'Missing opencv-contrib-python. Run: pip install opencv-contrib-python --break-system-packages ❌', 'success': False}
            return
        recognizer.train(faces_list, np.array(labels_list))
        recognizer.write(MODEL_PATH)
        np.save(LABELS_PATH, label_map)

        msg = f"Model trained successfully on {len(set(labels_list))} student(s) with {len(faces_list)} images ✅"
        training_status = {'running': False, 'message': msg, 'success': True}

    except Exception as e:
        training_status = {'running': False, 'message': f'Training failed: {str(e)} ❌', 'success': False}


@app.route('/train')
def train_model():
    if not login_required():
        return redirect(url_for('home'))
    if session.get('role') not in ('admin', 'faculty'):
        return redirect(url_for('home'))

    registered_students = 0
    total_images = 0

    if os.path.exists(FACES_DIR):
        for roll_number_str in os.listdir(FACES_DIR):
            student_dir = os.path.join(FACES_DIR, roll_number_str)
            if os.path.isdir(student_dir):
                imgs = [f for f in os.listdir(student_dir) if f.endswith('.jpg')]
                if len(imgs) > 0:
                    registered_students += 1
                    total_images += len(imgs)

    last_trained = "Never"
    model_exists = os.path.exists(MODEL_PATH)
    if model_exists:
        mod_time = os.path.getmtime(MODEL_PATH)
        trained_dt = datetime.fromtimestamp(mod_time)
        delta = datetime.now() - trained_dt
        if delta.days > 0:
            last_trained = f"{delta.days} day(s) ago"
        elif delta.seconds >= 3600:
            last_trained = f"{delta.seconds // 3600} hour(s) ago"
        elif delta.seconds >= 60:
            last_trained = f"{delta.seconds // 60} min(s) ago"
        else:
            last_trained = "Just now"

    return render_template('train_model.html',
                           registered_students=registered_students,
                           total_images=total_images,
                           last_trained=last_trained,
                           model_exists=model_exists)


@app.route('/train/start', methods=['POST'])
def train_start():
    global training_status
    if not login_required():
        return jsonify({'error': 'Unauthorized'}), 401
    if session.get('role') not in ('admin', 'faculty'):
        return jsonify({'error': 'Unauthorized'}), 403

    if training_status['running']:
        return jsonify({'error': 'Training already in progress'}), 400

    training_status = {'running': True, 'message': 'Training started...', 'success': False}
    thread = threading.Thread(target=_run_training, daemon=True)
    thread.start()
    return jsonify({'status': 'started'})


@app.route('/train/status')
def train_status():
    if not login_required():
        return jsonify({'error': 'Unauthorized'}), 401
    if session.get('role') not in ('admin', 'faculty'):
        return jsonify({'error': 'Unauthorized'}), 403
    return jsonify(training_status)


# ══════════════════════════════
# MARK ATTENDANCE — DISABLED (use Live Attendance)
# ══════════════════════════════
@app.route('/mark-attendance')
def mark_attendance():
    flash("Attendance is now marked automatically via classroom camera. Contact your teacher. 📷")
    return redirect(url_for('student_dashboard'))


# ══════════════════════════════
# UPDATE PROFILE (Student)
# ══════════════════════════════
@app.route('/change-password', methods=['POST'])
def change_password():
    if not login_required():
        return redirect(url_for('home'))
    if session.get('role') not in ('student', 'faculty'):
        return redirect(url_for('home'))

    user_id = session['user_id']
    new_password = request.form.get('new_password', '').strip()
    confirm_password = request.form.get('confirm_password', '').strip()
    redirect_target = request.form.get('redirect_to', 'dashboard')
    if redirect_target == 'profile':
        next_url = url_for('view_profile')
    elif session.get('role') == 'faculty':
        next_url = url_for('faculty_dashboard')
    else:
        next_url = url_for('student_dashboard')

    if not new_password or len(new_password) < 4:
        flash("Password must be at least 4 characters ❌")
        return redirect(next_url)

    if new_password != confirm_password:
        flash("Passwords do not match ❌")
        return redirect(next_url)

    conn = get_db()
    conn.execute("UPDATE users SET password = ? WHERE id = ?", (new_password, user_id))
    conn.commit()
    conn.close()

    flash("Password changed successfully ✅")
    return redirect(next_url)


# ══════════════════════════════
# VIEW PROFILE (Student)
# ══════════════════════════════
@app.route('/profile')
def view_profile():
    if not login_required():
        return redirect(url_for('home'))
    if session.get('role') not in ('student', 'faculty'):
        return redirect(url_for('home'))

    user_id = session['user_id']
    conn = get_db()

    if session.get('role') == 'faculty':
        user = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
        faculty = conn.execute('SELECT * FROM faculty WHERE user_id = ?', (user_id,)).fetchone()
        conn.close()
        student_data = {
            'name': user['name'] if user else '',
            'email': user['email'] if user else '',
            'roll_number': faculty['department'] if faculty else '',
            'department': faculty['subject'] if faculty else '',
        }
    else:
        student = conn.execute(
            'SELECT s.*, u.name, u.email FROM students s JOIN users u ON s.user_id = u.id WHERE u.id = ?',
            (user_id,)
        ).fetchone()
        conn.close()
        student_data = {}
        if student:
            student_data = {
                'name': student['name'],
                'email': student['email'],
                'roll_number': student['roll_number'],
                'department': student['department']
            }

    return render_template('student_profile.html', student_data=student_data)


# ══════════════════════════════
# VIEW ATTENDANCE (Student)
# ══════════════════════════════
@app.route('/view-attendance')
def view_attendance():
    if not login_required(role='student'):
        return redirect(url_for('home'))

    user_id = session['user_id']
    conn = get_db()
    student = conn.execute(
        "SELECT s.id FROM students s WHERE s.user_id = ?", (user_id,)
    ).fetchone()

    records = []
    if student:
        records = conn.execute('''
            SELECT a.date, a.time, a.status,
                   sess.subject,
                   u.name AS teacher_name
            FROM attendance a
            LEFT JOIN sessions sess ON a.session_id = sess.id
            LEFT JOIN faculty f ON sess.faculty_id = f.id
            LEFT JOIN users u ON f.user_id = u.id
            WHERE a.student_id = ?
            ORDER BY a.date DESC, a.time DESC
        ''', (student['id'],)).fetchall()

    conn.close()

    return render_template('view_attendance.html', records=records)


# ══════════════════════════════
# VIEW REPORTS (Admin)
# ══════════════════════════════
@app.route('/reports')
def reports():
    if not login_required():
        return redirect(url_for('home'))

    role = session.get('role')
    filter_date = request.args.get('date', date.today().isoformat())
    conn = get_db()

    if role == 'faculty':
        user_id = session['user_id']
        faculty = conn.execute("SELECT id FROM faculty WHERE user_id = ?", (user_id,)).fetchone()
        if faculty:
            records = conn.execute('''
                SELECT a.id, u.name, s.roll_number, s.department, a.date, a.time, a.status, sess.subject
                FROM attendance a
                JOIN students s ON a.student_id = s.id
                JOIN users u ON s.user_id = u.id
                JOIN sessions sess ON a.session_id = sess.id
                WHERE a.date = ? AND sess.faculty_id = ?
                ORDER BY a.time DESC
            ''', (filter_date, faculty['id'])).fetchall()
        else:
            records = []
    else:
        # Admin access - show all
        records = conn.execute('''
            SELECT a.id, u.name, s.roll_number, s.department, a.date, a.time, a.status, sess.subject
            FROM attendance a
            JOIN students s ON a.student_id = s.id
            JOIN users u ON s.user_id = u.id
            LEFT JOIN sessions sess ON a.session_id = sess.id
            WHERE a.date = ?
            ORDER BY a.time DESC
        ''', (filter_date,)).fetchall()

    conn.close()
    return render_template('reports.html',
                           records=records,
                           filter_date=filter_date)


# ══════════════════════════════
# DELETE ATTENDANCE (Admin)
# ══════════════════════════════
@app.route('/delete_attendance/<int:attendance_id>', methods=['POST'])
def delete_attendance(attendance_id):
    if not login_required():
        return redirect(url_for('home'))

    role = session.get('role')
    if role not in ('admin', 'faculty'):
        return redirect(url_for('home'))

    came_from = request.args.get('from', 'reports')
    conn = get_db()

    try:
        if role == 'faculty':
            user_id = session['user_id']
            faculty = conn.execute("SELECT id FROM faculty WHERE user_id = ?", (user_id,)).fetchone()
            if not faculty:
                conn.close()
                flash("Unauthorized ❌")
                return redirect(url_for('faculty_dashboard'))
            # Faculty can delete their own records OR manually-marked records (NULL session_id)
            record = conn.execute('''
                SELECT a.id, a.date FROM attendance a
                LEFT JOIN sessions s ON a.session_id = s.id
                WHERE a.id = ? AND (s.faculty_id = ? OR a.session_id IS NULL)
            ''', (attendance_id, faculty['id'])).fetchone()
        else:
            record = conn.execute("SELECT id, date FROM attendance WHERE id = ?", (attendance_id,)).fetchone()

        if not record:
            flash("Record not found or unauthorized ❌")
            conn.close()
            if came_from == 'today' and role == 'faculty':
                return redirect(url_for('faculty_present_today'))
            return redirect(url_for('reports'))

        record_date = record['date']
        att = conn.execute("SELECT student_id, session_id FROM attendance WHERE id = ?", (attendance_id,)).fetchone()
        student_id_to_unmark = att['student_id'] if att else None
        session_id_to_unmark = att['session_id'] if att else None

        conn.execute("DELETE FROM attendance WHERE id = ?", (attendance_id,))
        conn.commit()
    except sqlite3.OperationalError as e:
        conn.rollback()
        flash("Database error while deleting attendance. Please try again. ⚠️")
        if came_from == 'today' and role == 'faculty':
            return redirect(url_for('faculty_present_today'))
        return redirect(url_for('reports'))
    finally:
        conn.close()

    # maintain camera suppression for this student so it isn't re-added
    if student_id_to_unmark is not None:
        key = (student_id_to_unmark, session_id_to_unmark)
        # remove from marked set if present, then block
        marked_today.discard(key)
        blocked_today.add(key)

    flash("Attendance record removed ✅")
    if came_from == 'today':
        if role == 'faculty':
            return redirect(url_for('faculty_present_today'))
        return redirect(url_for('admin_students_today'))
    return redirect(url_for('reports', date=record_date))


# ══════════════════════════════
# MANUAL ATTENDANCE (Admin + Faculty)
# ══════════════════════════════
@app.route('/manual_attendance', methods=['GET', 'POST'])
def manual_attendance():
    # Direct session check (avoids any proxy issues with login_required wrapper)
    if 'user_id' not in session:
        return redirect(url_for('home'))

    user_role = session.get('role')
    if user_role not in ('admin', 'faculty'):
        return redirect(url_for('home'))

    conn = get_db()

    # For faculty: resolve their faculty record
    faculty_record = None
    if user_role == 'faculty':
        faculty_record = conn.execute(
            "SELECT id FROM faculty WHERE user_id = ?", (session['user_id'],)
        ).fetchone()
        if not faculty_record:
            conn.close()
            flash("Faculty profile not found ❌")
            return redirect(url_for('faculty_dashboard'))

    if request.method == 'POST':
        student_id = request.form.get('student_id')
        att_date = request.form.get('date')
        att_time = request.form.get('time') or datetime.now().strftime('%H:%M:%S')
        status = request.form.get('status', 'Present')

        if not student_id or not att_date:
            flash("Please select a student and date ❌")
            students = conn.execute('''
                SELECT s.id, u.name, s.roll_number, s.department
                FROM students s JOIN users u ON s.user_id = u.id
                ORDER BY u.name
            ''').fetchall()
            conn.close()
            return render_template('manual_attendance.html', students=students)

        # For faculty: link to their most recent session on that date (if any)
        linked_session_id = None
        if faculty_record:
            sess_row = conn.execute(
                "SELECT id FROM sessions WHERE faculty_id = ? AND date = ? ORDER BY id DESC LIMIT 1",
                (faculty_record['id'], att_date)
            ).fetchone()
            if sess_row:
                linked_session_id = sess_row['id']

        # Allow faculty to overwrite: if a record already exists for this student+date+session,
        # UPDATE it instead of blocking. Admin and faculty can always re-mark.
        existing = conn.execute(
            "SELECT id FROM attendance WHERE student_id = ? AND date = ? AND session_id IS ?",
            (student_id, att_date, linked_session_id)
        ).fetchone()

        if existing:
            # Update the existing record with the new status/time
            conn.execute(
                "UPDATE attendance SET time = ?, status = ? WHERE id = ?",
                (att_time, status, existing['id'])
            )
            conn.commit()
            conn.close()
            flash("Attendance updated successfully ✅")
        else:
            conn.execute(
                "INSERT INTO attendance (student_id, session_id, date, time, status) VALUES (?, ?, ?, ?, ?)",
                (student_id, linked_session_id, att_date, att_time, status)
            )
            conn.commit()
            conn.close()
            # Protect against the camera re-marking this student later in the same session
            key = (int(student_id), linked_session_id)
            marked_today.add(key)
            blocked_today.discard(key)
            flash("Attendance added successfully ✅")

        return redirect(url_for('manual_attendance'))

    # GET: show form
    students = conn.execute('''
        SELECT s.id, u.name, s.roll_number, s.department
        FROM students s JOIN users u ON s.user_id = u.id
        ORDER BY u.name
    ''').fetchall()
    conn.close()

    return render_template('manual_attendance.html', students=students)


# ══════════════════════════════
# MANAGE STUDENTS (Admin)
# ══════════════════════════════
@app.route('/admin/students')
def admin_students():
    if not login_required(role='admin'):
        return redirect(url_for('home'))

    conn = get_db()
    students = conn.execute('''
        SELECT s.id, u.name, u.email, s.roll_number, s.department
        FROM students s JOIN users u ON s.user_id = u.id
        ORDER BY u.name
    ''').fetchall()
    conn.close()

    return render_template('manage_students.html',
                           students=students,
                           mode='all',
                           title='All Students')


@app.route('/admin/students/today')
def admin_students_today():
    if not login_required(role='admin'):
        return redirect(url_for('home'))

    today = date.today().isoformat()
    conn = get_db()
    students = conn.execute('''
        SELECT s.id, u.name, u.email, s.roll_number, s.department, a.id as attendance_id, a.time, a.status
        FROM attendance a
        JOIN students s ON a.student_id = s.id
        JOIN users u ON s.user_id = u.id
        WHERE a.date = ?
        ORDER BY a.time DESC
    ''', (today,)).fetchall()
    conn.close()

    return render_template('manage_students.html',
                           students=students,
                           mode='today',
                           title="Today's Attendance")


@app.route('/admin/edit_student/<int:student_id>', methods=['GET', 'POST'])
def edit_student(student_id):
    if not login_required(role='admin'):
        return redirect(url_for('home'))

    conn = get_db()

    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        roll_number = request.form.get('roll_number')
        department = request.form.get('department')

        # Get user_id for this student
        student = conn.execute(
            "SELECT user_id FROM students WHERE id = ?", (student_id,)
        ).fetchone()

        if not student:
            conn.close()
            flash("Student not found ❌")
            return redirect(url_for('admin_students'))

        user_id = student['user_id']

        # Check email uniqueness (exclude current user)
        existing = conn.execute(
            "SELECT id FROM users WHERE email = ? AND id != ?", (email, user_id)
        ).fetchone()
        if existing:
            conn.close()
            flash("Email already in use by another user ❌")
            return redirect(url_for('edit_student', student_id=student_id))

        # Check roll number uniqueness (exclude current student)
        existing_roll = conn.execute(
            "SELECT id FROM students WHERE roll_number = ? AND id != ?", (roll_number, student_id)
        ).fetchone()
        if existing_roll:
            conn.close()
            flash("Roll number already in use ❌")
            return redirect(url_for('edit_student', student_id=student_id))

        conn.execute("UPDATE users SET name = ?, email = ? WHERE id = ?",
                     (name, email, user_id))
        conn.execute("UPDATE students SET roll_number = ?, department = ? WHERE id = ?",
                     (roll_number, department, student_id))
        conn.commit()
        conn.close()

        flash("Student updated successfully ✅")
        return redirect(url_for('admin_students'))

    # GET: load student data
    student = conn.execute('''
        SELECT s.id, u.name, u.email, s.roll_number, s.department
        FROM students s JOIN users u ON s.user_id = u.id
        WHERE s.id = ?
    ''', (student_id,)).fetchone()
    conn.close()

    if not student:
        flash("Student not found ❌")
        return redirect(url_for('admin_students'))

    return render_template('edit_student.html', student=student)


@app.route('/admin/delete_student/<int:student_id>', methods=['POST'])
def delete_student(student_id):
    if not login_required(role='admin'):
        return redirect(url_for('home'))

    conn = get_db()

    try:
        student = conn.execute(
            "SELECT user_id, roll_number FROM students WHERE id = ?", (student_id,)
        ).fetchone()

        if not student:
            flash("Student not found ❌")
            return redirect(url_for('admin_students'))

        user_id = student['user_id']
        roll_number = student['roll_number']

        # Delete attendance records
        conn.execute("DELETE FROM attendance WHERE student_id = ?", (student_id,))
        # Delete student record
        conn.execute("DELETE FROM students WHERE id = ?", (student_id,))
        # Delete user record
        conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
        conn.commit()
    except sqlite3.OperationalError:
        conn.rollback()
        flash("Database busy. Try again in a moment ⚠️")
        return redirect(url_for('admin_students'))
    finally:
        conn.close()

    # Remove face images folder
    face_dir = os.path.join(FACES_DIR, roll_number)
    if os.path.exists(face_dir):
        shutil.rmtree(face_dir)

    # cleanup the attendance caches
    to_remove = [k for k in marked_today if k[0] == student_id]
    for k in to_remove:
        marked_today.discard(k)
    to_block = [k for k in blocked_today if k[0] == student_id]
    for k in to_block:
        blocked_today.discard(k)

    flash("Student deleted successfully ✅")
    return redirect(url_for('admin_students'))


# ══════════════════════════════
# FACULTY DASHBOARD
# ══════════════════════════════
@app.route('/faculty_dashboard')
def faculty_dashboard():
    if not login_required(role='faculty'):
        return redirect(url_for('home'))

    user_id = session['user_id']
    conn = get_db()
    
    faculty = conn.execute(
        "SELECT * FROM faculty WHERE user_id = ?", (user_id,)
    ).fetchone()

    # If somehow no faculty profile found, redirect gracefully
    if not faculty:
        conn.close()
        flash("Faculty profile not found. Please contact the admin. ❌")
        return redirect(url_for('home'))

    active_session = conn.execute(
        "SELECT id, subject, start_time FROM sessions WHERE faculty_id = ? AND is_active = 1",
        (faculty['id'],)
    ).fetchone()

    total_students = conn.execute("SELECT COUNT(*) as cnt FROM students").fetchone()['cnt']
    
    # Always count distinct students present today across ALL sessions (not just current)
    today = date.today().isoformat()
    today_presence = conn.execute('''
        SELECT COUNT(DISTINCT a.student_id) as cnt
        FROM attendance a
        LEFT JOIN sessions s ON a.session_id = s.id
        WHERE a.date = ? AND (s.faculty_id = ? OR a.session_id IS NULL)
    ''', (today, faculty['id'])).fetchone()['cnt']

    conn.close()
    return render_template('faculty_dashboard.html', 
                           faculty=faculty, 
                           active_session=active_session,
                           total_students=total_students,
                           today_presence=today_presence)


@app.route('/faculty_live')
def faculty_live():
    """Full-page live attendance view for faculty."""
    if not login_required(role='faculty'):
        return redirect(url_for('home'))
    return render_template('faculty_live.html')


@app.route('/faculty/present_today')
def faculty_present_today():
    if not login_required(role='faculty'):
        return redirect(url_for('home'))

    user_id = session['user_id']
    today = date.today().isoformat()
    conn = get_db()
    
    faculty = conn.execute("SELECT id FROM faculty WHERE user_id = ?", (user_id,)).fetchone()
    
    if not faculty:
        conn.close()
        return redirect(url_for('faculty_dashboard'))

    students = conn.execute('''
        SELECT s.id, u.name, u.email, s.roll_number, s.department,
               a.time, a.status, a.id as attendance_id
        FROM attendance a
        JOIN students s ON a.student_id = s.id
        JOIN users u ON s.user_id = u.id
        LEFT JOIN sessions sess ON a.session_id = sess.id
        WHERE a.date = ? AND (sess.faculty_id = ? OR a.session_id IS NULL)
        GROUP BY a.student_id
        ORDER BY a.time DESC
    ''', (today, faculty['id'])).fetchall()
    
    conn.close()

    return render_template('manage_students.html',
                           students=students,
                           mode='today',
                           title="Today's Attendance")


@app.route('/start_session', methods=['POST'])
def start_session():
    if not login_required(role='faculty'):
        return jsonify({'error': 'Unauthorized'}), 401

    user_id = session['user_id']
    conn = get_db()
    faculty = conn.execute("SELECT id, subject FROM faculty WHERE user_id = ?", (user_id,)).fetchone()
    
    if not faculty:
        conn.close()
        return jsonify({'error': 'Faculty profile not found'}), 404

    # Close any existing active sessions
    conn.execute("UPDATE sessions SET is_active = 0, end_time = ? WHERE faculty_id = ? AND is_active = 1",
                 (datetime.now().strftime('%H:%M:%S'), faculty['id']))

    # Start new session
    today = date.today().isoformat()
    now = datetime.now().strftime('%H:%M:%S')
    conn.execute(
        "INSERT INTO sessions (faculty_id, subject, date, start_time, is_active) VALUES (?, ?, ?, ?, 1)",
        (faculty['id'], faculty['subject'], today, now)
    )
    conn.commit()
    conn.close()

    # NOTE: Do NOT clear marked_today here — the DB check prevents duplicates
    # and clearing would allow already-present students to be re-marked
    return jsonify({'success': True, 'redirect': '/faculty_live'})


@app.route('/stop_session', methods=['POST'])
def stop_session():
    if not login_required(role='faculty'):
        return jsonify({'error': 'Unauthorized'}), 401

    user_id = session['user_id']
    conn = get_db()
    faculty = conn.execute("SELECT id FROM faculty WHERE user_id = ?", (user_id,)).fetchone()

    if faculty:
        now = datetime.now().strftime('%H:%M:%S')
        conn.execute("UPDATE sessions SET is_active = 0, end_time = ? WHERE faculty_id = ? AND is_active = 1",
                     (now, faculty['id']))
        conn.commit()
    
    conn.close()
    return jsonify({'success': True})


# ══════════════════════════════
# RECOGNIZED TODAY (Updated for Faculty)
# ══════════════════════════════
@app.route('/recognized_today')
def recognized_today():
    if not login_required():
        return jsonify([])

    today = date.today().isoformat()
    conn = get_db()
    
    # Show all students marked present today (faculty or admin both get the same full list)
    records = conn.execute('''
        SELECT u.name, s.roll_number, MIN(a.time) as time, a.status
        FROM attendance a
        JOIN students s ON a.student_id = s.id
        JOIN users u ON s.user_id = u.id
        WHERE a.date = ?
        GROUP BY a.student_id
        ORDER BY MIN(a.time) DESC
    ''', (today,)).fetchall()

    conn.close()

    return jsonify([{
    'name': r['name'],
    'roll_number': r['roll_number'],
    'time': r['time'],
    'status': r['status'] if 'status' in r.keys() else 'Present'
} for r in records])

# (admin/stats route defined earlier in file)


def send_otp_email(to_email, otp_code, student_name):
    """Send OTP using Gmail SMTP."""
    try:
        html_body = f"""
        <html>
        <body style="font-family:Arial,sans-serif;background:#f3f4f6;padding:30px;">
          <div style="max-width:480px;margin:auto;background:white;border-radius:16px;overflow:hidden;">

            <div style="background:#2563eb;padding:20px;">
              <h2 style="color:white;margin:0;">Smart Attendance System</h2>
              <p style="color:#bfdbfe;margin:4px 0 0;">Email Verification</p>
            </div>

            <div style="padding:30px;">
              <p>Hello <b>{student_name}</b>,</p>
              <p>Your verification OTP is:</p>
              <h1 style="letter-spacing:10px;color:#1d4ed8;text-align:center;">{otp_code}</h1>
              <p>This OTP will expire in <b>10 minutes</b>.</p>
              <p style="color:#6b7280;font-size:13px;">If you did not request this, please ignore this email.</p>
            </div>

          </div>
        </body>
        </html>
        """

        msg = MIMEMultipart("alternative")
        msg["Subject"] = "Smart Attendance – Email Verification OTP"
        msg["From"]    = GMAIL_USER
        msg["To"]      = to_email
        msg.attach(MIMEText(html_body, "html"))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(GMAIL_USER, GMAIL_PASSWORD)
            server.sendmail(GMAIL_USER, to_email, msg.as_string())

        print(f"[EMAIL SENT] OTP delivered to {to_email}")
        return True

    except Exception as e:
        print(f"[EMAIL ERROR] Failed to send OTP to {to_email}: {e}")
        print(traceback.format_exc())
        return False
    
@app.route('/student-register', methods=['GET', 'POST'])
def student_self_register():
    if request.method == 'GET':
        return render_template('student_self_register.html')

    # ── POST: Validate & save ──
    full_name   = request.form.get('full_name', '').strip()
    reg_number  = request.form.get('reg_number', '').strip()
    email       = request.form.get('email', '').strip().lower()
    password    = request.form.get('password', '')
    confirm_pw  = request.form.get('confirm_password', '')
    department  = request.form.get('department', '').strip()

    def err(msg):
        return render_template('student_self_register.html', error=msg,
                               full_name=full_name, reg_number=reg_number,
                               email=email, department=department)

    # ── Server-side validations ──
    if not all([full_name, reg_number, email, password, confirm_pw, department]):
        return err("All fields are required.")

    if not email.endswith('@srmist.edu.in'):
        return err("Only SRM Institute email IDs (@srmist.edu.in) are allowed for registration.")

    import re
    if len(password) < 8:
        return err("Password must be at least 8 characters.")
    if not re.search(r'[A-Z]', password):
        return err("Password must contain at least one uppercase letter.")
    if not re.search(r'\d', password):
        return err("Password must contain at least one number.")
    if password != confirm_pw:
        return err("Passwords do not match.")

    conn = get_db()

    # Duplicate email check (users table + pending/approved requests)
    if conn.execute("SELECT id FROM users WHERE email = ?", (email,)).fetchone():
        conn.close()
        return err("This email is already registered. Please log in.")

    existing_req = conn.execute(
        "SELECT id, status, email_verified FROM student_registration_requests WHERE email = ? AND status != 'Rejected'",
        (email,)
    ).fetchone()
    if existing_req:
        # If the previous request was never OTP-verified (student abandoned the flow),
        # silently delete it so they can register fresh.
        if not existing_req['email_verified']:
            conn.execute("DELETE FROM student_registration_requests WHERE id = ?", (existing_req['id'],))
            conn.commit()
        elif existing_req['status'] == 'Pending':
            conn.close()
            return err("A registration request with this email is already pending admin approval.")
        else:
            conn.close()
            return err("This email has already been used for registration.")

    # Duplicate reg_number check
    if conn.execute("SELECT id FROM students WHERE roll_number = ?", (reg_number,)).fetchone():
        conn.close()
        return err("This registration number is already in the system.")

    existing_roll_req = conn.execute(
        "SELECT id, email_verified FROM student_registration_requests WHERE reg_number = ? AND status != 'Rejected'",
        (reg_number,)
    ).fetchone()
    if existing_roll_req:
        # Same logic: if old request was never verified, remove it so they can retry.
        if not existing_roll_req['email_verified']:
            conn.execute("DELETE FROM student_registration_requests WHERE id = ?", (existing_roll_req['id'],))
            conn.commit()
        else:
            conn.close()
            return err("A request with this registration number is already pending.")

    # ── Hash password & generate OTP ──
    hashed_pw = generate_password_hash(password)
    otp_code  = str(random.randint(100000, 999999))
    otp_expiry = (datetime.now() + timedelta(minutes=10)).isoformat()
    created_at = datetime.now().isoformat()

    cursor = conn.execute(
        """INSERT INTO student_registration_requests
           (full_name, reg_number, email, password, department, status,
            otp_code, otp_expiry, email_verified, created_at)
           VALUES (?, ?, ?, ?, ?, 'Pending', ?, ?, 0, ?)""",
        (full_name, reg_number, email, hashed_pw, department,
         otp_code, otp_expiry, created_at)
    )
    request_id = cursor.lastrowid
    conn.commit()
    conn.close()

    # ── Send OTP email ──
    email_sent = send_otp_email(email, otp_code, full_name)

    # Store request_id in session for the OTP page
    session['pending_reg_id'] = request_id
    session['pending_reg_email'] = email

    if email_sent:
        return render_template('student_otp_verify.html',
                               email=email,
                               request_id=request_id,
                               message=None)
    else:
        return render_template('student_otp_verify.html',
                               email=email,
                               request_id=request_id,
                               message=f"⚠️ Email could not be delivered. "
                                       f"(Dev fallback — your OTP is: {otp_code})")


@app.route('/student-verify-otp', methods=['POST'])
def student_verify_otp():
    # BUG FIX: Fall back to the hidden form field when the session is lost
    # (e.g. page refresh, server restart, cookie expiry).
    request_id = session.get('pending_reg_id') or request.form.get('request_id')
    reg_email  = session.get('pending_reg_email', '') or request.form.get('reg_email', '')
    entered_otp = request.form.get('otp', '').strip()

    if not request_id:
        flash("Session expired. Please register again.")
        return redirect(url_for('student_self_register'))

    try:
        request_id = int(request_id)
    except (ValueError, TypeError):
        flash("Invalid session. Please register again.")
        return redirect(url_for('student_self_register'))

    conn = get_db()
    req = conn.execute(
        "SELECT * FROM student_registration_requests WHERE id = ?", (request_id,)
    ).fetchone()

    if not req:
        conn.close()
        flash("Registration not found. Please try again.")
        return redirect(url_for('student_self_register'))

    reg_email = reg_email or req['email']

    # Check if already verified (double-submit protection)
    if req['email_verified']:
        conn.close()
        flash("✅ Your email is already verified. Please wait for admin approval.")
        return redirect(url_for('home'))

    # Check too many wrong attempts (locked out after 5)
    otp_attempts = req['otp_attempts'] if 'otp_attempts' in req.keys() else 0
    if otp_attempts >= 5:
        conn.close()
        return render_template('student_otp_verify.html',
                               email=reg_email,
                               request_id=request_id,
                               error="Too many incorrect attempts. Please click 'Resend OTP' to get a new one.")

    # Check expiry
    if not req['otp_expiry'] or datetime.now() > datetime.fromisoformat(req['otp_expiry']):
        conn.close()
        return render_template('student_otp_verify.html',
                               email=reg_email,
                               request_id=request_id,
                               error="OTP has expired. Please request a new one.")

    # Check OTP match
    if req['otp_code'] != entered_otp:
        conn.execute(
            "UPDATE student_registration_requests SET otp_attempts = COALESCE(otp_attempts, 0) + 1 WHERE id = ?",
            (request_id,)
        )
        conn.commit()
        conn.close()
        attempts_left = max(0, 4 - otp_attempts)
        return render_template('student_otp_verify.html',
                               email=reg_email,
                               request_id=request_id,
                               error=f"Incorrect OTP. {attempts_left} attempt(s) remaining.")

    # ── Mark email as verified, clear OTP ──
    conn.execute(
        "UPDATE student_registration_requests SET email_verified=1, otp_code=NULL, otp_expiry=NULL WHERE id=?",
        (request_id,)
    )
    conn.commit()
    conn.close()

    session.pop('pending_reg_id', None)
    session.pop('pending_reg_email', None)

    flash("✅ Email verified! Your registration request has been submitted. Please wait for admin approval.")
    return redirect(url_for('home'))


@app.route('/student-resend-otp', methods=['POST'])
def student_resend_otp():
    # BUG FIX: Fall back to the hidden form field when the session is lost.
    request_id = session.get('pending_reg_id') or request.form.get('request_id')
    reg_email  = session.get('pending_reg_email', '') or request.form.get('reg_email', '')

    if not request_id:
        flash("Session expired. Please register again.")
        return redirect(url_for('student_self_register'))

    try:
        request_id = int(request_id)
    except (ValueError, TypeError):
        flash("Invalid session. Please register again.")
        return redirect(url_for('student_self_register'))

    conn = get_db()
    req = conn.execute(
        "SELECT * FROM student_registration_requests WHERE id = ?", (request_id,)
    ).fetchone()

    if not req:
        conn.close()
        flash("Registration not found. Please register again.")
        return redirect(url_for('student_self_register'))

    reg_email = reg_email or req['email']

    new_otp    = str(random.randint(100000, 999999))
    new_expiry = (datetime.now() + timedelta(minutes=10)).isoformat()
    conn.execute(
        "UPDATE student_registration_requests SET otp_code=?, otp_expiry=?, otp_attempts=0 WHERE id=?",
        (new_otp, new_expiry, request_id)
    )
    conn.commit()
    conn.close()

    # Restore session values in case they were lost
    session['pending_reg_id'] = request_id
    session['pending_reg_email'] = reg_email

    email_sent = send_otp_email(reg_email, new_otp, req['full_name'])
    if email_sent:
        msg = "✅ A new OTP has been sent to your email."
    else:
        msg = f"⚠️ Email could not be delivered. (Dev fallback — your new OTP is: {new_otp})"

    return render_template('student_otp_verify.html',
                           email=reg_email,
                           request_id=request_id,
                           message=msg)


# ══════════════════════════════════════════════════════
# ADMIN — REGISTRATION REQUESTS
# ══════════════════════════════════════════════════════

@app.route('/admin/registration-requests')
def admin_registration_requests():
    if not login_required(role='admin'):
        return redirect(url_for('home'))

    search = request.args.get('search', '').strip()
    status_filter = request.args.get('status', 'All')

    conn = get_db()

    query = "SELECT * FROM student_registration_requests WHERE email_verified = 1"
    params = []

    if status_filter != 'All':
        query += " AND status = ?"
        params.append(status_filter)

    if search:
        query += " AND (full_name LIKE ? OR email LIKE ? OR reg_number LIKE ? OR department LIKE ?)"
        like = f"%{search}%"
        params.extend([like, like, like, like])

    query += " ORDER BY created_at DESC"
    requests_list = conn.execute(query, params).fetchall()

    pending_count = conn.execute(
        "SELECT COUNT(*) as cnt FROM student_registration_requests WHERE status='Pending' AND email_verified=1"
    ).fetchone()['cnt']

    conn.close()
    return render_template('admin_registration_requests.html',
                           requests=requests_list,
                           pending_count=pending_count,
                           search=search,
                           status_filter=status_filter)


@app.route('/admin/approve-request/<int:req_id>', methods=['POST'])
def admin_approve_request(req_id):
    if not login_required(role='admin'):
        return redirect(url_for('home'))

    conn = get_db()
    req = conn.execute(
        "SELECT * FROM student_registration_requests WHERE id = ? AND status = 'Pending'", (req_id,)
    ).fetchone()

    if not req:
        conn.close()
        flash("Request not found or already processed.")
        return redirect(url_for('admin_registration_requests'))

    # Insert into users table
    user_cursor = conn.execute(
        "INSERT INTO users (email, password, role, name) VALUES (?, ?, 'student', ?)",
        (req['email'], req['password'], req['full_name'])
    )
    user_id = user_cursor.lastrowid

    # Insert into students table
    conn.execute(
        "INSERT INTO students (user_id, roll_number, department) VALUES (?, ?, ?)",
        (user_id, req['reg_number'], req['department'])
    )

    # Mark request as Approved
    conn.execute(
        "UPDATE student_registration_requests SET status='Approved' WHERE id=?", (req_id,)
    )

    # Clean up any other unverified/duplicate requests for the same email or reg_number
    conn.execute(
        """DELETE FROM student_registration_requests
           WHERE id != ? AND email_verified = 0
           AND (email = ? OR reg_number = ?)""",
        (req_id, req['email'], req['reg_number'])
    )

    conn.commit()
    conn.close()

    flash(f"✅ {req['full_name']} approved successfully! Student can now log in.")
    return redirect(url_for('admin_registration_requests'))


@app.route('/admin/reject-request/<int:req_id>', methods=['POST'])
def admin_reject_request(req_id):
    if not login_required(role='admin'):
        return redirect(url_for('home'))

    conn = get_db()
    req = conn.execute(
        "SELECT full_name FROM student_registration_requests WHERE id = ?", (req_id,)
    ).fetchone()

    if req:
        conn.execute(
            "UPDATE student_registration_requests SET status='Rejected' WHERE id=?", (req_id,)
        )
        conn.commit()
        flash(f"❌ {req['full_name']}'s registration has been rejected.")
    else:
        flash("Request not found.")
    conn.close()
    return redirect(url_for('admin_registration_requests'))


# ══════════════════════════════════════════════════════
# ADMIN — DELETE REGISTRATION REQUEST
# Completely removes a stuck/abandoned request so the
# student can re-register with the same email/reg number.
# ══════════════════════════════════════════════════════
@app.route('/admin/delete-request/<int:req_id>', methods=['POST'])
def admin_delete_request(req_id):
    if not login_required(role='admin'):
        return redirect(url_for('home'))

    conn = get_db()
    req = conn.execute(
        "SELECT full_name, email FROM student_registration_requests WHERE id = ?", (req_id,)
    ).fetchone()

    if req:
        conn.execute("DELETE FROM student_registration_requests WHERE id = ?", (req_id,))
        conn.commit()
        flash(f"🗑️ Request for {req['full_name']} ({req['email']}) deleted. They can now re-register.")
    else:
        flash("Request not found.")
    conn.close()
    return redirect(url_for('admin_registration_requests'))


@app.route('/admin/approve-all-requests', methods=['POST'])
def admin_approve_all_requests():
    if not login_required(role='admin'):
        return redirect(url_for('home'))

    conn = get_db()
    pending = conn.execute(
        "SELECT * FROM student_registration_requests WHERE status='Pending' AND email_verified=1"
    ).fetchall()

    approved = 0
    for req in pending:
        # Skip if email already in users (edge case)
        existing = conn.execute("SELECT id FROM users WHERE email=?", (req['email'],)).fetchone()
        if existing:
            continue
        user_cur = conn.execute(
            "INSERT INTO users (email, password, role, name) VALUES (?, ?, 'student', ?)",
            (req['email'], req['password'], req['full_name'])
        )
        user_id = user_cur.lastrowid
        conn.execute(
            "INSERT INTO students (user_id, roll_number, department) VALUES (?, ?, ?)",
            (user_id, req['reg_number'], req['department'])
        )
        conn.execute(
            "UPDATE student_registration_requests SET status='Approved' WHERE id=?", (req['id'],)
        )
        approved += 1

    conn.commit()
    conn.close()

    flash(f"✅ {approved} registration(s) approved successfully!")
    return redirect(url_for('admin_registration_requests'))


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)