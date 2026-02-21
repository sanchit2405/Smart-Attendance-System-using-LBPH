from flask import Flask, render_template, request, redirect, url_for, session, flash, Response, jsonify
from database import get_db, init_db
from datetime import date, datetime
import cv2
import numpy as np
import os
import shutil
import threading

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

# Global frame for capture
latest_frame = None
capture_camera_active = False

# Training state
training_status = {'running': False, 'message': '', 'success': False}


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

    with camera_lock:
        cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cam.isOpened():
        cam.release()
        return

    recognizer = None
    label_map = {}

    if os.path.exists(MODEL_PATH) and os.path.exists(LABELS_PATH):
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(MODEL_PATH)
        label_map = np.load(LABELS_PATH, allow_pickle=True).item()

    student_name_cache = {}
    conn = get_db()
    rows = conn.execute(
        "SELECT s.id, u.name FROM students s JOIN users u ON s.user_id = u.id"
    ).fetchall()
    conn.close()

    for row in rows:
        student_name_cache[row['id']] = row['name']

    BUFFER_SIZE = 25
    CONFIDENCE_THRESH = 100
    AVG_CONF_THRESH = 100
    VOTE_MAJORITY = 0.5
    IOU_MATCH_THRESH = 0.35
    MAX_MISSING_FRAMES = 10
    MIN_FRAMES_TO_DECIDE = 20
    RETRY_AFTER_UNKNOWN = 30

    clahe = cv2.createCLAHE(2.0, (8, 8))

    next_track_id = 0
    tracks = {}
    marked_today = set()

    try:
        while True:
            success, frame = cam.read()
            if not success or frame is None:
                break

            h_frame, w_frame = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            raw_faces = face_cascade.detectMultiScale(
                gray, 1.1, 6, minSize=(60, 60)
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
                        'stable_count': 0
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

                if confirmed_id and confirmed_id not in marked_today:
                    today = date.today().isoformat()
                    now = datetime.now().strftime('%H:%M:%S')
                    conn = get_db()
                    existing = conn.execute(
                        "SELECT id FROM attendance WHERE student_id=? AND date=?",
                        (confirmed_id, today)
                    ).fetchone()
                    if not existing:
                        conn.execute(
                            "INSERT INTO attendance (student_id, date, time, status) VALUES (?, ?, ?, 'Present')",
                            (confirmed_id, today, now)
                        )
                        conn.commit()
                    conn.close()
                    marked_today.add(confirmed_id)

                if 'Scanning' in display_name:
                    color = (0, 165, 255)
                elif display_name == 'Unknown':
                    color = (0,0,255)
                else:
                    color = (0,200,0)

                cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
                cv2.putText(frame,display_name,(x,y-10),
                            cv2.FONT_HERSHEY_SIMPLEX,0.62,color,2)

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

    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

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
    user = conn.execute(
        "SELECT * FROM users WHERE email = ? AND password = ? AND role = ?",
        (email, password, role)
    ).fetchone()
    conn.close()

    if user:
        session['user_id'] = user['id']
        session['role'] = user['role']
        session['name'] = user['name']

        if role == 'admin':
            return redirect(url_for('admin_dashboard'))
        else:
            return redirect(url_for('student_dashboard'))
    else:
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
        attendance_count = conn.execute(
            "SELECT COUNT(*) as cnt FROM attendance WHERE student_id = ?",
            (student['id'],)
        ).fetchone()['cnt']

        # Check if face images exist (need at least 15 total images)
        student_face_dir = os.path.join(FACES_DIR, student['roll_number'])
        if os.path.exists(student_face_dir):
            face_count = len([f for f in os.listdir(student_face_dir) if f.endswith('.jpg')])
            face_registered = face_count >= 15

        # Check today's attendance status
        today = date.today().isoformat()
        today_record = conn.execute(
            "SELECT time, status FROM attendance WHERE student_id = ? AND date = ?",
            (student['id'], today)
        ).fetchone()
        if today_record:
            today_status = today_record['status']
            today_time = today_record['time']

        # Student profile data
        student_data = {
            'name': student['name'],
            'email': student['email'],
            'roll_number': student['roll_number'],
            'department': student['department']
        }

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

    today = date.today().isoformat()
    today_attendance = conn.execute(
        "SELECT COUNT(*) as cnt FROM attendance WHERE date = ?", (today,)
    ).fetchone()['cnt']

    conn.close()

    return render_template('admin_dashboard.html',
                           total_students=total_students,
                           today_attendance=today_attendance)


# ══════════════════════════════
# LIVE ATTENDANCE CONTROL PANEL
# ══════════════════════════════
@app.route('/live_attendance')
def live_attendance():
    if not login_required(role='admin'):
        return redirect(url_for('home'))

    return render_template('live_attendance.html')


# ══════════════════════════════
# RECOGNIZED TODAY (for live auto-refresh)
# ══════════════════════════════
@app.route('/recognized_today')
def recognized_today():
    if not login_required(role='admin'):
        return jsonify([])

    today = date.today().isoformat()
    conn = get_db()
    records = conn.execute('''
        SELECT u.name, s.roll_number, a.time
        FROM attendance a
        JOIN students s ON a.student_id = s.id
        JOIN users u ON s.user_id = u.id
        WHERE a.date = ?
        ORDER BY a.time DESC
    ''', (today,)).fetchall()
    conn.close()

    return jsonify([{'name': r['name'], 'roll_number': r['roll_number'], 'time': r['time']} for r in records])


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

        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.train(faces_list, np.array(labels_list))
        recognizer.write(MODEL_PATH)
        np.save(LABELS_PATH, label_map)

        msg = f"Model trained successfully on {len(set(labels_list))} student(s) with {len(faces_list)} images ✅"
        training_status = {'running': False, 'message': msg, 'success': True}

    except Exception as e:
        training_status = {'running': False, 'message': f'Training failed: {str(e)} ❌', 'success': False}


@app.route('/train')
def train_model():
    if not login_required(role='admin'):
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
    if not login_required(role='admin'):
        return jsonify({'error': 'Not logged in'}), 401

    if training_status['running']:
        return jsonify({'error': 'Training already in progress'}), 400

    training_status = {'running': True, 'message': 'Training started...', 'success': False}
    thread = threading.Thread(target=_run_training, daemon=True)
    thread.start()
    return jsonify({'status': 'started'})


@app.route('/train/status')
def train_status():
    if not login_required(role='admin'):
        return jsonify({'error': 'Not logged in'}), 401
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
    if not login_required(role='student'):
        return redirect(url_for('home'))

    user_id = session['user_id']
    new_password = request.form.get('new_password', '').strip()
    confirm_password = request.form.get('confirm_password', '').strip()
    # Check if we should redirect back to profile or dashboard
    redirect_target = request.form.get('redirect_to', 'dashboard')
    
    # Determine the target URL
    next_url = url_for('view_profile') if redirect_target == 'profile' else url_for('student_dashboard')

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
    if not login_required(role='student'):
        return redirect(url_for('home'))

    user_id = session['user_id']
    conn = get_db()
    student = conn.execute(
        "SELECT s.*, u.name, u.email FROM students s JOIN users u ON s.user_id = u.id WHERE u.id = ?",
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
            SELECT a.date, a.time, a.status
            FROM attendance a
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
    if not login_required(role='admin'):
        return redirect(url_for('home'))

    filter_date = request.args.get('date', date.today().isoformat())

    conn = get_db()
    records = conn.execute('''
        SELECT a.id, u.name, s.roll_number, s.department, a.date, a.time, a.status
        FROM attendance a
        JOIN students s ON a.student_id = s.id
        JOIN users u ON s.user_id = u.id
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
    if not login_required(role='admin'):
        return redirect(url_for('home'))

    # Optional: caller can pass ?from=today to redirect back to today's attendance view
    came_from = request.args.get('from', 'reports')

    conn = get_db()
    record = conn.execute("SELECT date FROM attendance WHERE id = ?", (attendance_id,)).fetchone()

    if not record:
        conn.close()
        flash("Record not found ❌")
        return redirect(url_for('reports'))

    record_date = record['date']
    conn.execute("DELETE FROM attendance WHERE id = ?", (attendance_id,))
    conn.commit()
    conn.close()

    flash("Attendance record removed ✅")
    if came_from == 'today':
        return redirect(url_for('admin_students_today'))
    return redirect(url_for('reports', date=record_date))


# ══════════════════════════════
# MANUAL ATTENDANCE (Admin)
# ══════════════════════════════
@app.route('/manual_attendance', methods=['GET', 'POST'])
def manual_attendance():
    if not login_required(role='admin'):
        return redirect(url_for('home'))

    conn = get_db()

    if request.method == 'POST':
        student_id = request.form.get('student_id')
        att_date = request.form.get('date')
        att_time = request.form.get('time', datetime.now().strftime('%H:%M:%S'))
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

        # Check if already marked
        existing = conn.execute(
            "SELECT id FROM attendance WHERE student_id = ? AND date = ?",
            (student_id, att_date)
        ).fetchone()

        if existing:
            flash("Attendance already exists for this student on this date ⚠️")
            students = conn.execute('''
                SELECT s.id, u.name, s.roll_number, s.department
                FROM students s JOIN users u ON s.user_id = u.id
                ORDER BY u.name
            ''').fetchall()
            conn.close()
            return render_template('manual_attendance.html', students=students)

        conn.execute(
            "INSERT INTO attendance (student_id, date, time, status) VALUES (?, ?, ?, ?)",
            (student_id, att_date, att_time, status)
        )
        conn.commit()
        conn.close()

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

    student = conn.execute(
        "SELECT user_id, roll_number FROM students WHERE id = ?", (student_id,)
    ).fetchone()

    if not student:
        conn.close()
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
    conn.close()

    # Remove face images folder
    face_dir = os.path.join(FACES_DIR, roll_number)
    if os.path.exists(face_dir):
        shutil.rmtree(face_dir)

    flash("Student deleted successfully ✅")
    return redirect(url_for('admin_students'))


# ══════════════════════════════
# RUN APP
# ══════════════════════════════
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)