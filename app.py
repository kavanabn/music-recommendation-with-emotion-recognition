from flask import Flask, render_template, Response, jsonify, request, session, redirect, url_for
import sqlite3
import hashlib
from camera import get_frame, show_text, emotion_dict, music_rec
import pandas as pd
from googleapiclient.discovery import build
import cv2

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

# YouTube API setup
YOUTUBE_API_KEY = 'AIzaSyBtN74CIAGsXZHzXVUCAb1yYTvly-8axkU'  # Replace with your actual API key

def get_video_id_for_emotion(emotion):
    youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
    query = f"{emotion} mood music"
    request = youtube.search().list(
        q=query,
        part="snippet",
        maxResults=1,
        type="video"
    )
    response = request.execute()
    if response["items"]:
        return response["items"][0]["id"]["videoId"]
    return None

def get_video_url_for_emotion(emotion):
    # Return a YouTube search URL for the emotion
    query = f"{emotion} mood music"
    return f"https://www.youtube.com/results?search_query={query.replace(' ', '+')}"

# Initialize camera and music data
headings = ("Name", "Album", "Artist")
df1 = music_rec()
df1 = df1.head(15)

# --- Database Setup ---
def init_db():
    conn = sqlite3.connect('music_app.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS mood_entries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        emotion TEXT NOT NULL,
        confidence REAL,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )''')
    conn.commit()
    conn.close()

# --- Auth Helpers ---
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password, password_hash):
    return hash_password(password) == password_hash

# --- Camera and Music Logic ---
# Remove VideoCamera class usage
def gen():
    while True:
        frame, df1, emotion = get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# --- Routes ---
@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('dashboard.html', headings=headings, data=df1)

@app.route('/video_feed')
def video_feed():
    try:
        return Response(gen(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        import numpy as np
        img = np.zeros((400, 600, 3), dtype=np.uint8)
        cv2.putText(img, 'Camera Error', (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 3, cv2.LINE_AA)
        _, jpeg = cv2.imencode('.jpg', img)
        return Response(jpeg.tobytes(), mimetype='image/jpeg')

@app.route('/get_recommendations')
def get_recommendations():
    from camera import show_text, emotion_dict, music_rec  # Import the current emotion index, dict, and music_rec
    emotion_index = show_text[0]  # Get the latest detected emotion index
    emotion = emotion_dict.get(emotion_index, "Happy")
    video_url = get_video_url_for_emotion(emotion)
    music_df = music_rec()  # Always get the latest music recommendations for the current emotion
    music_list = music_df.to_dict(orient='records')
    return jsonify({
        'emotion': emotion,
        'music': music_list,
        'video_url': video_url
    })

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = sqlite3.connect('music_app.db')
        cursor = conn.cursor()
        cursor.execute('SELECT id, password_hash FROM users WHERE username = ?', (username,))
        user = cursor.fetchone()
        conn.close()
        if user and verify_password(password, user[1]):
            session['user_id'] = user[0]
            session['username'] = username
            return redirect(url_for('dashboard'))
        else:
            return render_template('login.html', error='Invalid credentials')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = sqlite3.connect('music_app.db')
        cursor = conn.cursor()
        try:
            cursor.execute('INSERT INTO users (username, password_hash) VALUES (?, ?)', (username, hash_password(password)))
            conn.commit()
            user_id = cursor.lastrowid
            session['user_id'] = user_id
            session['username'] = username
            conn.close()
            return redirect(url_for('dashboard'))
        except sqlite3.IntegrityError:
            conn.close()
            return render_template('register.html', error='Username already exists')
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/mood_calendar')
def mood_calendar():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('mood_calendar.html')

@app.route('/analytics')
def analytics():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('analytics.html')

@app.route('/profile')
def profile():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('profile.html')

@app.route('/set_emotion', methods=['POST'])
def set_emotion():
    from camera import emotion_dict, show_text
    emotion_name = request.json.get('emotion')
    # Find the index for the emotion name (case-insensitive)
    emotion_index = None
    for idx, name in emotion_dict.items():
        if name.lower() == emotion_name.lower():
            emotion_index = idx
            break
    if emotion_index is not None:
        show_text[0] = emotion_index
        return jsonify({'success': True, 'emotion': emotion_name})
    else:
        return jsonify({'success': False, 'error': 'Invalid emotion'}), 400

if __name__ == '__main__':
    init_db()
    app.run(debug=True)
