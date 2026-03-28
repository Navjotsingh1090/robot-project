import cv2
import face_recognition
import numpy as np
import pyttsx3
import speech_recognition as sr
import mysql.connector
import pickle
from datetime import datetime
import time

# ---------------- CONFIG ----------------
CAMERA_INDEX = 5
MIC_INDEX = 5
TOLERANCE = 0.40   # 🔥 more strict = more accurate
COOLDOWN = 10

# ---------------- VOICE ----------------
engine = pyttsx3.init('espeak')
engine.setProperty('rate', 150)

def speak(text):
    print("Robot:", text)
    engine.say(text)
    engine.runAndWait()

speak("System ready")

# ---------------- DATABASE ----------------
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Robo@1234",
    database="robot"
)
cursor = db.cursor()

def save_user(name, encodings_list):
    for encoding in encodings_list:
        data = pickle.dumps(encoding)
        cursor.execute(
            "INSERT INTO users (name, encoding) VALUES (%s, %s)",
            (name, data)
        )
    db.commit()

def load_users():
    cursor.execute("SELECT name, encoding FROM users")
    rows = cursor.fetchall()

    names = []
    encodings = []

    for row in rows:
        names.append(row[0])
        encodings.append(pickle.loads(row[1]))

    return names, encodings

# ---------------- SPEECH INPUT ----------------
def get_name():
    r = sr.Recognizer()
    r.energy_threshold = 300

    try:
        with sr.Microphone(device_index=MIC_INDEX) as source:
            speak("What is your name")
            r.adjust_for_ambient_noise(source, duration=1)
            audio = r.listen(source, timeout=5, phrase_time_limit=3)

        name = r.recognize_google(audio)
        name = name.split()[0]

        speak(f"I heard {name}, say yes to confirm")

        with sr.Microphone(device_index=MIC_INDEX) as source:
            audio = r.listen(source, timeout=3)

        confirm = r.recognize_google(audio).lower()

        if "yes" in confirm:
            return name

    except:
        pass

    speak("Please type your name")
    return input("Enter name: ")

# ---------------- GREETING ----------------
def greet(name, count):
    hour = datetime.now().hour

    if count == 1:
        if hour < 12:
            speak(f"Good Morning {name}")
        else:
            speak(f"Good Evening {name}")
    elif count == 2:
        speak("Hello guys")
    else:
        speak("Hello everyone")

# ---------------- CAMERA ----------------
video = cv2.VideoCapture(5)

video.set(3, 640)
video.set(4, 480)

time.sleep(2)

if not video.isOpened():
    print("❌ Camera not working")
    exit()
else:
    print(f"✅ Camera index {CAMERA_INDEX} working")

# ---------------- LOAD USERS ----------------
known_names, known_encodings = load_users()

last_seen = {}
process_frame = True

# ---------------- MAIN LOOP ----------------
while True:
    ret, frame = video.read()
    if not ret:
        break

    small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

    if process_frame:
        faces = face_recognition.face_locations(rgb)
        encodings = face_recognition.face_encodings(rgb, faces)

        people_count = len(faces)

        for encoding in encodings:
            name = "Unknown"

            if len(known_encodings) > 0:
                distances = face_recognition.face_distance(known_encodings, encoding)
                best_index = np.argmin(distances)

                if distances[best_index] < TOLERANCE:
                    name = known_names[best_index]

            now = time.time()

            # ---------------- KNOWN PERSON ----------------
            if name != "Unknown":
                if name not in last_seen or now - last_seen[name] > COOLDOWN:
                    greet(name, people_count)
                    last_seen[name] = now

            # ---------------- NEW PERSON ----------------
            else:
                if "unknown" not in last_seen or now - last_seen.get("unknown", 0) > COOLDOWN:
                    speak("Hello")
                    name = get_name()

                    if name:
                        samples = []
                        speak("Look at camera")

                        for i in range(5):
                            ret, frame = video.read()
                            small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

                            faces = face_recognition.face_locations(rgb)
                            encs = face_recognition.face_encodings(rgb, faces)

                            if len(encs) > 0:
                                samples.append(encs[0])

                            time.sleep(0.5)

                        if len(samples) > 0:
                            save_user(name, samples)
                            speak(f"Nice to meet you {name}")

                            known_names.append(name)
                            known_encodings.extend(samples)

                            last_seen[name] = now
                            last_seen["unknown"] = now

    process_frame = not process_frame

    # ---------------- DRAW ----------------
    for (top, right, bottom, left) in faces:
        top *= 2
        right *= 2
        bottom *= 2
        left *= 2
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

    cv2.imshow("AI Robot Vision", frame)

    if cv2.waitKey(1) == 27:
        break

video.release()
cv2.destroyAllWindows()