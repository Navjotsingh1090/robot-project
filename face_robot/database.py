import pickle

import mysql.connector
import numpy as np

from face_robot import config
from face_robot.names import normalize_name

db = None
cursor = None


def init_db():
    global db, cursor
    try:
        db = mysql.connector.connect(
            host=config.DB_HOST,
            user=config.DB_USER,
            password=config.DB_PASSWORD,
            database=config.DB_NAME,
        )
        cursor = db.cursor()
        print("✅ Database connected")
    except Exception as e:
        db = None
        cursor = None
        print(f"⚠️ Database unavailable: {e}")
        print("Running without persistence.")


def close_db():
    global db, cursor
    if db is not None:
        try:
            if cursor is not None:
                cursor.close()
        finally:
            db.close()
    db = None
    cursor = None


def save_user(name, encodings, face_file_hash=None):
    if cursor is None or db is None:
        print("⚠️ Skipping save: database not connected.")
        return

    hash_column_ok = True
    for index, enc in enumerate(encodings):
        data = pickle.dumps(enc)
        want_hash = bool(face_file_hash and index == 0)

        while True:
            try:
                if want_hash and hash_column_ok:
                    cursor.execute(
                        "INSERT INTO users (name, encoding, face_file_hash) VALUES (%s, %s, %s)",
                        (name, data, face_file_hash),
                    )
                else:
                    cursor.execute(
                        "INSERT INTO users (name, encoding) VALUES (%s, %s)",
                        (name, data),
                    )
                break
            except mysql.connector.Error as exc:
                if want_hash and hash_column_ok and getattr(exc, "errno", None) == 1054:
                    hash_column_ok = False
                    want_hash = False
                    continue
                raise

    db.commit()


def remove_user(name):
    if cursor is None or db is None:
        print("⚠️ Skipping delete: database not connected.")
        return 0

    clean_name = normalize_name(name)
    if clean_name is None:
        print("⚠️ Invalid user name.")
        return 0

    cursor.execute("DELETE FROM users WHERE LOWER(name) = LOWER(%s)", (clean_name,))
    deleted_rows = cursor.rowcount
    db.commit()
    return deleted_rows


def load_users():
    if cursor is None:
        return {}, {}

    try:
        cursor.execute("SELECT name, encoding FROM users")
    except mysql.connector.Error:
        return {}, {}

    rows = cursor.fetchall()
    user_samples = {}

    for row in rows:
        clean_name = normalize_name(row[0])
        if clean_name is None:
            continue
        user_samples.setdefault(clean_name, []).append(pickle.loads(row[1]))

    user_profiles = {
        name: np.mean(np.array(encodings), axis=0)
        for name, encodings in user_samples.items()
        if encodings
    }

    return user_profiles, user_samples
