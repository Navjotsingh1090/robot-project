from face_robot import database
from face_robot import face_storage


def remove_user_command(raw_name: str) -> int:
    database.init_db()
    deleted_rows = database.remove_user(raw_name)
    deleted_files = face_storage.delete_user_face_images(raw_name)
    if deleted_rows > 0:
        print(f"✅ Removed {deleted_rows} saved face record(s) for {raw_name}")
    else:
        print(f"ℹ️ No saved user found for {raw_name}")
    if deleted_files > 0:
        print(f"🗑️ Deleted {deleted_files} face image file(s) from {face_storage.config.FACE_FILES_DIR}")

    database.close_db()
    return 0
