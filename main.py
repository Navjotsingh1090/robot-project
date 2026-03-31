"""Entry point: run the vision loop or `python main.py remove-user <name>`."""

import sys

from face_robot import config

_DELETE_MODE = len(sys.argv) >= 3 and sys.argv[1] == "remove-user"


def main():
    if _DELETE_MODE:
        from face_robot.cli import remove_user_command

        raise SystemExit(remove_user_command(sys.argv[2]))

    from face_robot.audio_utils import suppress_native_stderr

    with suppress_native_stderr(config.SUPPRESS_AUDIO_BACKEND_NOISE):
        try:
            import face_recognition  # noqa: F401
        except ImportError:
            print("❌ Missing dependency: face_recognition")
            print("Install it in your active environment and rerun.")
            raise SystemExit(1)
        from face_robot.runner import run

        try:
            run()
        except KeyboardInterrupt:
            # Runner also handles this, but keep entrypoint quiet in case it bubbles up.
            raise SystemExit(0)


if __name__ == "__main__":
    main()
