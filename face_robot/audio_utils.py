import contextlib
import os
import sys

from face_robot import config


@contextlib.contextmanager
def suppress_native_stderr(enabled=True):
    if not enabled:
        yield
        return

    try:
        stderr_fd = sys.stderr.fileno()
    except (AttributeError, ValueError, OSError):
        yield
        return

    saved_stderr_fd = os.dup(stderr_fd)
    try:
        with open(os.devnull, "w", encoding="utf-8") as null_stream:
            os.dup2(null_stream.fileno(), stderr_fd)
            yield
    finally:
        os.dup2(saved_stderr_fd, stderr_fd)
        os.close(saved_stderr_fd)


@contextlib.contextmanager
def suppress_stderr():
    with open(os.devnull, "w", encoding="utf-8") as null_stream:
        with contextlib.redirect_stderr(null_stream):
            yield


def audio_probe_context():
    if not config.SUPPRESS_AUDIO_BACKEND_NOISE:
        return contextlib.nullcontext()
    stack = contextlib.ExitStack()
    stack.enter_context(suppress_native_stderr(True))
    stack.enter_context(suppress_stderr())
    return stack
