import subprocess


def git_revision() -> str | None:
    """Short git commit of the checkout the process runs from, or None outside a git checkout."""
    try:
        result = subprocess.run(["git", "rev-parse", "--short", "HEAD"], check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None
