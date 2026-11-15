# used to clear old streamed output embeddings and inbox files for testing
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INBOX_DIR = PROJECT_ROOT / "artifacts" / "inbox"
EMBEDDINGS_DIR = PROJECT_ROOT / "artifacts" / "embeddings"

def main():
    if not INBOX_DIR.exists():
        print(f"Inbox does not exist: {INBOX_DIR}")
    else:
        deleted = 0
        for path in INBOX_DIR.iterdir():
            if path.is_file():
                path.unlink()
                deleted += 1

        print(f"Deleted {deleted} file(s) from {INBOX_DIR}")

    if not EMBEDDINGS_DIR.exists():
        print(f"Embeddings directory does not exist: {EMBEDDINGS_DIR}")
    else:
        deleted = 0
        for path in EMBEDDINGS_DIR.iterdir():
            if path.is_dir():
                shutil.rmtree(path)
                deleted += 1

        print(f"Deleted {deleted} folder(s) from {EMBEDDINGS_DIR}")

if __name__ == "__main__":
    main()