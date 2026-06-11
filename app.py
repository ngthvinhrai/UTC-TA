import os, shutil, atexit
from src.frontend.gui import Interface
from dotenv import load_dotenv

load_dotenv()

KSTORE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "src", "backend", "store", "kstore"))

def cleanup_kstore():
    if os.path.exists(KSTORE_DIR):
        shutil.rmtree(KSTORE_DIR)

if __name__ == "__main__":
    atexit.register(cleanup_kstore)
    inf = Interface()
    inf.initialize()