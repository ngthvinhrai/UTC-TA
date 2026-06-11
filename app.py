import os, shutil, atexit
from src.frontend.gui import Interface
from dotenv import load_dotenv
import requests

load_dotenv()

KSTORE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "src", "backend", "store", "kstore"))
FASTAPI_URL = os.getenv("FASTAPI_URL")

def cleanup_kstore():
    if os.path.exists(KSTORE_DIR):
        shutil.rmtree(KSTORE_DIR)
    requests.get(
        url=f"{FASTAPI_URL}/clear_vector_store"
    )

if __name__ == "__main__":
    atexit.register(cleanup_kstore)
    inf = Interface()
    inf.initialize()
    inf.run()