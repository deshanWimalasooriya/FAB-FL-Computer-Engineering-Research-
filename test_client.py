
import Client
import threading
import time
import requests

def trigger():
    time.sleep(10)
    requests.post('http://127.0.0.1:8000/start_round')

threading.Thread(target=trigger, daemon=True).start()

Client.connect_device(N=3)
Client.start_process(N=3, alpha=0.1)
