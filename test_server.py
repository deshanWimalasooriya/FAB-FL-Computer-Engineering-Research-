import subprocess
import time
import requests

server = subprocess.Popen(['python', 'Server.py'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
time.sleep(5)

client_script = """
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
"""
with open("test_client.py", "w") as f:
    f.write(client_script)

client = subprocess.Popen(['python', 'test_client.py'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
try:
    stdout, stderr = client.communicate(timeout=45)
    print("CLIENT STDOUT:", stdout)
    print("CLIENT STDERR:", stderr)
except Exception as e:
    print("Client timeout/error", e)

server.terminate()
stdout, stderr = server.communicate()
print('SERVER STDOUT:')
print(stdout)
print('SERVER STDERR:')
print(stderr)
