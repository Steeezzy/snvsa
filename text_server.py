import socket
import time
import random

# varied word stream — mix of concrete and abstract concepts
sentences = [
    "the cat sat on the mat",
    "dog runs fast across field",
    "sun rises over the mountain",
    "rain falls on the river",
    "fire burns bright in darkness",
    "wind moves through the trees",
    "water flows down the hill",
    "bird flies high in sky",
    "child learns from every mistake",
    "memory fades without repetition",
    "knowledge grows through experience",
    "pattern emerges from the noise",
]

words = []
for s in sentences:
    words.extend(s.split())

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind(("localhost", 9998))
server.listen(1)
print("text server ready on port 9998")

conn, _ = server.accept()
print("language agent connected")
f = conn.makefile('w')

idx = 0
while True:
    try:
        word = words[idx % len(words)]
        f.write(word + "\n")
        f.flush()
        idx += 1
        time.sleep(0.05)  # one word every 50ms
    except BrokenPipeError:
        break

server.close()
