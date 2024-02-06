import json
import random
import socket
import struct
import numpy as np

PORT, HOST_IP = 8081, '127.0.0.1'

def send_list(sock, data):
    serialized_data = json.dumps(data)  # Serialize the list as JSON

    data_size = len(serialized_data)
    sock.sendall(struct.pack('!I', data_size))  # Send the size of the data

    sock.sendall(serialized_data.encode())  # Send the JSON string over

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # Reuse TCP
    s.bind((HOST_IP, PORT))
    s.listen(1)
    print("Starting to listen...")
    conn, addr = s.accept()
    with conn:
        print('Connected by', addr)
        while True:
            tile_x = np.random.randint(0, 10, size=5).tolist()
            tile_y = np.random.randint(0, 10, size=5).tolist()
            tile_type = np.random.randint(0, 2, size=5).tolist()
            send_lst = list(zip(tile_type, tile_x, tile_y))
            send_lst = [tuple(x) for x in send_lst]
            send_list(conn, send_lst)