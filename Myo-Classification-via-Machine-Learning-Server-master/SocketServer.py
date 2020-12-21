from __future__ import print_function
import socket
from config import *

class SocketServer:
    def __init__(self, port):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind(('0.0.0.0', port))
        self.sock.listen(0)
        self.conn, self.addr = self.sock.accept()

        print("SOCK:", self.conn)

    def send(self, predicted_value):
        print("MSG: ", encode_string[predicted_value])
        self.conn.sendall(encode_string[predicted_value].encode())

encode_table = [
    0b00001,    # 0: Thumb open
    0b00010,    # 1: Index finger open
    0b00100,    # 2: Middle finger open
    0b01000,    # 3: Ring finger open
    0b10000,    # 4: Pinky finger open
    0b11111,    # 5: Five fingers open
    0b00000     # 6: All fingers closed
]
encode_string = [
    '00001',
    '00010',
    '00100',
    '01000',
    '10000',
    '11111',
    '00000'
]