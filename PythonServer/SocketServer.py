import socket

class SocketServer:
    '''
        Socket server for NodeMCU.
    '''
    def __init__(self, port):
        if type(port) != int:
            raise Exception(f'Error in SocketServer.__init__(self, port)\n\tArgument {port}\'s type have to int.')
        
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind(('0.0.0.0', port))
        self.sock.listen(0)
        self.client, self.client_addr = self.sock.accept()

    def __str__(self):
        return str(self.client_addr)

    def send(self, data):
        '''
            Send 1 byte data.
        '''
        if type(data) == int and data < 256:
            data = data.to_bytes(1, byteorder='big')
        self.client.send(data)

    def close(self):
        self.client.close()
        self.sock.close() 


if __name__ == "__main__":
    sock = SocketServer(55555)
    
    while True:
        bits = int(input("CMD: "), 2)
        print(bits)
    
        if bits == 0:
            break

        sock.send(bits)

    sock.close()
