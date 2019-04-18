import binascii
import logging
import multiprocessing
import select
import socket
import time


class VSTCPServer(object):

    def __init__(self, host, port, timeout=1, logfile=None):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.settimeout(timeout)
        self.conn = False
        self.lock = True
        self.logger = logging.getLogger('tcp_server')
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(logging.StreamHandler())
        if logfile:
            fh = logging.FileHandler(logfile)
            fh.setLevel(logging.INFO)
            self.logger.addHandler(fh)

    def handle(self):
        data = self.conn.recv(1)
        self.logger.debug(binascii.hexlify(data))

    def start(self):
        self.sock.bind((self.host, self.port))
        self.logger.debug("we got {}:{}".format(self.host, self.port))
        self.sock.listen(5)
        self.logger.debug("now listening")
        while not self.conn:
            try:
                self.conn, addr = self.sock.accept()
                self.conn.setblocking(True)
            except socket.timeout:
                self.logger.debug("accept timeout")
        self.sock.setblocking(True)
        self.logger.debug("Connected with {}:{}".format(addr[0], addr[1]))
        while self.lock:
            ready = select.select([self.conn], [], [], self.timeout)
            if ready[0]:
                self.handle()
            else:
                self.logger.debug("read timeout")

    def stop(self):
        self.lock = False
        self.sock.close()


if __name__ == '__main__':
    SERVER = VSTCPServer("127.0.0.1", 1935)
    PROCESS = multiprocessing.Process(target=SERVER.start)
    PROCESS.start()
    time.sleep(5)
    SERVER.stop()
    PROCESS.join()

