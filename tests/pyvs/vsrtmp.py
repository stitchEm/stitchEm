#!/usr/bin/env python

"""
Stupid RTMP server that does a correct handshake and then sends a predefined
sequence of RTMP frame to establish a connexion with the client. Then data
are just thrown away

http://wwwimages.adobe.com/content/dam/Adobe/en/devnet/rtmp/pdf/rtmp_specification_1.0.pdf
"""

import binascii
import logging
import random
import time
from optparse import OptionParser

import vstcp

VERBOSE = False
C0 = binascii.unhexlify("03")
S0 = C0
ZERO = int(time.time() * 1000)
ETIME = hex(int(time.time()))[2:]
RANDOM = ""

for _ in xrange(1536 - 8):
    r = hex(random.randint(0, 255))[2:]
    if len(r) == 1:
        r = "0" + r
    RANDOM += r

class RTMPPacket(object):

    def __str__(self):
        return self.header + self.body

    def __add__(self, other):
        return str(self) + str(other)

    def __radd__(self, other):
        return other + str(self)

    @property
    def body(self):
        raise Exception("implement in subclass")

    @property
    def length(self):
        return len(self.body) / 2

    @property
    def basic_header(self):
        def first_byte(rest):
            return "{0:#0{1}x}".format(self.hfmt * 2 ** 6 + rest,
                                       1 * 2 + 2)[2:]
        if self.chunk_streamid < 64:
            return first_byte(self.chunk_streamid)
        elif self.chunk_streamid < 320:
            return first_byte(0) + "{0:#0{1}x}".format(self.chunk_streamid,
                                                       1 * 2 + 2)[2:]
        elif self.chunk_streamid < 65600:
            return first_byte(2 ** 6 - 1) + "{0:#0{1}x}".format(
                self.chunk_streamid, 2 * 2 + 2)[2:]
        else:
            raise Exception("Stream ID not supported")

    @property
    def message_header(self):
        if self.hfmt == 0:
            # 11 bytes long
            return "{0:#0{1}x}".format(
                16777210    * 16 ** (8 * 2) +\
                self.length * 16 ** (5 * 2) +\
                self.typeid * 16 ** (4 * 2) +\
                self.streamid,
                11 * 2 + 2)[2:]
        elif self.hfmt == 1:
            # 7 bytes long
            return "{0:#0{1}x}".format(
                1           * 16 ** (4 * 2) +\
                self.length * 16 ** (1 * 2) +\
                self.typeid,
                7 * 2 + 2)[2:]
        elif self.hfmt == 2:
            # 3 bytes long
            return "{0:#0{1}x}".format(1, 3 * 2 + 2)[2:]
        else:
            raise Exception("unknown header format")

    @property
    def header(self):
        return self.basic_header + self.message_header

class RTMPChunkSizePacket(RTMPPacket):

    def __init__(self, size):
        if size >= 2 ** 32:
            raise Exception("chunk size too big")
        self.size = size
        self.hfmt = 0
        self.chunk_streamid = 2
        self.typeid = 1
        self.streamid = 0

    @property
    def body(self):
        return "{0:#0{1}x}".format(self.size, 4 * 2 + 2)[2:]

class RMTPGenericPacket(RTMPPacket):

    def __init__(self, hfmt, chunk_streamid, body, streamid=1):
        self.hfmt = hfmt
        self.chunk_streamid = chunk_streamid
        self.streamid = streamid
        self.hard_body = body
        self.typeid = 20

    @property
    def body(self):
        return self.hard_body

class RTMPCommandResult(RMTPGenericPacket):

    def __init__(self):
        super(RTMPCommandResult, self).__init__(
            hfmt=0,
            chunk_streamid=3,
            streamid=0,
            body="0200075f726573756c74003ff0000000000000030006666d7356657202000e464d532f332c352c352c32303034000c6361706162696c697469657300403f00000000000000046d6f6465003ff00000000000000000090300056c6576656c0200067374617475730004636f646502001d4e6574436f6e6e656374696f6e2e436f6e6e6563742e53756363657373000b6465736372697074696f6e020015436f6e6e656374696f6e207375636365656465642e0008636c69656e746964004094e40000000000000e6f626a656374456e636f64696e67000000000000000000000009",
            )

class RTMPOnPublish(RMTPGenericPacket):

    def __init__(self):
        super(RTMPOnPublish, self).__init__(
            hfmt=1,
            chunk_streamid=3,
            body="02000b6f6e46435075626c69736800000000000000000005030004636f64650200174e657453747265616d2e5075626c6973682e5374617274000b6465736372697074696f6e020027506c6561736520666f6c6c6f7775702077697468207075626c69736820636f6d6d616e642e2e2e000009",
            )

class RTMPCommandResult1(RMTPGenericPacket):

    def __init__(self):
        super(RTMPCommandResult1, self).__init__(
            hfmt=1,
            chunk_streamid=3,
            body="0200075f726573756c7400401000000000000005003ff0000000000000",
            )

class RTMPCommandResult2(RMTPGenericPacket):

    def __init__(self):
        super(RTMPCommandResult2, self).__init__(
            hfmt=0,
            chunk_streamid=3,
            body="0200075f726573756c74004014000000000000050101",
            streamid=1,
            )

class RTMPWindowAcknowledgement(RTMPPacket):

    def __init__(self, size):
        self.size = size
        self.hfmt = 1
        self.chunk_streamid = 2
        self.typeid = 5

    @property
    def body(self):
        return "{0:#0{1}x}".format(self.size, 4 * 2 + 2)[2:]

class RTMPSetPeerBandwidth(RTMPPacket):

    def __init__(self, size, limit_type):
        self.size = size
        self.ltype = limit_type
        self.hfmt = 1
        self.chunk_streamid = 2
        self.typeid = 6

    @property
    def body(self):
        return "{0:#0{1}x}".format(
            self.size * 16 ** (1 * 2) + self.ltype, 5 * 2 + 2)[2:]

class RTMPUserControlMessageStreamBegin(RTMPPacket):

    def __init__(self, hfmt=1):
        self.hfmt = hfmt
        self.chunk_streamid = 2
        self.typeid = 4

    @property
    def body(self):
        return "{0:#0{1}x}".format(1, 6 * 2 + 2)[2:]

def bin_str_to_int(string):
    try:
        return int(binascii.hexlify(string), 16)
    except ValueError:
        raise Exception("not bin data, someone is smoking")

def hexa_str_to_int(string):
    try:
        return int(string, 16)
    except ValueError:
        raise Exception("not hexa data, someone is smoking")

def split_n(n, string):
    tmp = []
    while string:
        tmp.append(string[:n])
        string = string[n:]
    return tmp

def hexa_str_to_streamid(string):
    tab = split_n(2, string)[::-1]
    string = "".join(tab)
    return hexa_str_to_int(string)

FMT_TO_LENGTH = {
    0 : 11,
    1 : 7,
    2 : 3,
    3 : 0,
    }

CHUNK_SIZE = 4096
HARD = [
    RTMPChunkSizePacket(CHUNK_SIZE),
    RTMPWindowAcknowledgement(2500000)  + \
        RTMPSetPeerBandwidth(2500000, 0)    + \
        RTMPUserControlMessageStreamBegin() + \
    RTMPCommandResult(),
    RTMPOnPublish(),
    RTMPCommandResult1() + \
        RTMPUserControlMessageStreamBegin(2),
    RTMPCommandResult2(),
    RTMPUserControlMessageStreamBegin(2),
    ]

class RTMPServer(vstcp.VSTCPServer):

    def __init__(self, host, port, timeout=1, logfile=None):
        super(RTMPServer, self).__init__(host, port, timeout, logfile)
        self.handshake = False
        self.c1 = None
        self.s1 = None
        self.state = -1
        self.video = 0
        self.audio = 0
        self.current_video = 0
        self.current_audio = 0
        self.current_bytes = 0
        self.time_ref = 0
        self.previous_length = 0

    def mock(self):
        self.state += 1
        self.logger.debug("We are in state " + str(self.state))
        self.conn.sendall(binascii.unhexlify(
            str(HARD[self.state]).replace(" ", "")))

    def c0(self):
        self.c1 = None
        self.s1 = None
        self.handshake = False
        self.state = -1

    def c0_c1(self):
        self.conn.sendall(S0)
        self.c1 = self.data
        self.s1 = binascii.unhexlify(ETIME + "00000000" + RANDOM)
        self.conn.sendall(self.s1)

    def handle(self):
        #hanshake and all
        if self.state < len(HARD) - 1:
            self.data = self.conn.recv(1)
            if self.handshake:
                basic_header = bin_str_to_int(self.data)
                fmt = basic_header >> 6
                tmp = ""
                if fmt == 0:
                    tmp = self.conn.recv(11)
                elif fmt == 1:
                    tmp = self.conn.recv(7)
                if not tmp:
                    return
                message_header = binascii.hexlify(tmp)
                message_length = hexa_str_to_int(message_header[3 * 2:6 * 2])
                self.logger.debug("packet with body length {}".format(
                    message_length))
                tmp = self.conn.recv(message_length)
                self.mock()
                if self.state in [0, 2]:
                    self.mock()
                return

            if self.data == '':
                return
            if self.data == C0:
                self.logger.debug("received handshake C0")
                self.c0()
                self.data = self.conn.recv(1536)
                if self.data[4:8] == binascii.unhexlify("00000000"):
                    self.logger.debug("received handshake C1")
                    self.c0_c1()
            else:
                if not self.handshake:
                    self.data += self.conn.recv(1536 - 1)
                if self.data[4:8] == binascii.unhexlify("00000000"):
                    if self.data == self.s1:
                        self.logger.debug("received handshake C2")
                        self.handshake = True
                        self.conn.sendall(self.c1)
                        self.time_ref = time.time()
        else:
            # take the first byte
            max_size = 128
            first = ""
            while len(first) == 0:
                first = self.conn.recv(1)
            basic_header = bin_str_to_int(first)
            fmt = basic_header >> 6
            csid = basic_header % 64
            # look if the basic RTMP header is 1, 2 or 3 bytes long
            if csid == 0:
                tmp = self.conn.recv(1)
                chunck_streamid = bin_str_to_int(tmp)
            elif csid == 63:
                tmp = self.conn.recv(2)
                chunck_streamid = bin_str_to_int(tmp)
            else:
                tmp = ""
                chunck_streamid = csid
            self.logger.debug(
                "received packet of format: {}, CSID: {}".format(
                    fmt, chunck_streamid))
            message_length = 0
            # Done with basic header, lets git through message header
            if fmt == 0:
                tmp = self.conn.recv(11)
                message_header = binascii.hexlify(tmp)
                timestamp = hexa_str_to_int(message_header[:3 * 2])
                message_length = hexa_str_to_int(message_header[3 * 2:6 * 2])
                message_typeid = hexa_str_to_int(message_header[6 * 2:7 * 2])
                if message_typeid == 9:
                    self.current_video += 1
                    self.video += 1
                    self.current_bytes += message_length
                elif message_typeid == 8:
                    self.current_audio += 1
                    self.audio += 1
                    self.current_bytes += message_length
                message_steamid = hexa_str_to_streamid(message_header[7 * 2:])
                self.previous_length = message_length
                self.logger.debug(
                    "packet with body length {}, type ID {} and stream ID {}".format(
                        message_length, message_typeid, message_steamid))

            elif fmt == 1:
                tmp = self.conn.recv(7)
                message_header = binascii.hexlify(tmp)
                timestamp = hexa_str_to_int(message_header[:3 * 2])
                message_length = hexa_str_to_int(message_header[3 * 2:6 * 2])
                message_typeid = hexa_str_to_int(message_header[6 * 2:7 * 2])
                if message_typeid == 9:
                    self.current_video += 1
                    self.video += 1
                    self.current_bytes += message_length
                elif message_typeid == 8:
                    self.current_audio += 1
                    self.audio += 1
                    self.current_bytes += message_length
                self.previous_length = message_length
                self.logger.debug(
                    "packet with body length {}, type ID {}".format(
                        message_length, message_typeid))

            elif fmt == 2:
                tmp = self.conn.recv(3)
                message_header = binascii.hexlify(tmp)
                timestamp = hexa_str_to_int(message_header[:3 * 2])
                message_length = self.previous_length
                self.current_bytes += message_length
                self.logger.debug("packet type 2")

            elif fmt == 3:
                timestamp = 0
                self.logger.debug("Packet of format 3 don't have headers")
            else:
                raise Exception("Well, this is weird")
            if timestamp == 16777215:
                tmp = self.conn.recv(4)
                timestamp = bin_str_to_int(tmp)
            self.logger.debug("timestamp : {}".format(timestamp))
            # header is done, let's analyse the body
            body = ""
            first = True
            while message_length:
                # if the mesage is long, take the chunk maximum size
                if message_length > max_size:
                    # one byte for the header
                    tmp = self.conn.recv(max_size + 1)
                    message_length -= max_size
                # else just read the all chunk
                else:
                    tmp = self.conn.recv(message_length)
                    message_length = 0
                if first:
                    first = False
                # after the first packet there will a 1 byte header
                # (probably C3)
                else:
                    tmp = tmp[1:]
                body += tmp
            # let's be sure about our body length
            rest = self.previous_length - len(body)
            if rest:
                tmp = self.conn.recv(rest)
                body += tmp
            now = time.time()
            if now - self.time_ref >= 2:
                self.time_ref = now
                self.logger.info(
                    "received {} video and {} audio packets for {} bytes ({}KB/s)".format(
                        self.current_video,
                        self.current_audio,
                        self.current_bytes,
                        self.current_bytes/2000.,
                        ))
                self.current_video = 0
                self.current_audio = 0
                self.current_bytes = 0


if __name__ == '__main__':
    PARSER = OptionParser()
    PARSER.add_option(
        '-a',
        '--address',
        dest='address',
        default='127.0.0.1',
        help="listening IP address. Default '127.0.0.1'",
        )
    PARSER.add_option(
        '-p',
        '--port',
        dest='port',
        default=1935,
        type="int",
        help='listening port number. Default 1935',
        )
    PARSER.add_option(
        '-l',
        '--logfile',
        dest='logfile',
        default='vsrtmp.log',
        help="log file. Default ./vsrtmp.log",
        )
    (OPTIONS, _) = PARSER.parse_args()
    SERVER = RTMPServer(OPTIONS.address, OPTIONS.port, logfile=OPTIONS.logfile)
    SERVER.start()

