import httplib
import json

class SBHTTPClient(object):

    def __init__(self, addr, port, path):
        self.path = path
        self.addr = addr
        self.port = port
        self.connection = None

    def request(self, method, path, body, headers=None):
        if self.connection is not None:
            self.connection.close()
        self.connection = httplib.HTTPConnection(self.addr, self.port)
        if headers is None:
            headers = {}
        return self.connection.request(method, path, body, headers)

    def getresponse(self):
        res = self.connection.getresponse()
        if res.status != 200:
            raise Exception("Request failed : {} ({})".format(res.reason,
                                                              res.status))
        try:
            res.body = json.loads(res.read())
        except:
            raise Exception("Cannot parse response")
        return res

    def query(self, cmd):
        body = '{{"name" : "{}"}}'.format(cmd)

        if cmd == "debug.stop_server" or cmd == "stitcher.stop_stream":
            try:
                self.request('POST', self.path, body)
            except:
                pass
        else:
            self.request('POST', self.path, body)

    def query_with_body(self, body):
        self.request('POST', self.path, json.dumps(body))
