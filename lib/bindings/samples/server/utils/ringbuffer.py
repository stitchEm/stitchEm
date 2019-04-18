import collections
import threading


class RingBuffer(object):
    """Ring buffer
    """

    def __init__(self, size=4096):
        self._buf = collections.deque(maxlen=size)
        self._count = 0
        self.lock = threading.Lock()

    def put(self, data):
        """Adds object to the end of the buffer
        """
        with self.lock:
            self._buf.append(data)
            self._count += 1

    def get_last(self, size):
        """Retrieves the last count objects from the buffer
        """
        data = []
        with self.lock:
            if size is None:
                size = len(self._buf)
            else:
                size = size if size < len(self._buf) else len(self._buf)
            for i in xrange(size):
                data.append(self._buf[len(self._buf) - i - 1])
        return data

    def count(self):
        """Returns the overall count of objects added to the buffer
        """
        with self.lock:
            count = self._count
        return count
