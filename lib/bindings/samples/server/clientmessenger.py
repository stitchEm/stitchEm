import logging

from utils.ringbuffer import RingBuffer

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class ClientMessenger(object):
    """ Handles communication to the client
    """

    def __init__(self):
        """Init
        """
        self.clear()

    def clear(self):
        self.buffer = RingBuffer(256)

    def get_message_count(self):
        """ Get the number of errors in the buffer

        Returns:
            The number of errors.
        """
        return self.buffer.count()

    def get_last_messages(self, count):
        """ Get the 'count' last errors from the buffer

        Args:
            count(int): Number of errors to get
        Returns:
            The last 'count' errors.
        """
        return self.buffer.get_last(count)

    def _send(self, message):
        """ Sends a message to the box

        The message will be stored in a ring buffer and available for the
        clients to download later.

        Args:
            message: Message to store in the buffer.
        Note:
            Only the last 256 messages are kept.
        """
        logger.info(message)
        self.buffer.put(message)

    def send_error(self, e):
        """Raise an asynchronous error"""
        self._send({'error': e.payload})

    def send_event(self, name, payload=None):
        """Sends an asynchronous message"""
        if payload and not (isinstance(payload, dict) or isinstance(payload, object)):
            logger.error("Payload for {} is not an object, ignored: {}".format(name, str(payload)))
            payload = None
        self._send({'event': {'name': name, 'payload': payload}})


CLIENT_MESSENGER = ClientMessenger()
