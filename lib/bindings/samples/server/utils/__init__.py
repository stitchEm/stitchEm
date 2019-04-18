import logging
from types import MethodType

import blinker

from utils import async


def send_async(self, *sender, **kwargs):
    async.defer(self.send, *sender, **kwargs)


def send_logged(self, *sender, **kwargs):
    logging.info("Signal from: {}. Signal name is: {}".format(str(sender),
                                                              getattr(self, "name", "")))
    self.send_orig(*sender, ** kwargs)

blinker.Signal.send_orig = blinker.Signal.send
#blinker.Signal.send = MethodType(send_logged, None, blinker.Signal)
blinker.Signal.send_async = MethodType(send_async, None, blinker.Signal)

