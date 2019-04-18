import threading
import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

class Loop(threading.Thread):
    """Calls a function every interval seconds:

            t = Loop(30.0, f, args=[], kwargs={})
            t.start()
            t.cancel()     # stop the loop's action if it's still waiting

    """

    STOP = True
    CONTINUE = False

    def __init__(self, interval, function, name=None, restart=False, args=[], kwargs={}):
        super(Loop, self).__init__(name=name)
        self.current_interval = 0
        self.interval = interval
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self.finished = threading.Event()
        self.running = threading.Event()
        self.running.set()
        self.daemon = True
        self.restart = restart

    def start(self, paused=False):
        if paused:
            self.running.clear()
        super(Loop, self).start()

    def set_interval(self, interval):
        self.interval = interval

    def is_finished(self):
        return self.finished.is_set()

    def pause(self):
        self.running.clear()

    def unpause(self):
        self.running.set()

    def is_paused(self):
        return not self.running.is_set()

    def finish(self):
        self.unpause()
        self.finished.set()

    def cancel(self):
        self.finish()
        self.join()

    def run(self):
        logging.info("Starting loop {}".format(self.name))

        while True:
            try:
                while not self.finished.is_set():
                    self.finished.wait(self.current_interval)
                    self.running.wait()
                    if not self.finished.is_set():
                        stop = self.function(*self.args, **self.kwargs)
                        if stop:
                            break
                    else:
                        break
                    self.current_interval = self.interval
                break
            except Exception as e:
                logging.error("Exception in loop {}: {}".format(self.name, str(e)))
                if not self.restart:
                    break
                logging.info("Restarting loop {}".format(self.name))

        logging.info("Stopped loop {}".format(self.name))
