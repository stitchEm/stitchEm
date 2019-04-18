import threading
from collections import OrderedDict

from tornado.ioloop import IOLoop

import utils.async


class AlgorithmScheduler:
    """
    Thread safe Data Structure that let's you schedule and unschedule algorithms.
    """
    def __init__(self):
        self.schedule_lock = threading.Lock()
        self.scheduled_algorithms = OrderedDict()
        self.to_be_scheduled = {}
        self.to_be_scheduled_lock = threading.Lock()

    def schedule(self, algorithm, delay=0):
        """
        Schedules algorithm to be executed at some point in the future
        :param algorithm: Algorithm
        :param delay: The algorithm won't be scheduled for specified amount of time in seconds
        """
        if delay:
            self._schedule_delayed(algorithm, delay)
        else:
            self._schedule(algorithm)

    def _schedule(self, algorithm):
        with self.schedule_lock:
            # If there is already scheduled algorithm with same name - remove it.
            # We don't just reassign here because we want newly inserted algorithm to be at the end of the queue
            self.scheduled_algorithms.pop(algorithm.name, None)
            self.scheduled_algorithms[algorithm.name] = algorithm

    def _schedule_delayed(self, algorithm, delay):
        def delayed_task():
            with self.to_be_scheduled_lock:
                self.to_be_scheduled.pop(algorithm.name, None)
            self._schedule(algorithm)

        with self.to_be_scheduled_lock:
            self.to_be_scheduled[algorithm.name] = utils.async.delay(delay, delayed_task)

    def unschedule(self, algorithm_name):
        with self.to_be_scheduled_lock:
            callback = self.to_be_scheduled.pop(algorithm_name, None)
            if callback:
                IOLoop.current().remove_timeout(callback)

        with self.schedule_lock:
            self.scheduled_algorithms.pop(algorithm_name, None)

    def reschedule(self, algorithm):
        if algorithm.repeat:
            self.schedule(algorithm, delay=algorithm.delay)

    def get_next_algorithm(self):
        with self.schedule_lock:
            if self.scheduled_algorithms:
                return self.scheduled_algorithms.popitem(last=False)[1]
        return None
