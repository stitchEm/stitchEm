import time


class Timer(object):
    """Auxiliary class for meassuring elapsed time
    """

    def __init__(self, name):
        self.name = name
        self.elapsed = None

    def start(self):
        """Cheks the starting point
        """
        self.elapsed = time.time()

    def reset(self):
        """Restart the timer
        """
        self.elapsed = None

    def get_elapsed(self):
        """Returns the current elapsed time since its starting point
        Returns:
            number: The elapsed time in seconds, as an integer
        """
        if self.elapsed is None:
            return 0
        else:
            return int(time.time() - self.elapsed)
