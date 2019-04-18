from tornado.ioloop import IOLoop


def defer(mcallable, *args, **kwargs):
    """ Defer a function for next loop iteration"""
    return IOLoop.current().spawn_callback(mcallable, *args, **kwargs)


def delay(my_delay, mcallable, *args, **kwargs):
    """ Delay function call by time specified in delay"""
    return IOLoop.current().call_later(my_delay, mcallable, *args, **kwargs)
