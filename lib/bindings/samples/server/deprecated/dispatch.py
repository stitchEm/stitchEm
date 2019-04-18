# This file should be deprecated when we will get rid of nginx stream recording


from collections import OrderedDict
import logging


# Todo Replace all usages (nginx) by blinker
class Event(object):
    def __init__(self):
        self.__name = None

    def setName(self, name):
        self.__name = name

    def getName(self):
        return self.__name


class EventDispatcher(object):
    def __init__(self):
        self._listeners = {}

    def dispatch(self, eventName):
        logging.info("Dispatching " + eventName)
        if eventName not in self._listeners:
            return
        for listener in self._listeners[eventName].values():
            # TODO : store arguments
            listener()
        return

    def addListener(self, eventName, listener, priority=0):
        if eventName not in self._listeners:
            self._listeners[eventName] = {}
        self._listeners[eventName][priority] = listener
        self._listeners[eventName] = OrderedDict(sorted(self._listeners[eventName].items(), key=lambda item: item[0]))

    def removeListener(self, eventName, listener=None):
        if eventName not in self._listeners:
            return
        if not listener:
            del self._listeners[eventName]
        else:
            for p, l in self._listeners[eventName].items():
                if l is listener:
                    self._listeners[eventName].pop(p)
                    return
