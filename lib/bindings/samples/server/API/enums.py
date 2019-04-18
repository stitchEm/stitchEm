from enum import Enum


# TODO : Still used by NGINX. Remove when we get rid of NGINX recording


class RecordingStatus(Enum):
    Stopped, Started = range(2)


class NetworkStatus(Enum):
    Connected, Disconnected = range(2)
