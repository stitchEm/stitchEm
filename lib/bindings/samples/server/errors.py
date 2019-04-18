from time import time

"""Error tree module
"""


# Exception hierarchy

class VSError(Exception):
    """StitchingBox Error Base class

    Args:
        expr(string): Context or expression where the error occurred.
        msg(string): Explanation of the error.

    Exceptions tree::

      VSError
        - InternalError
          - FatalError
          - APIError
            - InvalidParameter
            - MissingParameter
            - InvalidMessageFormat
          - ConfigurationError
          - LogError
            - LogBackupNoSpace
            - LogBackupNoDrive
            - LogBackupCopyError
          - OutputError
            - OutputAlreadyStarted
            - OutputAlreadyStopped
            - MultipleOutputsForbidden
            - RecordingError
              - RecordingPathInvalid
              - RecordingDriveUnplugged
              - RecordingDriveFull
            - StreamingError
              - StreamingInvalidURL
              - StreamingAuthFailed
              - StreamingConnectionLost
          - StitcherError
            - StitcherVideoModeInvalidError
            - StitcherVideoModeUnsupportedError
            - StitcherVideoModeChangeForbiddenError
            - StitcherCannotBeStarted
            - StitcherCannotBeStopped
            - StitcherNotStarted
          - AudioError
            - AudioInvalidSourceError
            - AudioSourceChangeForbiddenError
          - PresetsError
            - PresetAlreadyCreated
            - PresetDoesNotExist
            - PresetInvalidName
            - PresetCannotBeDeleted
          - AlgorithmError
            - AlgorithmRunningError
          - CameraError
            - CameraDisconnected
            - CameraVideoFail
          - SocialError
          - WebsocketError (only for debug)
            - AlreadyConnected
            - NotConnected
            - CannotConnect
            - InvalidParameter
         - SystemError
            - WifiError
              - BadPassword
              - InternalError
    """

    id = 0

    def __init__(self, msg=""):
        Exception.__init__(self)
        self.payload = {'message': msg,
                        'code': self.__class__.__name__,
                        'time': int(time()),
                        'id': VSError.id}
        VSError.id += 1


#


class InternalError(VSError):
    """Some other event that cannot be handled from the client side.
    """


class FatalError(VSError):
    """This error depicts the worst case and should block the rest of the
    execustion requested by the client.
    """


#


class APIError(VSError):
    """This describes possible API calls errors.
    """


class InvalidParameter(APIError):
    """The parameter/s of the API call is not the expected one. (Type)
    """


class MissingParameter(APIError):
    """Missing parameter in the API call.
    """


class InvalidMessageFormat(APIError):
    """The JSON message is not valid. (Syntax error)
    """


class InvalidReturnValue(APIError):
    """Missing parameter in the API call.
    """


#


class ConfigurationError(VSError):
    """Error related with the project configuration.
    """


#


class LogError(VSError):
    """Error related with the Box error management
    """


class LogBackupNoSpace(LogError):
    """The backup unit for the log copy has not enough space
    """


class LogBackupNoDrive(LogError):
    """The destination drive for the log backup is not available.
    """


class LogBackupCopyError(LogError):
    """Error during log backup
    """


#
class OutputError(VSError):
    """Generic output error
    """


class OutputAlreadyStarted(OutputError):
    """The output is already started.
    """


class OutputAlreadyStopped(OutputError):
    """The output is already stopped.
    """

class MultipleOutputsForbidden(OutputError):
    """The output cannot be started as another one is already started
    """

class RecordingError(OutputError):
    """Errors related with the recording feature.
    """


class RecordingPathInvalid(RecordingError):
    """The path of recording is not valid in the server.
    """


class RecordingDriveUnplugged(RecordingError):
    """The recording drive is not valid anymore.

    For example: SDCARD extracted during recording.
    """


class RecordingDriveFull(RecordingError):
    """The recording drive has no more space available.
    """


#
class StreamingError(OutputError):
    """Errors related with the streaming.
    """


class StreamingInvalidURL(StreamingError):
    """The streaming URL is not valid. (Server responding 404)
    """


class StreamingAuthFailed(StreamingError):
    """The credentials provided to login are not valid.
    """


class StreamingConnectionLost(StreamingError):
    """The connection to the server is lost.
    """


#
class StitcherError(VSError):
    """Errors comming from the stitching process.
    """


class StitcherVideoModeInvalidError(StitcherError):
    """ A video mode cannot be applied to the stitcher because it is invalid
    """


class StitcherVideoModeUnsupportedError(StitcherError):
    """ A video mode cannot be applied to the stitcher because it is not supported
    """


class StitcherVideoModeChangeForbiddenError(StitcherError):
    """ The current video mode cannot be changed because server is currently streaming or recording
    """


class StitcherCannotBeStarted(StitcherError):
    """ The stitching can't be started.
    """


class StitcherCannotBeStopped(StitcherError):
    """The stitching can't be stopped.
    """


class StitcherNotStarted(StitcherError):
    """The stitcher cannot start (libvideostitch error)
    """


#
class AudioError(VSError):
    """Errors related to audio
    """


class AudioInvalidSourceError(AudioError):
    """ The specified audio source is not valid
    """

class AudioSourceChangeForbiddenError(AudioError):
    """ Audio source cannot be changed while streaming or recording is ongoing
    """

class AudioDelaySourceError(AudioError):
    """ The specified audio source has not an adjustable delay
    """


class AudioInvalidDelayValueError(AudioError):
    """ The delay value is out of acceptable range
    """


#

class PresetError(VSError):
    """Preset errors
    """

    def __init__(self, msg=""):
        VSError.__init__(self, msg)


class PresetDoesNotExist(PresetError):
    """Cannot find preset for removing it
    """


class PresetAlreadyCreated(PresetError):
    """Preset name already exists
    """


class PresetInvalidName(PresetError):
    """Preset name not valid
    """


#
class PresetCannotBeDeleted(PresetError):
    """This preset cannot be deleted
    """


#

class CameraError(VSError):
    """Camera errors
    """


class CameraIsNotConnected(CameraError):
    """Trying to do something while camera is disconnected
    """


class CameraVideoFail(CameraError):
    """The camera indicated a video fail
    """


class CameraInvalidCalibration(CameraError):
    """The calibration is not valid
    """

    def __init__(self, msg, calibrationFile):
        CameraError.__init__(self, msg)
        self.payload['file'] = calibrationFile

#

class SocialError(VSError):
    """social networks related errors
    """


#
class WebSocketError(VSError):
    """Websockets errors
    """


class AlreadyConnected(WebSocketError):
    """The socket is already connected
    """


class NotConnected(WebSocketError):
    """The socket is not connected
    """


class CannotConnect(WebSocketError):
    """Cannot connect to server
    """


class InvalidParameter(WebSocketError):
    """The message has invalid / missing paramters
    """


class AlgorithmError(VSError):
    """Generic Algorithm Error"""


class AlgorithmRunningError(AlgorithmError):
    """Error while running the algorithm"""


class FirmwareError(VSError):
    """Generic Firmware Error"""


class FirmwareUpdateNotAvailable(FirmwareError):
    """Cannot update firmware at this time"""

class FirmwareUpdateFailed(FirmwareError):
    """Firmware update failed"""


class InternalFirmwareError(FirmwareError):
    """Internal error while updating firmware"""


class ParserError(VSError):
    """Generic Parser Error"""


class ParserFileNotFound(ParserError):
    """Parser could not find the file"""
    
    
    
class SystemError(VSError):
    """Generic system conf error
    """


class WifiError(SystemError):
    """The wifi management failed.
    """


class WrongWifiPassword(WifiError):
    """The user provided password doesn't match the system one.
    """
    
class InternalWifiError(WifiError):
    """Something went wrong during config modification.
    """

