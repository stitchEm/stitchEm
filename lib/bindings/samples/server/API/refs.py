import API.schema

""" Define reference structures used by api schemas.
"""


def load():
    API.schema.ref(
        {
            "id": "Quaternion",
            "type": "array",
            "items": {
                "type": "number"
            }

        }
    )

    API.schema.ref(
        {
            "id": "AudioSourceName",
            "type": "string",
            "description": "An audio source name"
        }
    )

    API.schema.ref(
        {
            "id": "AudioSourceLayout",
            "type": "string",
            "description": "A layout to be used in conjonction with an audio source"
        }
    )

    API.schema.ref(
        {
            "id": "StreamPreset",
            "type": "object",
            "properties": {
                "name": {
                    "type": "string"
                },
                "filename": {
                    "type": "string"
                },
                "bitrate": {
                    "type": "integer"
                },
                "profile": {
                    "type": "string"
                },
                "pub_user": {
                    "type": "string"
                },
                "pub_passwd": {
                    "type": "string"
                }
            },
            "required": ["name"]
        }
    )

    API.schema.ref(
        {
            "id": "DriveStatus",
            "description": "Represents a drive status",
            "type": "string",
            "enum": ["NoDeviceDetected", "InvalidDevice", "DeviceNotCompatible", "DeviceReadOnly", "DeviceOk",
                     "DeviceRemovable", "NotEnoughMemory"]
        }
    )

    API.schema.ref(
        {
            "id": "Drive",
            "type": "object",
            "properties": {
                "warning_cluster_size": {
                    "type": "boolean"
                },
                "warning_disk_full": {
                    "type": "boolean"
                },
                "free_mb": {
                    "type": ["integer", "null"]
                },
                "total_mb": {
                    "type": ["integer", "null"]
                },
                "state": {
                    "$ref": "DriveStatus"
                }
            },

        }
    )

    # Messages

    API.schema.ref(
        {
            "id": "Error",
            "type": "object",
            "properties": {
                "code": {
                    "type": "string"
                },
                "message": {
                    "type": ["string", "null"]
                },
                "time": {
                    "type": "integer"
                },
                "id": {
                    "type": "integer"
                }
            }
        }
    )

    API.schema.ref(
        {
            "id": "Event",
            "type": "object",
            "properties": {
                "name": {
                    "type": "string"
                },
                "payload": {
                    "type": ["object", "null"]
                }
            }
        }
    )

    API.schema.ref(
        {
            "id": "Message",
            "type": "object",
            "properties": {
                "event": {
                    "$ref": "Event"
                },
                "error": {
                    "$ref": "Error"
                }
            }
        }
    )

    # Status

    API.schema.ref(
        {
            "id": "OutputStatus",
            "description": "Represents the status of an output (stream or record)",
            "type": "string",
            "enum": ["Stopped", "Starting", "Started", "Retrying", "Stopping"]
        }
    )

    API.schema.ref(
        {
            "id": "NetworkStatus",
            "description": "Represents the network (wifi / eth) status",
            "type": "string",
            "enum": ["Connected", "Disconnected"]
        }
    )

    API.schema.ref(
        {
            "id": "CameraStatus",
            "description": "Represents the camera status",
            "type": "string",
            "enum": ["Initial",
                     "Discovering",
                     "Connecting",
                     "CheckingFirmware",
                     "FirmwareIncompatible_MustBeUpgraded",
                     "FirmwareCompatible_CanBeUpgraded",
                     "FirmwareIncompatible_CannotBeDowngraded",
                     "FirmwareCompatible_CannotBeDowngraded",
                     "UpdatingFirmware",
                     "FetchingCalibration",
                     "StartingStreams",
                     "Connected"]
        }
    )


    API.schema.ref(
        {
            "id": "SocialNetworkStatus",
            "description": "Represents the status of a social network",
            "type": "string",
            "enum": ["Initial",
                  "CheckingLink",
                  "Linking",
                  "Disconnected",
                  "Connected"]
        }
    )

