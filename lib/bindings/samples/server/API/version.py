import logging
from os import path
import ConfigParser
from os import path as osp
import API.schema
import json

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

config = ConfigParser.SafeConfigParser()
FILE_PATH = path.realpath(path.abspath(__file__))
DIR_PATH = path.dirname(FILE_PATH)
config.readfp(open(DIR_PATH + '/version'))

BUILD_VERSION = ""
API_VERSION = config.get('DEFAULT', 'apiVersion')
SERVER_VERSION = config.get('DEFAULT', 'serverVersion')
SUPPORTED_FIRMWARES = config.get('DEFAULT', 'supportedFirmwares')
UNSUPPORTED_FIRMWARES = config.get('DEFAULT', 'unsupportedFirmwares')
FIRMWARE_VERSION = config.get('DEFAULT', 'firmwareVersion')
FIRMWARE_UPDATES = json.loads(config.get('DEFAULT', 'firmwareUpdates'))
API_HASH = ""
BOX_HARDWARE="undefined"
try:
	with open("/sys/devices/virtual/dmi/id/product_name") as f:
		BOX_HARDWARE=f.read().rstrip("\n")
except:
	pass

def set_globals():
    """ Set global variables
    """
    global BUILD_VERSION
    global API_HASH
    try:
        with open(osp.join(DIR_PATH, "..", "BuildVersionFile"), "r") as version_file:
            version_data = version_file.read()
            logger.info("\tBuild : " + version_data)
            BUILD_VERSION += version_data
    except IOError as error:
        logger.error("Error while loading the version information: {}".format(error))
    API_HASH = API.schema.hash_schema()
