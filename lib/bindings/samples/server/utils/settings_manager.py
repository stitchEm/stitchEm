import json
import os
import logging
from threading import Lock

import cli
import defaults

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

CURRENT_DIR = os.path.dirname(os.path.realpath(os.path.abspath(__file__)))
SETTINGS_PATH = os.path.join(defaults.USER_CONFIG_PATH, "server_settings.json")


class SettingsManager(object):
    """
    Manage settings
    """

    def __init__(self, cmdline_options, settings_path=SETTINGS_PATH):

        # Avoid saving some parameters not relevant to settings
        super(SettingsManager, self).__setattr__("mutex", Lock())
        super(SettingsManager, self).__setattr__("settings_path", settings_path)

        self._read_defaults()

        force_default = self.force_default
        if hasattr(cmdline_options, 'force_default') and cmdline_options.force_default:
            force_default = cmdline_options.force_default

        if not force_default:
            self._read_file_config()
        elif os.path.isfile(self.settings_path):
            os.remove(self.settings_path)

        self._read_cmdline(cmdline_options)

    def __setattr__(self, name, value):
        with self.mutex:
            super(SettingsManager, self).__setattr__(name, value)
            self.save_field(name)

    def _read_defaults(self):
        vars(self).update(defaults.DEFAULT_OPTIONS)

    def _read_file_config(self):
        with self.mutex:
            try:
                with open(self.settings_path) as settings_file:
                    self.__dict__.update(json.load(settings_file))
            except Exception:
                logger.error("Could not read settings file. Using default settings.")

    def _read_cmdline(self, cmdline_options):
        vars(self).update({key: value
                           for key, value in vars(cmdline_options).iteritems()
                           if value is not None})

    def save_field(self, field_name, output_file=None):
        """
        Save value of the field to the file.
        :param field_name:
        :param output_file Output file if it should be different from current settings path
        :return:
        """
        output_file = output_file if output_file else self.settings_path
        current_values = {}
        try:
            with open(output_file) as save_file:
                current_values = json.load(save_file)
        except Exception as error:
            logger.error("Error while reading settings file: {}".format(str(error)))

        current_values[field_name] = getattr(self, field_name)
        try:
            with open(output_file, mode='w') as save_file:
                json.dump(current_values, save_file, indent=4, sort_keys=True)
        except Exception as error:
            logger.error("Error while reading settings file: {}".format(str(error)))

        logger.info("Wrote {}:{}".format(field_name, current_values[field_name]))


SETTINGS = SettingsManager(cli.parse_args()[0])
