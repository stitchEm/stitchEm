import json
import os
from os import listdir, path
from os.path import isfile
import re

import defaults
from errors import PresetError, PresetDoesNotExist, PresetInvalidName, PresetCannotBeDeleted


class PresetManager(object):
    UNDELETABLE_FLAG = "undeletable"
    NAME_INDEX = "name"

    def __init__(self, presets_dir="streaming"):
        self.user_presets_dir = path.join(defaults.USER_PRESETS_DIR_PATH, presets_dir)
        self.system_presets_dir = os.path.join(defaults.SYSTEM_PRESETS_DIR_PATH, presets_dir)
        self.presets = []
        self.load()

    def create(self, parameters):
        """Creates a new preset
        """
        name = parameters[self.NAME_INDEX]
        self._write(name, parameters)
        self.load()

    def load(self):
        """Loads the list of presets from local files
        """
        self.presets = self._load_files()

    def list(self):
        """Returns the list of presets
        """
        return self.presets

    def remove(self, name):
        """Removes a preset
        """
        # check if preset is deletable ( = user defined)
        if self.get(name) and self.get(name).get(self.UNDELETABLE_FLAG, False):
            raise PresetCannotBeDeleted(
                "Cannot delete preset {}. It is marked as undeletable".format(name))

        self._remove(name)
        self.load()

    def get(self, name):
        """Gets a specific preset
        """
        for p in self.presets:
            preset_name = p.get(self.NAME_INDEX, None)
            if preset_name and preset_name == name:
                return p

    def _get_path(self, name):
        """Gets the complete path of a preset
        """
        return path.join(self.user_presets_dir, name + defaults.PRESET_EXT)

    def _write(self, name, data):
        """Write a preset into a file
        """
        if not re.match(r"^[a-zA-Z0-9_.-]{4,20}$", name):
            raise PresetInvalidName
        try:
            with open(self._get_path(name), 'w') as output:
                json.dump(data, output, indent=2, sort_keys=True)
        except:
            raise PresetError()

    def _read(self, preset_file):
        """Load a preset from a file
        """
        try:
            with open(preset_file, 'r') as input:
                return json.load(input)
        except Exception:
            return None

    def _remove(self, name):
        """Remove a presets file
        """
        preset_file_path = self._get_path(name)
        if not path.exists(preset_file_path):
            raise PresetDoesNotExist(
                "Cannot delete {}. File not found".format(name))
        try:
            os.remove(preset_file_path)
        except:
            raise PresetError("Error while removing preset")

    def _load_files(self):
        """Load all presets : from system preset directory and from user directory
        """
        presets = []
        for preset_filename in listdir(self.system_presets_dir):
            full_path = path.join(self.system_presets_dir, preset_filename)
            if isfile(full_path) and preset_filename.endswith(defaults.PRESET_EXT):
                if preset_filename != defaults.DEFAULT_PRESET_FILENAME:
                    preset = self._read(full_path)
                    # presets from system will be marked as undeletable
                    if preset:
                        preset[self.UNDELETABLE_FLAG] = True;
                        presets.append(preset)

        for preset_filename in listdir(self.user_presets_dir):
            full_path = path.join(self.user_presets_dir, preset_filename)
            if isfile(full_path) and preset_filename.endswith(defaults.PRESET_EXT):
                preset = self._read(full_path)
                if preset:
                    presets.append(preset)

        return presets
