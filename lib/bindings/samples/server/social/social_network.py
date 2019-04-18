from transitions.extensions import LockedMachine as Machine

import errors
from utils.settings_manager import SETTINGS

class SocialNetwork(object):
    """
    Abstract class to provide a link to a social API
    """

    def __init__(self, name):
        self.name = name

        # State Machine

        states = ["Initial",
                  "CheckingLink",
                  "Linking",
                  "Disconnected",
                  "Connected"]

        transitions = [
            {"source": "Initial",
             "trigger": "t_check_link",
             "dest": "CheckingLink"},

            {"source": ["Disconnected"],
             "trigger": "t_make_connection",
             "before": "_make_connection",
             "dest": "Linking"},

            {"source": ["CheckingLink", "Linking"],
             "trigger": "t_connected",
             "dest": "Connected"},

            {"source": ["CheckingLink", "Linking"],
             "trigger": "t_not_connected",
             "dest": "Disconnected"},

            {"source": "Connected",
             "trigger": "t_remove_connection",
             "before": "_remove_connection",
             "dest": "Disconnected"},
        ]

        self.machine = Machine(
            name=self.name, model=self, states=states, transitions=transitions,
            initial="Initial", async=True)
        self.t_check_link()

    def _saveLinkData(self, data):
        SETTINGS.social_links[self.name] = data
        #force save
        SETTINGS.social_links = SETTINGS.social_links

    def on_enter_CheckingLink(self):
        # check if link data is present
        if self.name in SETTINGS.social_links:
            if self._check_connection(SETTINGS.social_links[self.name]):
                return self.t_connected()
        self.t_not_connected()

    def _check_connection(self, connection_data):
        # this method is supposed to be redefined in child classes
        raise errors.FatalError("social network missing _check_connection implementation")

    def _make_connection(self, connection_data):
        # this method is supposed to be redefined in child classes
        raise errors.FatalError("social network missing _make_connection implementation")

    def _remove_connection(self):
        # remove link data
        if self.name not in SETTINGS.social_links:
            raise errors.FatalError("social network state error : was in connected state but cannot find link data")
        del (SETTINGS.social_links[self.name])

    def get_configuration(self):
        # this method is supposed to be redefined in child classes
        raise errors.FatalError("social network missing get_configuration implementation")

    def call_api(self, api_endpoint, parameters):
        # this method is supposed to be redefined in child classes
        raise errors.FatalError("social network missing call_api implementation")

