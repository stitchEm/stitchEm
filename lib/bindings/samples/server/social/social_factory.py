from facebook_network import Facebook

class SocialFactory(dict):
    """
    Factory class to provide social network instances
    """

    def __init__(self):
        facebook = Facebook()
        self[facebook.name] = facebook

    def get_configuration(self):
        return { social_network_name: self[social_network_name].get_configuration() for social_network_name in self }

SOCIAL_NETWORKS = SocialFactory()