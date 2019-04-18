import facebook
import hashlib
import hmac

from utils import async
import errors
from social_network import SocialNetwork

API_VERSION = "2.8"
APP_ID = "226191117840826"
APP_SECRET = "c7b1fa1c0adf7bf1427aaf461138c954"

class Facebook(SocialNetwork):
    """
    Provide link with Facebook API
    """

    def __init__(self):
        super(Facebook, self).__init__("facebook")

    def get_configuration(self):
        return {
            'version': '2.8',
            'app_id': APP_ID
        }

    def _check_connection(self, connection_data):
        token = connection_data
        self.graph = facebook.GraphAPI(access_token=token, version=API_VERSION)
        self.appsecret_proof = hmac.new(APP_SECRET.encode('ascii'),
                                        msg=token.encode('ascii'),
                                        digestmod=hashlib.sha256
                                        ).hexdigest()
        try:
            self.graph.get_object('me', appsecret_proof=self.appsecret_proof)
        except facebook.GraphAPIError:
            return False
        return True

    def _make_connection(self, connection_data):
        client_token = connection_data
        self.graph = facebook.GraphAPI(access_token=client_token, version=API_VERSION)
        try:
            response = self.graph.extend_access_token(app_id=APP_ID, app_secret=APP_SECRET)
            self.graph = facebook.GraphAPI(access_token=response['access_token'], version=API_VERSION)
            self.appsecret_proof = hmac.new(APP_SECRET.encode('ascii'),
                                            msg=response['access_token'].encode('ascii'),
                                            digestmod=hashlib.sha256
                                            ).hexdigest()
        except facebook.GraphAPIError:
            return async.defer(self.t_not_connected)
        self._saveLinkData(response['access_token'])
        async.defer(self.t_connected)


    #facebook api python implementation is a bit weird, and instead of api endpoint, object id should be sent
    def call_api(self, api_endpoint, parameters):
        if self.graph is None:
            raise errors.SocialError("Trying to call {} api but box has no link with this social network", self.name)
        if parameters is None:
            parameters = {'appsecret_proof': self.appsecret_proof}
        else:
            parameters['appsecret_proof'] = self.appsecret_proof
        return self.graph.get_object(api_endpoint, **parameters)
