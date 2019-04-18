from concurrent.futures import ThreadPoolExecutor
from tornado.concurrent import run_on_executor
from API.handlers import APIHandler
from API.schema import api
import errors
from social.social_factory import SOCIAL_NETWORKS

class SocialAPI(APIHandler):
    """REST interface related to social networks
    """
    executor = ThreadPoolExecutor(1)

    def __init__(self, extra):
        """Init
        """
        super(SocialAPI, self).__init__(extra)
        """
        self.server = extra["server"]
        self.project_manager = extra["project_manager"]
        self.output_manager = extra["output_manager"]
        """

    @api(name="MakeLink",
         endpoint="social.make_link",
         description="Link the box to a user account on a social network",
         parameters={
             "type": "object",
             "properties":
                 {
                     "social_network": {
                         "type": "string",
                         "enum": ["facebook"]
                     },
                     "token": {
                         "type": "string"
                     }
                 },
             "required": ["social_network", "token"]
         }
         )
    @run_on_executor
    def make_link(self, parameters):
        social_network_name = parameters.get("social_network")
        token = parameters.get("token")
        if social_network_name not in SOCIAL_NETWORKS:
            raise errors.InvalidParameter("the social network {} is not implemented".format(social_network_name))
        if not SOCIAL_NETWORKS[social_network_name].is_Disconnected():
            raise errors.InvalidParameter("cannot link to social network {} as it is in state {}".format(
                social_network_name, SOCIAL_NETWORKS[social_network_name].state))
        SOCIAL_NETWORKS[social_network_name].t_make_connection(token)


    @api(name="MakeLink",
         endpoint="social.unmake_link",
         description="Remove the link between the box and a user account on a social network",
         parameters={
             "type": "object",
             "properties":
                 {
                     "social_network": {
                         "type": "string",
                         "enum": ["facebook"]
                     },
                 },
             "required": ["social_network"]
         }
         )
    @run_on_executor
    def unmake_link(self, parameters):
        social_network_name = parameters.get("social_network")
        if social_network_name not in SOCIAL_NETWORKS:
            raise errors.InvalidParameter("the social network {} is not implemented".format(social_network_name))
        if not SOCIAL_NETWORKS[social_network_name].is_Connected():
            raise errors.InvalidParameter("cannot unlink from social network {} as it is in state {}".format(
                social_network_name, SOCIAL_NETWORKS[social_network_name].state))
        SOCIAL_NETWORKS[social_network_name].t_remove_connection()


    @api(name="Call",
         endpoint="social.call",
         description="Make a call to a social network",
         parameters={
             "type": "object",
             "properties":
                 {
                     "social_network": {
                         "type": "string",
                         "enum": ["facebook"]
                     },
                     "endpoint" : {
                         "type": "string"
                     },
                     "parameters" : {
                         "type": "object"
                     }
                 },
             "required": ["social_network", "endpoint"]
         },
         result={
             "type": "object"
         }
         )
    @run_on_executor
    def call(self, parameters):
        social_network_name = parameters.get("social_network")
        endpoint = parameters.get("endpoint")
        call_parameters = parameters.get("parameters")
        if social_network_name not in SOCIAL_NETWORKS:
            raise errors.InvalidParameter("the social network {} is not implemented".format(social_network_name))
        if not SOCIAL_NETWORKS[social_network_name].is_Connected():
            raise errors.InvalidParameter("cannot make API call to social network {} as it is in state {}".format(
                social_network_name, SOCIAL_NETWORKS[social_network_name].state))
        result = SOCIAL_NETWORKS[social_network_name].call_api(endpoint, call_parameters)
        return result if result is not None else {}