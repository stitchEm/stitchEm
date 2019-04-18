import logging
import hashlib
import jsonschema

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

refs = {}
apis = {}


def ref(schema):
    refs[schema.get("id")] = schema

def retrieve(uri):
    return refs[uri.replace("file:///", "")]


class api(object):
    """ Decorator used to document api calls.

    Parameters:
        name(string): Name that will be used as a type in the schema
        endpoint (string): API endpoint
        parameters({})(optional): Schema describing input parameters
        results({})(optional): Schema describing return value
        errors([VSErrors])(optional): List of possible errors
    """

    apis = []

    def __init__(self, **kwargs):
        self.name = kwargs["name"]
        self.endpoint = kwargs["endpoint"]
        self.parameters = kwargs.get("parameters", {"type": "null"})
        self.parameters_validator = jsonschema.Draft4Validator(self.parameters,
                                                               resolver=jsonschema.RefResolver("file://",
                                                                                               self.parameters,
                                                                                               handlers={
                                                                                                   "file": retrieve}))
        self.result = kwargs.get("result", {"type": "null"})
        self.result_validator = jsonschema.Draft4Validator(self.result,
                                                           resolver=jsonschema.RefResolver("file://",
                                                                                           self.result,
                                                                                           handlers={"file": retrieve}))
        self.errors = kwargs.get("errors", [])
        api.apis.append(self)
        apis[kwargs["endpoint"]] = self

    def __call__(self, f):
        return f

    @classmethod
    def validate_parameters(cls, endpoint, parameters):
        if endpoint in apis:
            apis[endpoint].parameters_validator.validate(parameters)

    @classmethod
    def validate_result(cls, endpoint, result):
        if endpoint in apis:
            apis[endpoint].result_validator.validate(result)


# Schema version info

def __recurse(o, s):
    r = ""
    if o is not None:
        if isinstance(o, dict):
            for key, value in o.iteritems():
                if key != "description":
                    r += key + " "
                    r += __recurse(value, s)
        elif isinstance(o, list):
            for item in o:
                r += __recurse(item, s)
        elif isinstance(o, str):
            r += o
        else:
            r += type(o).__name__
    return r + " "


def hash_schema():
    """ Computes a hash of all the schemas in the api
    """
    h = ""
    for a in refs:
        if a.enum is None:
            h += __recurse(a.properties, h)
        else:
            h += __recurse(a.enum, h)
    for a in api.apis:
        h += a.endpoint
        h += __recurse(a.parameters, h)
        h += __recurse(a.result, h)
        h += __recurse(a.errors, h)
    h = ''.join(sorted(h))
    hash_object = hashlib.md5(h)
    return hash_object.hexdigest()


