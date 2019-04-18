#!/usr/bin/env python

import inspect
import json
import logging
import schema
import jsonschema

from concurrent.futures import Future
import tornado
import tornado.gen
import tornado.web
from tornado.web import RequestHandler, StaticFileHandler

import errors


class BaseHandler(RequestHandler):
    def data_received(self, chunk):
        raise NotImplementedError()


class IndexPageHandler(BaseHandler):
    def get(self):
        self.render(MAIN_INDEX, status="Ready", info="Retrieving data ...")


class StaticTextFileHandler(StaticFileHandler):
    def set_default_headers(self):
        self.set_header('Content-Type', 'text/plain')


class DevStaticFileHandler(StaticFileHandler):
    def set_extra_headers(self, path):
        # Disable cache
        self.set_header(
            'Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0')


logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class CommandHandler(tornado.web.RequestHandler):
    """REST Command Handler

    Routes json command in the form
    {'name' : 'apiHandler.function', 'parameters' : {}}
    to the right apiHandler instance function
    (see https://developers.google.com/streetview/open-spherical-camera/
    for an overview of the protocol)

    This handler supports synchronous and asyncronous calls. Asynchronous API
    methods must be decorated with @run_on_executor and the enclosing class
    must provide an 'executor' ThreadPoolExecutor class instance.

    Example:::

        class MyAPIHandler(APIHandler):
            executor = ThreadPoolExecutor(1)

            @run_on_executor
            def asynchronousAPI (self, parameters) :
                return

            def synchronousAPI (self, parameters) :
                return

    """

    def initialize(self, apiHandlers, extra):
        self.apiHandlers = apiHandlers
        self.extra = extra
        self.verbose = extra["verbose"]

    def set_default_headers(self):
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Headers", "x-requested-with")
        self.set_header('Access-Control-Allow-Methods', 'POST')

    @tornado.gen.coroutine
    def post(self):
        try:
            data = json.loads(self.request.body)
        except ValueError:
            raise tornado.web.HTTPError(400, "Ill-formed message command")

        # Split command name
        command_name = data.get("name")
        if command_name is None:
            raise tornado.web.HTTPError(400, "No command name")
        if command_name.count(".") != 1:
            raise tornado.web.HTTPError(400, "Invalid command name")
        (class_name, function_name) = command_name.split(".")
        # Create a handler instance
        class_instance = self.apiHandlers.get(class_name)
        if class_instance is None:
            raise tornado.web.HTTPError(400, "Unknown handler " + class_name)
        instance = class_instance(self.extra)

        # Validate parameters
        validate = data.get("validate", True)
        log = data.get("log", True)
        try:
            parameters = data.get("parameters")
            if self.verbose and log:
                logger.info(
                    "> " + class_name + "." + function_name + "(" + \
                    json.dumps(parameters) + ")")
            if validate:
                schema.api.validate_parameters(command_name, parameters)
        except jsonschema.ValidationError as ve:
            e = errors.InvalidParameter(str(ve)).payload
            m = {'error': e}
            logger.error(command_name + ", invalid parameter:\n" + str(ve))
        else:
            # Call instance method
            if not hasattr(instance, function_name):
                raise tornado.web.HTTPError(
                    400, "Unknown function " + command_name)
            # call method
            function = getattr(instance, function_name)
            future = function(parameters)
            try:
                if isinstance(future, Future) or isinstance(
                        future, tornado.concurrent.Future):
                    result = yield future
                else:
                    f = Future()
                    f.set_result(future)
                    result = yield f
            except errors.VSError as e:
                m = {'error': e.payload}
            else:
                # Validate result
                try:
                    if validate:
                        schema.api.validate_result(command_name, result)
                except jsonschema.ValidationError as ve:
                    e = errors.InvalidReturnValue(str(ve)).payload
                    m = {'error': e}
                    logger.error(command_name + ", invalid return value:\n" + str(ve))
                else:
                    m = {'results': result}

        if self.verbose and log:
            logger.info("< " + json.dumps(m))
        self.write(m)
        self.finish()


class APIHandler(object):
    """Base class for APIHandlers
    """

    def __init__(self, extra):
        pass

    def get_methods(self):
        """Get the list of exposed methods for the API
        Returns:
            A list of methods available.
        """
        members = []
        for member in inspect.getmembers(self, predicate=inspect.ismethod):
            if member[0] not in ["__init__", "ok", "error"]:
                members.append(member[0])
        return members
