#!/usr/bin/env python

import logging
import os
import psutil
import subprocess
import xml.etree.ElementTree as ET
from os import path as osp

from tornado import httpclient, gen, web

import dispatch  # TODO: Replace with blinker
import errors
from API.enums import RecordingStatus

SCRIPTS_PATH = osp.join(os.sep, "opt", "videostitch", "bin", "sever", "config")
SCRIPT_INPUTS = osp.join(SCRIPTS_PATH, "gen_inputs.sh")
SCRIPT_OUTPUT = osp.join(SCRIPTS_PATH, "gen_output.sh")
NGINX_MONITOR = 'http://localhost:80/'


class RTMPServer:
    def __init__(self):
        self.http_client = httpclient.AsyncHTTPClient()
        self.recording_status = RecordingStatus.Stopped
        return

    def start_recording(self, drives):
        """Start recording the rtmp inputs
        """
        results = recorder.start().result()
        logging.info(results)
        success = True
        for val in results:
            success = success and results[val]
        if not success:
            raise errors.RecordingError('Starting recording inputs failed')
        self.recording_status = RecordingStatus.Started

    def stop_recording(self):
        """Stop recording the rtmp inputs
        """
        results = recorder.stop().result()
        success = True
        for val in results:
            success = success and (not results[val])
        if not success:
            raise errors.RecordingError('Stoppping recording inputs failed')
        self.recording_status = RecordingStatus.Stopped

    def is_recording(self):
        """True if the recording is enabled
        """
        return self.recording_status == RecordingStatus.Started

    def gen_inputs(self):
        """Generate the RTMP inputs
        """
        return instance.gen_inputs()

    def gen_output(self):
        """Generate RTMP output
        """
        return instance.gen_output()

    @gen.coroutine
    def get_streams_list(self):
        """Get the list of generated stream inputs (control page)
        """
        f = self.http_client.fetch(NGINX_MONITOR + 'stats.xsl')
        response = yield f
        xmlstr = response.body
        rtmp = ET.fromstring(xmlstr)
        r = []
        for application in rtmp.iter('application'):
            logging.error(application)
            app = {
                'name': application.find('name').text,
                'streams': []
            }
            for stream in application.iter('stream'):
                app['streams'].append(stream.find('name').text)

            if len(app['streams']) > 0:
                r.append(app)
        raise gen.Return(r)


class Nginx(dispatch.EventDispatcher):
    ON_RECORD_DONE = 'record_done'
    ON_PUBLISH = 'publish'
    ON_PUBLISH_DONE = 'publish_done'

    def __init__(self):
        # notifications
        dispatch.EventDispatcher.__init__(self)
        self.generating_inputs = False
        self.gen_inputs_process = None
        self.generating_output = False
        self.gen_output_process = None
        self.bin = os.getenv('NGINX_BIN', '/usr/local/nginx/sbin/nginx')
        self.pid_file = os.getenv('NGINX_PID', '/usr/local/nginx/logs/nginx.pid')
        self.conf_file = os.getenv('NGINX_CONF', '/usr/local/nginx/conf/nginx.conf')
        self._update_process()

    def _update_process(self):
        pid = self._update_pid()
        self.process = psutil.Process(pid)
        return self.process

    def _update_pid(self):
        try:
            with open(self.pid_file) as f:
                pid = int(f.read().strip('\n'))
            if psutil.pid_exists(pid):
                return pid
            else:
                return None
        except:
            return None

    def status(self):
        s = self.process.status()
        return s

    def is_running(self):
        s = self.process.is_running()
        return s

    def terminate(self):
        self.process.terminate()

    def kill(self):
        self.process.kill()

    def start(self):
        subprocess.Popen(['systemctl', 'start', 'nginx.service'])
        self._update_process()

    def reload(self):
        subprocess.Popen(['systemctl', 'reload', 'nginx.service'])
        self._update_process()

    def stop(self):
        subprocess.Popen(['systemctl', 'stop', 'nginx.service'])
        if self.process.is_running():
            self.process.terminate()

    def set_daemon(self, enabled):
        if enabled:
            subprocess.call(['systemctl', 'unmask', 'nginx.service'])
            subprocess.call(['systemctl', 'enable', 'nginx.service'])
            subprocess.call(['systemctl', 'start', 'nginx.service'])
        else:
            subprocess.call(['systemctl', 'stop', 'nginx.service'])
            subprocess.call(['systemctl', 'disable', 'nginx.service'])
            subprocess.call(['systemctl', 'mask', 'nginx.service'])

    def is_daemon(self):
        p = subprocess.Popen(['systemctl', 'status', 'nginx.service'], stdout=subprocess.PIPE)
        output = p.communicate()[0]
        p.stdout.close()
        i = output.find('enabled;')
        if i == -1:
            return False
        else:
            return True

    def get_conf_file(self):
        return self.conf_file

    def set_conf_file(self, nginx_conf):
        self.conf_file = nginx_conf
        os.environ.set('NGINX_PATH', nginx_conf)
        self.reload()

    def gen_inputs(self):
        if self.generating_inputs:
            self.gen_inputs_process.terminate()
            self.gen_inputs_process = None
            self.generating_inputs = False
            return 'stop generating inputs.'
        else:
            self.gen_inputs_process = subprocess.Popen([SCRIPT_INPUTS])
            self.generating_inputs = True
            return 'start generating inputs.'

    def gen_output(self):
        if self.generating_output:
            self.gen_output_process.terminate()
            self.gen_output_process = None
            self.generating_output = False
            return 'stop generating output.'
        else:
            self.gen_output_process = subprocess.Popen([SCRIPT_OUTPUT])
            self.generating_output = True
            return 'start generating output.'


# helper class to receive notifications and dispatch them
# see nginx rtmp module documentation for info

class NginxNotification(dispatch.Event):
    def __init__(self, name, request):
        super(NginxNotification, self).__init__()
        self.__name = name
        self.request = request


# FIXME : only nginx requests should be allowed

class NginxNotificationHandler(web.RequestHandler):
    def get(self):
        self.notify()

    def post(self):
        # FIXME
        self.notify()

    def notify(self):
        logging.warning('nginx.notification : ')
        logging.warning(self.request.arguments['call'])

        if self.request.headers['Host'] != 'localhost':
            logging.error(self.request.headers)
            self.finish()
        else:
            event = self.request.arguments['call']
            e = NginxNotification(event, self.request)
            self.finish()
            instance.dispatch(e)
            # self.finish()


# helper class to work with nginx's rtmp module record controls
# see : https://github.com/arut/nginx-rtmp-module/wiki/Control-module

class NginxRecorder(object):
    def __init__(self, server, app, stream, rec):
        self.server = server
        self.app = app
        self.stream = stream
        self.rec = rec
        self.url = None
        self.http_client = httpclient.HTTPClient()
        # if vs_server is restarted while nginx is recording
        # this will erroneously be set to False
        self.__recording = False

    def _on_fetch(self, response):

        # response to which call ?
        if 'control/record/start' in response.effective_url:
            call = 'start'
        elif 'control/record/stop' in response.effective_url:
            call = 'stop'
        if response.error:
            logging.error(response.error)

        elif response.reason == 'No Content' and response.code == 204:
            self.log_response(call, response)
            self.__recording = False

        elif response.reason == 'OK' and response.code == 200:
            if call == 'start':
                self.__recording = True
            else:
                self.__recording = False
        else:
            logging.error('Unexpected response type : ')
            self.log_response(call, response)
            self.log_state()

    def is_recording(self):
        return self.__recording

    def set_recording(self, b):
        self.__recording = b

    def start(self):
        url = '/'.join([self.server, 'control', 'record', 'start'])
        url += '?app=' + self.app
        url += '&name=' + self.stream
        url += '&rec=' + self.rec
        self.url = url
        try:
            r = self.http_client.fetch(url)
            self._on_fetch(r)
        except httpclient.HTTPError as e:
            logging.error("Cannot fetch url" + str(e))
        else:
            return True

    def stop(self):
        url = '/'.join([self.server, 'control', 'record', 'stop'])
        url += '?app=' + self.app
        url += '&name=' + self.stream
        url += '&rec=' + self.rec
        try:
            r = self.http_client.fetch(url)
            self._on_fetch(r)
        except httpclient.HTTPError as e:
            logging.error("Cannot fetch url" + str(e))
        else:
            return True

    def log_response(self, call, response):
        logging.warning(
            'NginxRecorder.' + call + '() - ' + self.url + ': ' + response.reason + ' : ' + str(response.code))

    def log_state(self):
        if self.__recording:
            logging.info(self.stream + ' recording state is ON')
        else:
            logging.info(self.stream + ' recording state is OFF')


class Recorder(object):
    def __init__(self):
        self.recorders = (NginxRecorder(NGINX_MONITOR, 'inputs', '0_0', 'dump'),
                          NginxRecorder(NGINX_MONITOR, 'inputs', '0_1', 'dump'),
                          NginxRecorder(NGINX_MONITOR, 'inputs', '1_0', 'dump'),
                          NginxRecorder(NGINX_MONITOR, 'inputs', '1_1', 'dump'))

        def _on_record_done(event, nginx):
            args = event.request.arguments
            app = args.app
            stream = args.name
            # the 'record_done' notification happens before NginxRecorder._on_fetch
            for r in self.recorders:
                if r.app == app and r.stream == stream:
                    r.set_recording(False)
                    if r.is_recording():
                        logging.warning(app + '/' + stream + ' marked as recording after nginx notified record_done')
                    break

        # handle notifications even if recording stops intempestively
        instance.addListener(Nginx.ON_RECORD_DONE, _on_record_done)

    @gen.coroutine
    def start(self):
        # http://www.tornadoweb.org/en/stable/guide/coroutines.html#parallelism
        try:
            for recorder in self.recorders:
                recorder.start()
        except:
            logging.error("Cannot start recording inputs")
        logging.info(self.status())
        raise gen.Return(self.status())

    @gen.coroutine
    def stop(self):
        try:
            for recorder in self.recorders:
                recorder.stop()
        except:
            logging.error("Cannot stop recording inputs")
        raise gen.Return(self.status())

    def status(self):
        r = {}
        for recorder in self.recorders:
            r[str(recorder.stream)] = recorder.is_recording()
        return r

    def dispose(self):
        self.recorders = None
        instance.removeListener(Nginx.ON_RECORD_DONE, self._on_record_done)


instance = Nginx()
recorder = Recorder()
