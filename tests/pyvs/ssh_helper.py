import subprocess
import ConfigParser

from pexpect import pxssh

STREAMER_IP = '10.0.0.161'
VIDEO_TIMEOUT = 1260

class RemoteStreamer(object):

    def __init__(self):
        self.ssh_connection_timeout = VIDEO_TIMEOUT
        self.stream_box = pxssh.pxssh(timeout=self.ssh_connection_timeout,
                                      options={"StrictHostKeyChecking": "no"})
        self.connect_ssh()
        self.get_current_box_ip()

    def connect_ssh(self):
        self.stream_box.login(server=STREAMER_IP,
                              username='videostitch',
                              ssh_key='/data/videostitch/.ssh/streamer.key')

    def get_current_box_ip(self):
        cmd =['ip', 'route', 'get', '1']
        raw = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        ip_route_output = raw.communicate()
        self.box_address = ip_route_output[0].split(' \n', 2)[0].split(
            ' ', -1)[-1]

    def start_inputs(self, start_inputs_script_path):
        self.stream_box.sendline('{} {}'.format(start_inputs_script_path,
                                                self.box_address))
        self.stream_box.waitnoecho()

    def stop_inputs(self, stop_inputs_script_path):
        self.stream_box.sendline('{} {}'.format(stop_inputs_script_path,
                                                self.box_address))
        self.stream_box.waitnoecho()

    def logout(self, force=True):
        self.stream_box.waitnoecho()
        self.stream_box.close(force)


class RemotePi(object):

    def __init__(self, raspberry_config_path):
        self.ssh_connection_timeout = VIDEO_TIMEOUT
        self.stream_box = pxssh.pxssh(timeout=self.ssh_connection_timeout,
                                      options={"StrictHostKeyChecking": "no"})
        self.settings = ConfigParser.ConfigParser()
        self.settings.read(raspberry_config_path)
        self.connect_ssh()

    def connect_ssh(self):
        self.stream_box.login(server=self.settings.get('pi', 'ip'),
                              username=self.settings.get('pi', 'user'),
                              ssh_key='/data/videostitch/.ssh/id_rsa')

    def play_audio(self):
        audio_path = self.settings.get('pi','audio_path')
        self.stream_box.sendline('aplay {}'.format(audio_path))
        self.stream_box.waitnoecho()

    def stop_audio(self):
        self.stream_box.sendline('pkill -9 -f aplay')
        self.stream_box.waitnoecho()

    def logout(self, force=True):
        self.stream_box.waitnoecho()
        self.stream_box.close(force)
