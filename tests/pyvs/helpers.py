import json
import subprocess

import numpy as np
import psutil

import vswave

SHORT_FFPROBESIZE = "100000"
LONG_FFPROBESIZE = "15000000"

def ensure_stream_is_closed(stream):
    while True:
        try:
            subprocess.check_output(
                ['ffprobe', '-probesize', '100000',  stream])
        except subprocess.CalledProcessError:
            break


def get_ffprobe_video_outputs(probsize_bytes, file_or_stream, *args):
    """ Method that return the formats and the VIDEO stream from a file as DICT.
        params:
                probsize_bytes: ffprobe byte size. usually SHORT_FFPROBESIZE or
                                                  LONG_FFPROBESIZE
                file_or_stream: It can probe a RTMP stream or a local file.
                                (for files you have to provide the full path)
                *args:  Anothers ffprobe arguments you might need.
    """
    return __ffprobe_raw_subprocess(
        file_or_stream, '-probesize', probsize_bytes, '-show_format',
        '-show_streams', '-select_streams', 'v', '-read_intervals', '%+03')


def get_ffprobe_audio_outputs(probsize_bytes, file_or_stream, *args):
    """ Method that return the formats and the AUDIO stream from a file as DICT.
        params:
                probsize_bytes: ffprobe byte size. usually SHORT_FFPROBESIZE or
                                                  LONG_FFPROBESIZE
                file_or_stream: It can probe a RTMP stream or a local file.
                                (for files you have to provide the full path)
                *args:  Anothers ffprobe arguments you might need.
    """

    return __ffprobe_raw_subprocess(
        file_or_stream, '-probesize', probsize_bytes, '-show_format',
        '-show_streams', '-select_streams', 'a', '-read_intervals', '%+03')


def record_audio_sample_from_broadcasting(stream, output_sample_file_location,
                                          seconds_of_recording):
    """ Method that record a audio sample on a from the stream."""
    cmd = ['ffmpeg', '-y', '-i', stream, '-t' , str(seconds_of_recording),
           '-vn', output_sample_file_location]
    subprocess.check_call(cmd)


def get_wave_from_video_file(video_file_input, sample_out_audio_path):
    cmd = ['ffmpeg', '-loglevel', 'quiet', '-y', '-i', video_file_input,
           '-vn', sample_out_audio_path]
    subprocess.check_call(cmd)


def record_audio_mkv_sample_from_broadcasting(
        stream, output_mkv_sample_file_location, seconds_of_recording):
    """ Method that record a audio sample on a from the stream."""
    cmd = [
        "ffmpeg", "-i", stream, "-t" , seconds_of_recording,
        "-analyzeduration", "4000000", "-c:v", "copy", "-c:a", "copy",
        output_mkv_sample_file_location]
    subprocess.check_call(cmd)


def get_sox_audio_stat(path):
    """ Method that return Sounds infos like Maximum and Minimum amplitude,
        RMS and DC offset.
    """
    cmd = ['/usr/bin/sox', path ,'-n' ,'stat']
    raw = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE)
    array = filter(None, raw.communicate()[1].split("\n"))

    if array[-1] == 'Probably text, not sound':
        raise Exception("There is no audio on the file")

    sox_output = dict()

    for item in array:
        if ':' in item:
            item,valor = item.split(':')
            sox_output[item] = valor.strip()
    return sox_output


def validate_audio_is_not_silent(path):
    sox_output = get_sox_audio_stat(path)
    assert float(sox_output['Maximum amplitude']) > 0.1


def validate_audio_is_silent(path):
    sox_output = get_sox_audio_stat(path)
    assert float(sox_output['Maximum amplitude']) < 0.14


def play_sound(audio_playback_file, hardware_identifier=None):
    if hardware_identifier:
        cmd = ['aplay', '-D{}'.format(hardware_identifier),
               audio_playback_file]
    else:
        cmd = ['aplay', audio_playback_file]
    subprocess.Popen(cmd, stdin=None, stdout=None, close_fds=True)


def get_effective_bitrate(byte, stream, interval):
    output = __ffprobe_raw_subprocess(
        stream, "-probesize", byte, "-show_packets", "-select_streams", "v",
        "-read_intervals", "%+{}".format(interval))

    assert output.has_key("packets")

    total_duration = 0.0
    total_size = 0.0
    nb = 0

    for frame in output["packets"]:
        if frame.has_key("size"):
            total_size += long(frame["size"])
            nb += 1
        if frame.has_key("duration_time"):
            total_duration += float(frame["duration_time"])

    if nb == 0:
        raise Exception("no packets or missing size and duration in all of them")

    # in case duration_time return N/A, interval is a valid estimation
    if total_duration == 0:
        total_duration = float(interval)

    # total_size is in bytes and total_duration in s
    # bitrate is in kilobits per second
    # ((total_size * 8) / 1000) / (total_duration)
    return int(total_size / (total_duration * 125))


def get_frames_info(path, probsize_bytes):
    """ Retrive list of dicts containg  frames infos from a file.
        the frame list is positional, so the first frame will be the first
        item of the list.

        Example on How to use: object[0]['key_frame']
    """
    return __ffprobe_raw_subprocess(
        path, "-probesize", probsize_bytes, "-select_streams", "v",
        "-show_frames")["frames"]


def assert_gop_from_sample(frames, gop_size):
    for index, item in enumerate(frames):
        if index % gop_size == 0:
            assert item['key_frame'] == 1
            assert item['pict_type'] == "I"
        else:
            assert item['key_frame'] != 1
            assert item['pict_type'] != "I"


def assert_b_frames_pattern_from_sample(frames, b_frames_qty, gop_size,
                                        profile):
    """ Method that checks the position of b_frames and the max times it
        appear each time before a P or I frame.
    """

    max_b_frames_on_file = 0
    for index, item in enumerate(frames):
        if item['pict_type'] == "I":
            b_count = 0
        elif item['pict_type'] == "P":
            if index != len(frames) - 1 and (index + 1) % gop_size != 0 and\
                    profile != 'baseline':
                assert b_count == int(b_frames_qty)
                b_count = 0
        else:
            assert item['pict_type'] == "B"
            b_count += 1

        if b_count > max_b_frames_on_file:
            max_b_frames_on_file = b_count
    if profile == 'baseline':
        assert max_b_frames_on_file == 0
    else:
        assert max_b_frames_on_file == b_frames_qty


def __ffprobe_raw_subprocess(file_or_stream, *args):
    cmd = ['ffprobe', '-print_format', 'json', '-loglevel', 'quiet', "-i",
           file_or_stream]
    cmd.extend(args)
    return json.loads(subprocess.check_output(cmd))


def take_video_screen_shot(file_or_stream, output_file):
    cmd = ['ffmpeg', '-i', file_or_stream, '-vframes', '1','-y', output_file]
    subprocess.check_call(cmd)


def stop_vs_cmd():
    for process in psutil.process_iter():
        try:
            pname = process.name()
        except psutil.ZombieProcess:
            continue
        if "videostitch-cmd" in pname:
            process.terminate()
            process.wait()


def gen_win(n):
    """
    Generate an n-point Hann window
    """
    win = []
    for i in xrange(n):
        v = 0.5 - (0.5 * np.cos(2 * np.pi * i / (n - 1)))
        win.append(v)
    return win

def get_audio_channel_map(filename, f0):
    """
    Extract the audio of the video file to a wavfile and
    check the channel map of the wav file
    """
    if filename.find("wav") == -1:
        wavfile = filename.split('.')[0] + '.wav'
        cmd = ['ffmpeg', '-y', '-i', filename, '-vn', wavfile]
        subprocess.check_call(cmd)
    else:
        wavfile = filename

    x, info = vswave.read_wav(wavfile)
    rate = info['sample_rate']
    duration = info['duration']
    show_time = int(duration * rate / 2)
    n_fft = 512
    nb_chan, _ = np.shape(x)
    x = x[:, show_time: show_time + n_fft * 20]
    n_fft = np.shape(x)[1]
    win = gen_win(n_fft)
    freq = np.linspace(0, float(rate), n_fft)
    y_fft = np.empty([nb_chan, n_fft], dtype=np.complex)

    channel_map = []
    for i in xrange(nb_chan):
        y_fft[i] = np.fft.fft(np.multiply(x[i], win))
        i_max = np.argmax(y_fft[i][0:n_fft/2])
        f_max = freq[i_max]
        channel_map.append(int(np.round(f_max/f0))-1)

    return channel_map
