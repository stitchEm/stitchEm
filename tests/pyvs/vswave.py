import struct
import subprocess
import wave

import numpy as np

WAVE_EXTENDED = 65534
WAVE_FORMAT_PCM = 1
WAVE_FORMAT_FLOAT = 3

# SAMPLE FORMAT
SAMPLE_FORMAT_S8  = 1
SAMPLE_FORMAT_S16 = 2
SAMPLE_FORMAT_S24 = 3
SAMPLE_FORMAT_S32 = 4
SAMPLE_FORMAT_F32 = 5
SAMPLE_FORMAT_F64 = 6

def get_sample_format_string(sample_format):
    if sample_format == SAMPLE_FORMAT_S8:
        return 'int8'
    elif sample_format == SAMPLE_FORMAT_S16:
        return 'int16'
    elif sample_format == SAMPLE_FORMAT_S24:
        return 'int24'
    elif sample_format == SAMPLE_FORMAT_S32:
        return 'int32'
    elif sample_format == SAMPLE_FORMAT_F32:
        return 'flt32'
    elif sample_format == SAMPLE_FORMAT_F64:
        return 'flt64'
    else :
        raise Exception('Sample format {} not managed'.format(sample_format))


def get_sample_type(sample_format):
    if sample_format == SAMPLE_FORMAT_S8:
        return 'b'
    elif sample_format == SAMPLE_FORMAT_S16:
        return 'h'
    elif sample_format == SAMPLE_FORMAT_S24:
        raise Exception('Sample format s24 not managed')
    elif sample_format == SAMPLE_FORMAT_S32:
        return 'i'
    elif sample_format == SAMPLE_FORMAT_F32:
        return 'f'
    elif sample_format == SAMPLE_FORMAT_F64:
        return 'd'
    else:
        raise Exception('Sample format {} not managed'.format(sample_format))

def get_sample_scale_factor(sample_format):
    if sample_format == SAMPLE_FORMAT_S8:
        return float(pow(2, 7))
    elif sample_format == SAMPLE_FORMAT_S16:
        return float(pow(2, 15))
    elif sample_format == SAMPLE_FORMAT_S24:
        raise Exception("Sample format s24 not managed")
    elif sample_format == SAMPLE_FORMAT_S32:
        return float(pow(2, 23))
    elif sample_format == SAMPLE_FORMAT_F32:
        return 1.0
    elif sample_format == SAMPLE_FORMAT_F64:
        return 1.0
    else:
        raise Exception('Sample format {} not managed'.format(sample_format))

def get_sample_size_in_bytes(sample_format):
    if sample_format == SAMPLE_FORMAT_S8:
        return 1
    elif sample_format == SAMPLE_FORMAT_S16:
        return 2
    elif sample_format == SAMPLE_FORMAT_S24:
        return 3
    elif sample_format == SAMPLE_FORMAT_S32:
        return 4
    elif sample_format == SAMPLE_FORMAT_F32:
        return 4
    elif sample_format == SAMPLE_FORMAT_F64:
        return 8
    else:
        raise Exception('Sample format {} not managed'.format(sample_format))


def get_sample_format(format_tag, bits_per_sample):
    if format_tag == WAVE_FORMAT_PCM:
        if bits_per_sample == 8:
            return SAMPLE_FORMAT_S8
        elif bits_per_sample == 16:
            return SAMPLE_FORMAT_S16
        elif bits_per_sample == 24:
            return SAMPLE_FORMAT_S24
        elif bits_per_sample == 32:
            return SAMPLE_FORMAT_S32
        else:
            raise Exception(
                'Unexpected bits per sample {}'.format(bits_per_sample))
    elif format_tag == WAVE_FORMAT_FLOAT:
        if bits_per_sample == 32:
            return SAMPLE_FORMAT_F32
        elif bits_per_sample == 64:
            return SAMPLE_FORMAT_F64
        else:
            raise Exception(
                'Unexpected bits per sample {}'.format(bits_per_sample))

def read_fmt_chunk(chunk):
    _, w_format_tag, n_chan, sample_rate, \
    _, _, bits_per_sample = struct.unpack('<IHHIIHH', chunk[0:20])
    if w_format_tag == WAVE_EXTENDED:
        _, valid_bits_per_sample, _ = struct.unpack('<HHI', chunk[20:28])
        guid0, dummy = struct.unpack('<I2s', chunk[28:34])
        sample_format = get_sample_format(guid0, valid_bits_per_sample)
    else:
        sample_format = get_sample_format(w_format_tag, bits_per_sample)
    return sample_rate, n_chan, bits_per_sample, sample_format

def read_wav(f_name):
    """
    Import data form WAV file
    """
    with open(f_name, "rb") as f:
        f.read(12)  # Skip 'RIFF', file size, and 'WAVE'
        if str(f.read(4)) != 'fmt ':
            raise Exception("Expected to find 'fmt '!")

        sample_rate, n_chan, bits_per_sample, sample_format = read_fmt_chunk(
            f.read(34))
        sample_size = bits_per_sample / 8
        sample_type = get_sample_type(sample_format)
        sample_scale_factor = get_sample_scale_factor(sample_format)
        field = str(f.read(2))
        while field != 'da':
            field = str(f.read(2))
            if not field:
                raise Exception("Got to EOF without finding 'da' section!")
        field = str(f.read(2))
        if field != 'ta':
            raise Exception("Got to EOF without finding 'ta' section!")

        data_len = struct.unpack('i', f.read(4))[0] / (sample_size * n_chan)
        wave_info = {'sample_rate' : sample_rate,
                     'nb_channels': n_chan,
                     'sample_format': get_sample_format_string(sample_format),
                     'duration': float(data_len)/float(sample_rate)}
        out = np.empty([n_chan, data_len], dtype=np.double)
        for sample in xrange(data_len):
            for channel in xrange(n_chan):
                float_ = struct.unpack(sample_type, f.read(sample_size))[0]
                out[channel][sample] = float(float_) / sample_scale_factor

        return out, wave_info


def get_main_frequency(wave_file):
    # Analyze the file and get useful constants
    wav = wave.open(wave_file, "rb")
    sampling_rate = wav.getframerate()
    nb_channels = wav.getnchannels()
    step = 1. / sampling_rate
    duration = 1.0 # we analyze the first second of the file
    nb_samples = int(duration / step)

    # Read the file and do the FFT
    signal = wav.readframes(nb_samples)
    signal = np.fromstring(signal, 'Int16')
    signal = signal[::nb_channels]
    signal_fft = abs(np.fft.fft(signal))
    signal_fft = signal_fft.tolist()
    freq = np.fft.fftfreq(nb_samples, d=step)

    # Extract the peak frequency
    max_index = signal_fft.index(max(signal_fft))
    return abs(freq[max_index])

def extract_audio(video_file, output):
    subprocess.check_call(
        ["ffmpeg", "-i", video_file, "-acodec", "pcm_s16le", "-ac", "2",
         output])

def get_gain_peaks(wave_file, duration=60):
    wav = wave.open(wave_file, "rb")
    sampling_rate = wav.getframerate()
    nb_channels = wav.getnchannels()
    step = 1. / sampling_rate
    nb_samples = int(duration / step)

    signal = wav.readframes(nb_samples)
    signal = np.fromstring(signal, 'Int16')
    signal = signal[::nb_channels]

    res = []
    i = 0
    max_zeros = 1000
    zeros = 0
    gain_trigger = 10000
    while i < nb_samples:
        peak = 0
        gain = signal[i]
        peak_start_time = 0
        while (gain > gain_trigger or zeros <= max_zeros) and i < nb_samples:
            time = i * step
            if peak_start_time == 0:
                peak_start_time = time
            if gain > peak:
                peak = gain
            i += 1
            if i == nb_samples:
                break
            gain = signal[i]
            if abs(gain) <= 1:
                zeros += 1
            else:
                zeros = 0
        if peak != 0:
            res.append(peak_start_time)
        i += 1
    return res

