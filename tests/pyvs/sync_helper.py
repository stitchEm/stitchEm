import datetime
import os
from os import path as osp
import shutil
import subprocess
import time

from scipy.io import wavfile

import numpy as np
import cv2
import subprocess2

class SyncHelper(object):

    def __init__(self, raw_video_path, videos_path, sb_sync_ouputs_folder,
                 full_wave_sample, v_assets_folder, min_val, max_val):

        self.original_video_path = raw_video_path
        self.videos_path = videos_path
        self.sb_sync_ouputs_folder = sb_sync_ouputs_folder
        self.assets = self.identify_samples(v_assets_folder)
        self.wave_full_sample = full_wave_sample
        self.min_val_score = min_val
        self.max_val_score = max_val

    def run(self):
        itens_folder = self.remove_folder_from_file_list(self.videos_path)
        videos = sorted(itens_folder, key= self.get_video_dump_folder_int)

        if len(videos) != 0:
            for num in range(0, len(videos)):
                self.setup_video(videos[num])
                assert_result = self.assert_handler()
                if not assert_result:
                    return False
            return True

    def setup_video(self, videoname):
        self.video_folder_name = videoname.replace('.mp4', '')

        #create folders to handle the outputs (image frames and wav cutted)
        self.current_sync_video_folder = osp.join(self.sb_sync_ouputs_folder,
                                                  self.video_folder_name)
        self.current_audio_cutted_folder = osp.join(
            self.current_sync_video_folder, 'output_a')
        self.current_frames_cutted_folder = osp.join(
            self.current_sync_video_folder, 'output_v')
        os.makedirs(self.current_sync_video_folder)
        os.makedirs(self.current_audio_cutted_folder)
        os.makedirs(self.current_frames_cutted_folder)

        self.original_current_video_path = osp.join(self.original_video_path,
                                                    videoname)
        self.current_video_path = osp.join(self.videos_path, videoname)

        self.encoded_video_path = osp.join(self.sb_sync_ouputs_folder,
                                           "encoded_{}".format(videoname))

        self.audio_cutted_wav = osp.join(self.current_audio_cutted_folder,
                                         "cut.wav")

    def before_match(self):
        self.generate_audio_sample(self.current_video_path,
                                   self.audio_cutted_wav)
        self.dump_beep_location = self.get_dump_beep_location(
            self.audio_cutted_wav)
        self.dump_frame_number = self.get_video_frame(self.dump_beep_location)

        # Sometimes when cutting the grey area, the script also cut the
        # possible sound mark, thats why if it fails to find the it, then the
        # script will encode de video again on the except.

        try:
            self.beep_timestamp = self.detect_beep(self.wave_full_sample,
                                                   self.audio_cutted_wav)
            self.generate_video_frames(self.current_video_path,
                                       self.current_frames_cutted_folder)
        except:
            if not self.encoded_video:
                self.switch_to_encoded_video()

    def switch_to_encoded_video(self):
        self.re_encode_dumpedvideo(self.original_current_video_path,
                                   self.encoded_video_path)
        self.reset_for_encoded_video()
        self.dump_beep_location = self.get_dump_beep_location(
            self.audio_cutted_wav)
        self.dump_frame_number = self.get_video_frame(self.dump_beep_location)
        self.beep_timestamp = self.detect_beep(self.wave_full_sample,
                                               self.audio_cutted_wav)
        self.encoded_video = True

    def assert_handler(self):
        self.encoded_video = False
        match = False
        self.before_match()

        for item in self.assets:
            if round(self.beep_timestamp[0]) == round(item['audio second']):
                match = self.match_video_frame(
                    self.dump_frame_number,
                    self.current_frames_cutted_folder,
                    item['video frame template'])
                if not match and not self.encoded_video:
                    self.switch_to_encoded_video()
                    match = self.match_video_frame(
                        self.dump_frame_number,
                        self.current_frames_cutted_folder,
                        item['video frame template'])
        if not match:
            print("@@@@@@@@@@@@@@@@")
            print("Beep Found at {} seconds".format(self.beep_timestamp))
            print("Image mark suppose to be around frame {}".format(
                self.dump_frame_number))

        return match

    def reset_for_encoded_video(self):
        shutil.rmtree(self.current_audio_cutted_folder, ignore_errors=True)
        shutil.rmtree(self.current_frames_cutted_folder, ignore_errors=True)

        os.makedirs(self.current_audio_cutted_folder)
        os.makedirs(self.current_frames_cutted_folder)

        self.generate_audio_sample(self.encoded_video_path,
                                   self.audio_cutted_wav)
        self.generate_video_frames(self.encoded_video_path,
                                   self.current_frames_cutted_folder)

    def remove_folder_from_file_list(self, folder_path):
        itens = []
        for item in os.listdir(folder_path):
            if not os.path.isdir(osp.join(folder_path,item)):
                itens.append(item)
        return itens

    def get_dump_beep_location(self, beep_file):
        [rate, beep] = wavfile.read(beep_file)
        beep = beep[:, 1].astype(float)

        beep_trim =  self.trim(beep, rate)
        return beep_trim[1]

    def trim(self, beep, rate):
        y = np.abs(beep)
        imax = np.argmax(y, axis=0)
        max_beep_duration = 0.060
        threshold = 0.4

        max_beep_len = int(round(max_beep_duration * rate))
        beep_begin = max(0, imax - max_beep_len)
        beep_end = min(len(y), imax + max_beep_len)

        if y[imax] > threshold:
            y2 = beep[0:beep_begin]

            if len(y2) > 0:
                imax2 = np.argmax(y2, axis=0)
                if y2[imax2] > threshold and imax2 < imax:
                    beep_begin = max(0, imax2 - max_beep_len)
                    beep_end = min(len(y), imax2 + max_beep_len)
                    imax = imax2

        return beep[beep_begin:beep_end], float(imax) / float(rate)

    def find_next_max(self, a, begin):
        threshold = 0.4
        max_val = 0
        pos_max = 0
        if begin >= len(a):
            return [max_val, begin]

        for i in range(begin, min(len(a), begin + 2000)):
            if a[i] > threshold and a[i] > max_val:
                max_val = a[i]
                pos_max = i
            elif pos_max > begin:
                return [max_val, pos_max]

        return [0, min(len(a), begin + 2000)]

    def window_rms(self, a, window_size):
        a2 = np.power(a, 2)
        window = np.ones(window_size) / float(window_size)
        return np.sqrt(np.convolve(a2, window, 'valid'))

    def detect_beep(self, search_file, beep_file):
        """
        :type beep_file: file path to the beep
        :type search_file: file path to the search file
        """
        [rate, beep] = wavfile.read(beep_file)
        beep = beep[:, 1].astype(float)
        beep_trim = self.trim(beep, rate)
        beep = beep_trim[0]
        len_beep = len(beep)

        [rate, full_wave] = wavfile.read(search_file)
        left = full_wave[:, 1].astype(float)

        cc_beep = np.correlate(left, beep, "valid")
        cc_beep_abs = cc_beep * cc_beep
        m_beep = np.max(cc_beep_abs)
        cc_beep_norm = cc_beep_abs / m_beep

        win_rms_size = 1000
        cc_beep_r_m_s = self.window_rms(cc_beep_norm, win_rms_size)

        last_pos_max = 0
        i_max = 0
        cur_pos = 0

        beep_pos = np.array([])
        while cur_pos < (len(cc_beep_r_m_s)):
            [val, pos] = self.find_next_max(cc_beep_r_m_s, cur_pos)
            if val > 0.25:
                if (pos - last_pos_max) < len_beep:
                    beep_pos[i_max - 1] = pos
                else:
                    beep_pos = np.append(beep_pos, pos)
                    i_max += 1

                last_pos_max = pos
                cur_pos = last_pos_max + 1
            else:
                cur_pos = pos + 1
        beep_pos -= len_beep / 2

        return beep_pos / rate

    def match_video_frame(self, f_number, frames_folder, template_img):
        template_image = cv2.imread(template_img, 0)

        # As audio is more precise than image, it will get some frames back
        # and ahead of the frame. Those frames will not impact on the sync

        if f_number <= 0:
            return False

        frame_images_list = []
        for number in range(0,7):
            frames_ahead = f_number + number
            frames_behind = f_number - number

            if frames_behind <= 1:
                frames_behind = 1
            if frames_ahead > len(os.listdir(frames_folder)):
                frames_ahead = len(frames_folder)

            frame_images_list.append(osp.join(
                frames_folder,'{}.jpg'.format(str(frames_ahead))))
            frame_images_list.append(osp.join(
                frames_folder,'{}.jpg'.format(str(frames_behind))))

        for image in frame_images_list:
            if sample_image_matchmaking(image, template_image,
                                        self.max_val_score,
                                        self.min_val_score):
                return True
        return False

    def get_video_frame(self, seconds):
        fps = 30
        return int(round(seconds*fps,0))

    def get_video_dump_folder_int(self, x):
        return int(x.split('.')[1])

    def get_assets_folder_int(self, x):
        return int(x.split('.')[0])

    def identify_samples(self, v_assets_folder):
        assets = []
        audio_assets = [{'index': 1, 'second': 5.46614512},
                        {'index': 2, 'second': 20.47197279},
                        {'index': 3, 'second': 35.46331066},
                        {'index': 4, 'second': 50.4675737},
                        {'index': 5, 'second': 65.4661678},
                        {'index': 6, 'second': 80.46643991},
                        {'index': 7, 'second': 95.46614512},
                        {'index': 8, 'second': 110.4661678},
                        {'index': 9, 'second': 125.46639456},
                        {'index': 10, 'second': 140.46646259},
                        {'index': 11, 'second': 155.46634921}]

        video_assets = sorted(os.listdir(v_assets_folder),
                              key= self.get_assets_folder_int)

        for i in range(0,len(audio_assets)):
            assets.append({
                'audio index' : audio_assets[i]['index'],
                'audio second' : audio_assets[i]['second'],
                'video frame template' : osp.join(v_assets_folder,
                                                  video_assets[i])
                })
        return assets

    def generate_video_frames(self, file_input, output_frames_folder):
        cmd = ['ffmpeg', '-loglevel', 'quiet', '-i', file_input, '-vf',
               'fps=30', '{}/%d.jpg'.format(output_frames_folder)]
        subprocess.check_call(cmd)

    def generate_audio_sample(self, video_file_input, sample_out_audio_path):
        cmd = ['ffmpeg', '-loglevel', 'quiet', '-y', '-i', video_file_input,
               '-acodec', 'pcm_s16le', '-ac', '2', sample_out_audio_path]
        subprocess.check_call(cmd)

    def re_encode_dumpedvideo(self, video_path, encoded_video_path):
        """
        Acordding to the ffmpeg docs, if you specify a second on the cut and
        there is no key frame until x+ seconds, it will include y seconds of
        audio (with no video) at the start, then will start from the first key
        frame.

        This method will re-encode the video (takes longer) but will fix this
        problem. Only use this method whem the fast and normal one fails
        """
        command = [
            'ffmpeg', '-loglevel', 'quiet', '-i', video_path, '-to',
            '00:00:13', '-strict', '-2', '-async', '1', encoded_video_path]
        subprocess.check_call(command)
        time.sleep(1)


def dump_video_from_stream(nro_of_dumps, rtmp_address, tempdumpvideopath,
                           dump_videopath):
    for _ in xrange(nro_of_dumps):
        datastr = str(datetime.datetime.now()).replace(':','-')
        datastr = datastr.replace(' ', '')
        filename = 'videodump_{}.mp4'.format(datastr)

        video_path = osp.join(tempdumpvideopath, filename)

        use_rtmpdump(rtmp_address, video_path)

        new_video_path = osp.join(dump_videopath, filename)

        cut_gray_dumped_video(video_path, new_video_path)


def cut_gray_dumped_video(video_path, new_video_path):
    """
    This method cut the first 10 seconds of the video raw video that are
    usally gray. For a better result, the video should be re encoded to not
    miss INDEX frames but in overall performance, this is a fast way to have a
    video that fit most all of the scenarios.
    For fallback check, use re_encoded_dumpedvideo
    """
    command = ['ffmpeg', '-loglevel', 'quiet', '-ss', '00:00:10', '-i',
               video_path, '-to', '00:00:25', '-c:v', 'copy', '-c:a', 'copy',
               '-async', '1', '-y', new_video_path]
    subprocess.check_call(command)
    time.sleep(1)


def use_rtmpdump(rtmp_address, tempdumpvideopath):
    command = ['rtmpdump', '-r', rtmp_address, '-o', tempdumpvideopath ]
    pipe = subprocess2.Popen(command, stdout=subprocess2.PIPE,
                             stderr=subprocess2.PIPE)
    pipe.waitOrTerminate(30)


'''
The image matchmaking use OPENCV2 matchTemplate. This method have different
behavior rengarding the size of the image.

If the template that is the same size of the image you want to compare, then it
requires both to be converted to a numpy array and the result of the method
minMaxLoc will be the same value for the min_val and max_val. You should use the
method full_image_matchmaking

If the template is smaller, than the matchTemplate can handle by its own and the
result will be a different min_val and max_val.You should use the
method sample_image_matchmaking
'''

def full_image_matchmaking(image, template, min_score_val):
    img = cv2.imread(image, 0)
    template_image = cv2.imread(template, 0)

    type_match = eval('cv2.TM_CCOEFF_NORMED')
    check_img = cv2.matchTemplate(img, template_image, type_match)

    min_val_sample, max_val_sample, _, _ = cv2.minMaxLoc(check_img)
    return min_val_sample >= min_score_val and max_val_sample >= min_score_val

def sample_image_matchmaking(image, template, min_val, max_val):
    img = cv2.imread(image, 0)

    type_match = eval('cv2.TM_CCOEFF_NORMED')
    check_img = cv2.matchTemplate(img, template, type_match)

    min_val_sample, max_val_sample, _, _ = cv2.minMaxLoc(check_img)
    return min_val_sample <= min_val and max_val_sample >= max_val

