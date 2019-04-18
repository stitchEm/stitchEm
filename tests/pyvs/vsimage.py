import re
import subprocess
import sys
import os.path as osp

class Video(object):
    def __init__(self, path, max_frames=100, pic_name="frame-%03d.jpeg"):
        self.path = path
        self.max_frames = max_frames
        self.pic_name = pic_name

        def number_of_frames(path):
        # http://superuser.com/questions/84631/how-do-i-get-the-number-of-frames-in-a-VIDEO-on-the-linux-command-line
            cmd = 'NUL' if sys.platform == 'win32' else '/dev/null'
            output = subprocess.check_output(
                "ffprobe -select_streams v -show_streams {} ".format(path) +\
                    "2>{} | grep nb_frames | sed -e 's/nb_frames=//'".format(
                        cmd),
                shell=True)
            return int(output)

        def fps(path):
            return float(subprocess.check_output(
                r"ffprobe {} 2>&1 | sed -n 's/.*, \(.*\) fp.*/\1/p'".format(
                    path),
                shell=True,
            ))

        self.nb_frames = number_of_frames(path)
        self.fps = fps(path)
        self.duration = self.max_frames / self.fps

    def write_frames(self, file_output):
        # output all frames til MAX_FRAME from ffmpeg
        subprocess.check_output(
            "ffmpeg -i {} -t {} -f image2 -q:v 1 {}/{}".format(
                self.path, self.duration, file_output, self.pic_name),
            shell=True,
        )

def diff_image(im1, im2):
    res = subprocess.Popen(["compare", "-metric", "MAE", im1, im2, ":null"],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    res = res.stdout.read()
    return float(re.sub(r'.*\((.*)\)', r'\1', res))

if __name__ == "__main__":
    if len(sys.argv) == 3:
        FILE = "frame-%03d.jpeg"
        NB = 51
        VIDEO_FILE = sys.argv[1]
        print(VIDEO_FILE)
        OUTPUT_FOLDER = sys.argv[2]
        VIDEO = Video(path=VIDEO_FILE, max_frames=NB,
                      pic_name=FILE)
        print "fps:", VIDEO.fps
        print "number of frames:", VIDEO.nb_frames
        VIDEO.write_frames(OUTPUT_FOLDER)
        BASE = osp.join(OUTPUT_FOLDER, "frame-012.jpeg")
        for i in xrange(NB):
            a = diff_image(BASE, osp.join(OUTPUT_FOLDER, FILE % (i + 1)))
            print i, a
    else:
        print("Example usage: 'python seekframe.py input_VIDEO.mp4 " +\
            "/tmp/OUTPUT_FOLDER")
