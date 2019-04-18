import os
from setuptools import setup

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name="libvideostitch-python",
    version="2.3.0",
    author="VideoStitch SAS",
    author_email="nlz@video-stitch.com",
    description=("Python bindings for VideoStitch video stitching library."),
    url="http://video-stitch.com",
    scripts=['samples/cmd/vs_cmd.py', 'samples/server/vs_server.py'],
    install_requires=["tornado","psutils"],
    packages=['vs'],
    package_data={'vs': ['_vs.so',
                         'libvideostitch.so',
                         'libpng16.so.16',
                         'libopencv_core.so.3.0',
                         'libopencv_calib3d.so.3.0',
                         'libopencv_ml.so.3.0',
                         'libopencv_features2d.so.3.0',
                         'libopencv_flann.so.3.0',
                         'libopencv_imgproc.so.3.0',
                         'libopencv_video.so.3.0',
                        ]},
    )
