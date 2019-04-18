#!/bin/bash

VERBOSE=warning
FPS=30
LOCAL_SERVER=`ip route get 1 | awk '{print $NF;exit}'`
PORT=1935
STREAM=inputs/
FF=ffmpeg
INPUT=/home/videostitch/SB_TESTS/
ISUF=_short.mp4

$FF -v $VERBOSE -re \
    -i ${INPUT}0_0${ISUF}  \
    -i ${INPUT}0_1${ISUF}  \
    -i ${INPUT}1_0${ISUF}  \
    -i ${INPUT}1_1${ISUF}  \
    -r ${FPS} -f flv -c:v copy -c:a aac -ar 44100 -strict -2 rtmp://${LOCAL_SERVER}:${PORT}/${STREAM}0_0 \
    -r ${FPS} -f flv -c:v copy -an rtmp://${LOCAL_SERVER}:${PORT}/${STREAM}0_1 \
    -r ${FPS} -f flv -c:v copy -c:a aac -ar 44100 -strict -2 rtmp://${LOCAL_SERVER}:${PORT}/${STREAM}1_0 \
    -r ${FPS} -f flv -c:v copy -an rtmp://${LOCAL_SERVER}:${PORT}/${STREAM}1_1 &
