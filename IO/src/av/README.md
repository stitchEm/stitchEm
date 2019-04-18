# AV plugin documentation

`av` is an IO plugin. It allows to capture video input/output from/to video files (like mp4 files)
and to capture input from RTSP streams.

## AV Input Configuration

The av plugin can be used by Vahana VR throught a .vah project file. Please see the `*.vah file format
specification` for additional details.

Define an input for each camera. The `reader_config` member specifies how to read
it.

### Example

For a video file input :

	"inputs" : [
    {
      "width" : 2560,
      "height" : 2048,
      ...
      "reader_config" : "C:\\Users\\VideoStitch\\Vahana VR\\Projects\\test.mp4",
      ...
    }]

For an RTSP stream input:

	"inputs" : [
    {
      "width" : 2560,
      "height" : 2048,
      ...
      "reader_config" : "rtsp://10.0.0.203",
      ...
    }]

The RTSP stream always has to begin with: "rtsp://" to be read by the `av` plugin.

## AV Output configuration

### Video configuration

For a video file output :
	"output" :
    {
      "type"        : "mp4",
      "video_codec" : "h264",
      "filename"    : "C:\\Users\\VideoStitch\\Vahana VR\\Projects\\test.mp4",
      "audio_codec" : "aac",
      "sampling_rate" : 48000,
      "sample_format" : "fltp",
      "channel_layout" : "stereo",
      "audio_bitrate" : 192
    }

###### type
type : *string*
default : **required**
notes : muxer type : "mp4", "mov"

###### video_codec
type : *string*
default : **required**
notes : video codec : mjpeg, mpeg2, mpeg4, h264, prores, h264_nvenc

##### filename
type : *string*
default : **required**
notes : the output file name

###### bitrate
type : *int*
default : *15000000*
notes : target bitrate, in *bps*

###### profile
type : *string*
default : *main*
notes : H264 profile, baseline | main | high | constrained_high | high444 | stereo
Specified values will be ignored if the resolution & fps do not fit in the requested profile.

###### level
type : *automatic*
default : *automatic*
notes : Any of the H264 standard levels.
Specified values will be ignored if the resolution & fps do not fit in the requested level.

###### bitrate_mode
type : *string*
default : *VBR*
notes : bit rate control mode CBR, VBR (upper case).

###### gop
type : *int*
default : *250*
notes : The target GOP size. Unknown range (0~250 ?).
It is unknown wether scenecut detection can overide this or wether automatic GOP size is possible (eeg. w/ gop=0 as in libx264 encoder). All I-Frames are IDR-Frames.

###### b_frames
type : *int*
default : *0*
notes : number of B frames between two P frames

### Audio configuration

The audio has to be set also. The following table shows which parameters are supported according to the audio codec.

<table>
<tr><th>Audio codec</th><th>Sampling rate</th><th>Sample format</th><th>Channel layout</th><th>Audio bitrate</th></tr>
<tr><td>"aac"</td><td>44100, 48000</td><td>"fltp"</td><td>"mono", "stereo", "3.0", "4.0", "5.0", "5.1", "amb_wxyz"</td><td>64, 128, 192, 512</td></tr>
<tr><td>"mp3"</td><td>44100, 48000</td><td>"s16p"</td><td>"mono", "stereo"</td><td>64, 128, 192</td></tr>
</table>

###### channel_map
type : *array of int*
notes : optional setting to remap audio channels if needed. The array size has to match the number of channels of the channel layout.
For example: With a amb_wxyz layout and a channel_map = [0, 3, 1, 2], the default channel order is W, X, Y, Z. The resulting channel order will be W, Y, Z, X.
