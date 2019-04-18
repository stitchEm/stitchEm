# Magewell documentation

`magewell` is a Windows only IO plugin for Vahana VR. It allows Vahana VR to capture audio/video
input from the [Magewell capture cards](http://www.magewell.com/hardware?lang=en).
All the Magewell cards are supported.

It has been developed and tested with the [XI200DE-HDMI](http://www.magewell.com/hardware/hdmi-cards/xi200de-hdmi/xi200de-hdmi_features.html?lang=en)
and [XI400DE-HDMI](http://www.magewell.com/hardware/hdmi-cards/xi400de-hdmi/xi400de-hdmi_features.html?lang=en)
HDMI capture models.


## Set-up
Drivers can be found at this address : http://www.magewell.com/download?lang=en.
Please follow the documents to set up the card and the drivers.


## Configuration
The magewell plugin can be used by Vahana VR through a .vah project file. Please see
the `*.vah file format specification` for additional details.

Define an input for each input on the capture cards. The `reader_config` member specifies how to read
it.

### Example

    "inputs" : [
    {
      "width" : 1920,
      "height" : 1080,
      ...
      "reader_config" : {
          "type" : "magewell",
          "name" : "0",
          "pixel_format" : "RGB",
          "interleaved" : false,
          "frame_rate" : {
            "num" : 30,
            "den" : 1
          },
          "audio" : false,
          "audio_channels" : 0,
          "audio_sample_rate" : 0,
          "audio_sample_depth" : 0,
          "builtin_zoom" : "none"
        },
      ...
    }]

### Parameters
<table>
<tr><th>Member</th><th>Type</th><th>Default value</th><th></th></tr>
<tr><td><strong>type</strong></td><td>string</td><td>magewell</td><td><strong>Required</strong>. Defines a Magewell input.</td></tr>
<tr><td><strong>name</strong></td><td>string</td><td>-</td><td><strong>Required</strong>. The device's number (starting from <code>0</code>).</td></tr>
<tr><td>interleaved</td><td>bool</td><td>false</td><td>If the input is interlaced.</td>
<tr><td><strong>pixel_format</strong></td><td>string</td><td>-</td><td colspan="2"><strong>Required</strong>. The input pixel format. Supported values are <code>UYVY</code>, <code>YUY2</code>, <code>RGBA</code>, and <code>RGB</code>.</td></tr>
<tr><td><strong>builtin_zoom</strong></td><td>string</td><td>-</td><td><strong>Required</strong>. Defines the zoom behaviour. Available values are: <code>zoom</code>, <code>fill</code> or <code>none</code>.</td></tr>
<tr><td><strong>frame_rate</strong></td><td>struct</td><td>-</td><td><strong>Required</strong>. The input framerate.</td></tr>
<tr><td><strong>audio</strong></td><td>bool</td><td>-</td><td><strong>Required</strong>. Does this reader capture audio.</td></tr>
</table>

Note that the audio format is fixed to a 48 kHz sample rate, with double-channels and 16-bit sample.
