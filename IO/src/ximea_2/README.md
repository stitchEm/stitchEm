# Ximea documentation

`XIMEA` is an IO plugin for Vahana VR. It allows Vahana VR to capture audio/video
stream with Ximea cameras.

The plugin has been developed and tested with the following models:
 * [MX023MG-SY-X2G] *8


## Set-up
Drivers:



## Input configuration
The ximea plugin can be used by Vahana VR through a .vah project file. Please see
the `*.vah file format specification` for additional details.

Define an input for each input on the capture cards. The `reader_config` member specifies how to read
it.

### Example

    "inputs" : [
    {
    "reader_config" : {
          "type" : "ximea",
          "name" : "MX023MG-SY-X2G27",
          "pixel_format" : "Grayscale",
          "interleaved" : false,
          "frame_rate" : {
            "num" : 30,
            "den" : 1
          },
          "device" : 0,
          "bandwidth" : 1000
        },
        "width" : 1936,
        "height" : 1216,
      ...
    },
    {
    "reader_config" : {
          "type" : "ximea",
          "name" : "MX023MG-SY-X2G27",
          "pixel_format" : "Grayscale",
          "interleaved" : false,
          "frame_rate" : {
            "num" : 30,
            "den" : 1
          },
          "device" : 7,
          "bandwidth" : 1000,
          "fps_limit" : 30
        },
        "width" : 1936,
        "height" : 1216,
      ...
    }]

### Parameters
<table>
<tr><th>Member</th><th>Type</th><th>Value</th><th colspan="2"></th></tr>
<tr><td><strong>type</strong></td><td>string</td><td>Ximea</td><td colspan="2"><strong>Required</strong>. Defines an Ximea input.</td></tr>
<br /> Note that the <code>width</code> and <code>height</code> fields must match exactly an existing display mode below.</td></tr>
<tr><td><strong>device</strong></td><td>int</td><td>-</td><td><strong>Required</strong>. The input cam number (Starting from 0).</td></tr>
<tr><td> <strong>pixel_format</strong></td><td>string</td><td>-</td><td colspan="2"><strong>Unused</strong>. The input pixel format. Supported values are <code>Grayscale</code>.</td></tr>
<tr><td><strong>bandwidth</strong></td><td>int</td><td>-</td><td><strong>Optional</strong>. The PCIx bus bandwidth upper limit allowed for the cam (highly recommanded for avoiding your OS to freeze).</td></tr>
<tr><td><strong>fps_limit</strong></td><td>int</td><td>-</td><td><strong>Optional</strong>. The framerate upper limit for the cam (Work relativily bad).</td></tr>
</table>



## XIMEA supported display modes
There is only one available display mode but downscaling is available at camera level : 1936x1216
FPS is not fixed
