# AJA documentation

`AJA` is an IO plugin for Vahana VR. It allows Vahana VR to capture audio/video
input and stream audio/video output with the AJA Corvid capture cards.

The plugin has been developed and tested with the following models:
 * [AJA Corvid 88](https://www.aja.com/en/products/developer/corvid-88)


## Set-up
Drivers:



## Input configuration
The aja plugin can be used by Vahana VR through a .vah project file. Please see 
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
        "type" : "aja",
        "name" : "01",
        "device": 0,
        "channel": 1,
        "fps" : 30,
        "pixel_format" : "UYVY",
        "audio" : true,
      }
      ...
    },
    {
      "width" : 1920,
      "height" : 1080,
      ...
      "reader_config" : {
        "type" : "aja",
        "name" : "02",
        "device": 0,
        "channel" 2,
        "fps" : 30,
        "pixel_format" : "UYVY",
        "audio" : false
      }
      ...
    }]
    
### Parameters
<table>
<tr><th>Member</th><th>Type</th><th>Value</th><th colspan="2"></th></tr>
<tr><td><strong>type</strong></td><td>string</td><td>aja</td><td colspan="2"><strong>Required</strong>. Defines an AJA input.</td></tr>
<tr><td><strong>name</strong></td><td>string</td><td>-</td><td colspan="2"><strong>Required</strong>. The AJA input entry name.</td></tr>
<br /> Note that these fields width the <code>width</code> and <code>height</code> fields must match exactly an existing display mode below.</td></tr>
<tr><td><strong> device</strong></td><td>int</td><td>-</td><td><strong>Required</strong>. The input card number (Starting from 0).</td></tr>
<tr><td><strong> channel</strong></td><td>int</td><td>-</td><td><strong>Required</strong>. The intpu channel for the selected card.</td></tr>
<tr><td><strong> fps</strong></td><td>double</td><td>-</td><td><strong>Required</strong>. The input framerate.</td></tr>
<tr><td> <strong>pixel_format</strong></td><td>string</td><td>-</td><td colspan="2"><strong>Required</strong>. The input pixel format. Supported values are <code>UYVY</code>, <code>YV12</code> and <code>BGRU</code>.</td></tr>
<tr><td><strong>audio</strong></td><td>bool</td><td>-</td><td colspan="2"><strong>Required</strong>. Does this reader capture audio.</td></tr>
</table>


## Output configuration
An example on how to use the device 1 in the channel 2 with audio enabled at 29.97 fps.

### Example
  
    "outputs" : [
    {
      "type" : "aja",
      "filename" : "12",
      "device" : 1,
      "channel" : 2,
      "fps" : 29.97,
      "pixel_format" : "UYVY",
      "audio": true
    }]
    
### Parameters
<table>
<tr><th>Member</th><th>Type</th><th>Value</th><th colspan="2"></th></tr>
<tr><td><strong>type</strong></td><td>string</td><td>aja</td><td colspan="2"><strong>Required</strong>. Defines an AJA output.</td></tr>
<tr><td><strong>filename</strong></td><td>string</td><td>-</td><td colspan="2"><strong>Required</strong>. AJA output identifier.</td></tr>
 <br /> Note that these fields must match exactly an existing display mode below. You also want larger or equals <code>width</code> and a <code>height</code> than the pano's ones.</td></tr>
 <tr><td><strong> device</strong></td><td>int</td><td>-</td><td><strong>Required</strong>. The output card number (Starting from 0).</td></tr>
<tr><td><strong> channel</strong></td><td>int</td><td>-</td><td><strong>Required</strong>. The output channel for the selected card.</td></tr>
<tr><td> <strong> fps</strong></td><td>double</td><td>-</td><td><strong>Required</strong>. The output framerate.</td></tr>
<tr><td> <strong> pixel_format</strong></td><td>string</td><td>-</td><td colspan="2"><strong>Required</strong>. The output pixel format. Supported values are <code>UYVY</code>, <code>YV12</code> and <code>BGRU</code>.</td></tr>
<tr><td> <strong> audio</strong></td><td>bool</td><td>-</td><td><strong>Required</strong>. The output audio is enabled or disabled.</td></tr>
<tr><td> <strong> offset_x</strong></td><td>bool</td><td>-</td><td><strong>Optional</strong>. The horizontal panorama offset within the display.</td></tr>
<tr><td> <strong> offset_y</strong></td><td>bool</td><td>-</td><td><strong>Optional</strong>. The vertical panorama offset within the display.</td></tr>
</table>


##Audio
We only support one audio mode for input and output: 
48000 Hz 8 channels and 32 bits

## AJA supported display modes
The available display modes depend on the AJA cards.
For Corvid 88 you can get more information at https://www.aja.com/en/products/developer/corvid-88#techspecs
<table>
<tr><th>Mode name</th><th>Height</th><th>Width</th><th>Mode</th><th>Framerate</th></tr>
<tr><td>SD</td><td>525</td><td></td><td>i</td><td>29.97</td></tr>
<tr><td>SD</td><td>625</td><td></td><td>i</td><td>25</td></tr>
<tr><td>HD</td><td>720</td><td></td><td>p</td><td>50</td></tr>
<tr><td>HD</td><td>720</td><td></td><td>p</td><td>59.94</td></tr>
<tr><td>HD</td><td>720</td><td></td><td>p</td><td>60</td></tr>
<tr><td>HD</td><td>1080</td><td></td><td>i</td><td>25</td></tr>
<tr><td>HD</td><td>1080</td><td></td><td>i</td><td>29.97</td></tr>
<tr><td>HD</td><td>1080</td><td></td><td>i</td><td>30</td></tr>
<tr><td>HD</td><td>1080</td><td></td><td>PsF</td><td>23.98</td></tr>
<tr><td>HD</td><td>1080</td><td></td><td>PsF</td><td>24</td></tr>
<tr><td>HD</td><td>1080</td><td></td><td>PsF</td><td>29.97</td></tr>
<tr><td>HD</td><td>1080</td><td></td><td>PsF</td><td>30</td></tr>
<tr><td>HD</td><td>1080</td><td></td><td>p</td><td>23.98</td></tr>
<tr><td>HD</td><td>1080</td><td></td><td>p</td><td>24</td></tr>
<tr><td>HD</td><td>1080</td><td></td><td>p</td><td>25</td></tr>
<tr><td>HD</td><td>1080</td><td></td><td>p</td><td>29.97</td></tr>
<tr><td>HD</td><td>1080</td><td></td><td>p</td><td>30</td></tr>
<tr><td>HD</td><td>1080</td><td></td><td>p</td><td>50A/B</td></tr>
<tr><td>HD</td><td>1080</td><td></td><td>p</td><td>59.94A/B</td></tr>
<tr><td>HD</td><td>1080</td><td></td><td>p</td><td>60A/B</td></tr>
<tr><td>2K</td><td>1080</td><td>2048</td><td>p</td><td>23.98</td></tr>
<tr><td>2K</td><td>1080</td><td>2048</td><td>p</td><td>24</td></tr>
<tr><td>2K</td><td>1080</td><td>2048</td><td>p</td><td>25</td></tr>
<tr><td>2K</td><td>1080</td><td>2048</td><td>p</td><td>29.97</td></tr>
<tr><td>2K</td><td>1080</td><td>2048</td><td>p</td><td>30</td></tr>
<tr><td>2K</td><td>1080</td><td>2048</td><td>p</td><td>48A/B</td></tr>
<tr><td>2K</td><td>1080</td><td>2048</td><td>p</td><td>50A/B</td></tr>
<tr><td>2K</td><td>1080</td><td>2048</td><td>p</td><td>59.94A/B</td></tr>
<tr><td>2K</td><td>1080</td><td>2048</td><td>p</td><td>60A/B</td></tr>
<tr><td>2K</td><td>1080</td><td>2048</td><td>PsF</td><td>23.98</td></tr>
<tr><td>2K</td><td>1080</td><td>2048</td><td>PsF</td><td>24</td></tr>
<tr><td>2K</td><td>1080</td><td>2048</td><td>PsF</td><td>25</td></tr>
<tr><td>2K</td><td>1080</td><td>2048</td><td>PsF</td><td>29.97</td></tr>
<tr><td>2K</td><td>1080</td><td>2048</td><td>PsF</td><td>30</td></tr>
<tr><td>2K</td><td>1080</td><td>2048</td><td>PsF</td><td>48A/B</td></tr>
<tr><td>2K</td><td>1080</td><td>2048</td><td>PsF</td><td>50A/B</td></tr>
<tr><td>2K</td><td>1080</td><td>2048</td><td>PsF</td><td>59.94A/B</td></tr>
<tr><td>2K</td><td>1080</td><td>2048</td><td>PsF</td><td>60A/B</td></tr>
<tr><td>4K</td><td>2160</td><td>3840</td><td>p</td><td>23.98</td></tr>
<tr><td>4K</td><td>2160</td><td>3840</td><td>p</td><td>24</td></tr>
<tr><td>4K</td><td>2160</td><td>3840</td><td>p</td><td>25</td></tr>
<tr><td>4K</td><td>2160</td><td>3840</td><td>p</td><td>29.97</td></tr>
<tr><td>4K</td><td>2160</td><td>3840</td><td>p</td><td>30</td></tr>
<tr><td>4K</td><td>2160</td><td>3840</td><td>p</td><td>48A/B</td></tr>
<tr><td>4K</td><td>2160</td><td>3840</td><td>p</td><td>50A/B</td></tr>
<tr><td>4K</td><td>2160</td><td>3840</td><td>p</td><td>59.94A/B</td></tr>
<tr><td>4K</td><td>2160</td><td>3840</td><td>p</td><td>60A/B</td></tr>
<tr><td>4K</td><td>2160</td><td>4096</td><td>p</td><td>23.98</td></tr>
<tr><td>4K</td><td>2160</td><td>4096</td><td>p</td><td>24</td></tr>
<tr><td>4K</td><td>2160</td><td>4096</td><td>p</td><td>25</td></tr>
<tr><td>4K</td><td>2160</td><td>4096</td><td>p</td><td>29.97</td></tr>
<tr><td>4K</td><td>2160</td><td>4096</td><td>p</td><td>30</td></tr>
<tr><td>4K</td><td>2160</td><td>4096</td><td>p</td><td>48A/B</td></tr>
<tr><td>4K</td><td>2160</td><td>4096</td><td>p</td><td>50A/B</td></tr>
<tr><td>4K</td><td>2160</td><td>4096</td><td>p</td><td>59.94A/B</td></tr>
<tr><td>4K</td><td>2160</td><td>4096</td><td>p</td><td>60A/B</td></tr>
</table>
