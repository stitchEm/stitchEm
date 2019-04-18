# DeckLink documentation

`decklink` is an IO plugin for Vahana VR. It allows Vahana VR to capture audio/video
input and stream audio/video output with the Blackmagic Design [DeckLink capture cards](https://www.blackmagicdesign.com/products/decklink/).
All the DeckLink cards are supported.

The plugin has been developed and tested with the following models:
 * [DeckLink 4K Extreme](https://www.blackmagicdesign.com/products/decklink/techspecs/W-DLK-04)
 * [DeckLink SDI 4K](https://www.blackmagicdesign.com/products/decklink/techspecs/W-DLK-11)
 * [DeckLink Mini Monitor](https://www.blackmagicdesign.com/products/decklink/techspecs/W-DLK-05)
 * [DeckLink Duo](https://www.blackmagicdesign.com/products/decklink/techspecs/W-DLK-01)


## Set-up
Drivers and documentation can be found at this address with the latest Desktop Video package : https://www.blackmagicdesign.com/support/family/capture-and-playback.
Please follow the Desktop Video Manual to set up the card and the drivers.

Test its correct functioning with the Blackmagic Design Control Panel and the
Media Express software. In particular, set the input and the output you want to
use, as you can't do it yet trough the plugin.


## Input configuration
The decklink plugin can be used by Vahana VR through a .vah project file. Please see
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
        "type" : "decklink",
        "name" : "DeckLink SDI 4K (1)",
        "interleaved" : false,
          "frame_rate" : {
            "num" : 30,
            "den" : 1
          },
        "pixel_format" : "UYVY",
        "audio" : true,
        "audio_sample_depth" : 16,
        "audio_channels" 2:
      }
      ...
    },
    {
      "width" : 1920,
      "height" : 1080,
      ...
      "reader_config" : {
        "type" : "decklink",
        "name" : "DeckLink SDI 4K (2)",
        "interleaved" : false,
          "frame_rate" : {
            "num" : 30,
            "den" : 1
          },
        "pixel_format" : "UYVY",
        "audio" : false
      }
      ...
    }]

### Parameters
<table>
<tr><th>Member</th><th>Type</th><th>Value</th><th colspan="2"></th></tr>
<tr><td><strong>type</strong></td><td>string</td><td>decklink</td><td colspan="2"><strong>Required</strong>. Defines a DeckLink input.</td></tr>
<tr><td><strong>name</strong></td><td>string</td><td>-</td><td colspan="2"><strong>Required</strong>. The DeckLink input entry name.</td></tr>
<tr><td>interleaved</td><td>bool</td><td>-</td><td><strong>Required</strong>. If the input is interlaced.</td><td rowspan="2">See <a href="#decklink-available-display-modes">DeckLink available display modes</a> for detail
<br /> Note that these fields width the <code>width</code> and <code>height</code> fields must match exactly an existing display mode below.</td></tr>
<tr><td> frame_rate</td><td>struct</td><td>-</td><td><strong>Required</strong>. The input framerate.</td></tr>
<tr><td> pixel_format</td><td>string</td><td>-</td><td colspan="2"><strong>Required</strong>. The input pixel format. Supported values are <code>UYVY</code>, <code>YV12</code> and <code>BGRU</code>.</td></tr>
<tr><td><strong>audio</strong></td><td>bool</td><td>-</td><td colspan="2"><strong>Required</strong>. Does this reader capture audio.</td></tr>
<tr><td><strong>audio_sample_depth</strong></td><td>int</td><td>-</td><td colspan="2">The audio sample size (eg <code>16</code> or <code>32</code> bit).</td></tr>
<tr><td><strong>audio_channels</strong></td><td>int</td><td>-</td><td colspan="2">Number of channels (<code>1</code> for mono, <code>2</code> for stereo, etc).</td></tr>
</table>


## Output configuration
Each card will output on HDMI and SDI at the same time.

### Example

    "outputs" : [
    {
      "type" : "decklink",
      "filename" : "DeckLink SDI 4K (1)",
      "width" : 1920,
      "height" : 1080,
      "interleaved" : false,
      "fps" : 30,
      "pixel_format" : "UYVY"
    }]

### Parameters
<table>
<tr><th>Member</th><th>Type</th><th>Value</th><th colspan="2"></th></tr>
<tr><td><strong>type</strong></td><td>string</td><td>decklink</td><td colspan="2"><strong>Required</strong>. Defines a DeckLink output.</td></tr>
<tr><td><strong>filename</strong></td><td>string</td><td>-</td><td colspan="2"><strong>Required</strong>. The DeckLink output name.</td></tr>
<tr><td> width</td><td>int</td><td>-</td><td><strong>Required</strong>. The output width in pixels.</td><td rowspan="4">See <a href="#decklink-available-display-modes">DeckLink available display modes</a> for details.
 <br /> Note that these fields must match exactly an existing display mode below. You also want larger or equals <code>width</code> and a <code>height</code> than the pano's ones.</td></tr>
<tr><td> height</td><td>int</td><td>-</td><td><strong>Required</strong>. The output height in pixels.</td></tr>
<tr><td> interleaved</td><td>bool</td><td>-</td><td><strong>Required</strong>. If the output is interlaced.</td></tr>
<tr><td> fps</td><td>float</td><td>-</td><td><strong>Required</strong>. The output framerate.</td></tr>
<tr><td> pixel_format</td><td>string</td><td>-</td><td colspan="2"><strong>Required</strong>. The output pixel format. Supported values are <code>UYVY</code>, <code>YV12</code> and <code>BGRU</code>.</td></tr>
</table>


## DeckLink available display modes
The available display modes depend on the DeckLink cards.
See also the capabilities of each card type: https://www.blackmagicdesign.com/products/decklink/techspecs/

<table>
<tr><th>Mode name</th><th>Width</th><th>Height</th><th>Interleaved</th><th>Framerate</th></tr>
<tr><td>NTSC</td><td>720</td><td>486</td><td>true</td><td>29.97</td></tr>
<tr><td>NTSCp</td><td>720</td><td>486</td><td>false</td><td>59.94</td></tr>
<tr><td>PAL</td><td>720</td><td>576</td><td>true</td><td>25</td></tr>
<tr><td>PALp</td><td>720</td><td>576</td><td>false</td><td>50</td></tr>
<tr><td>HD720p50</td><td>1280</td><td>720</td><td>false</td><td>50</td></tr>
<tr><td>HD720p5994</td><td>1280</td><td>720</td><td>false</td><td>59.94</td></tr>
<tr><td>HD720p60</td><td>1280</td><td>720</td><td>false</td><td>60</td></tr>
<tr><td>HD1080p2398</td><td>1920</td><td>1080</td><td>false</td><td>23.98</td></tr>
<tr><td>HD1080p24</td><td>1920</td><td>1080</td><td>false</td><td>24</td></tr>
<tr><td>HD1080p25</td><td>1920</td><td>1080</td><td>false</td><td>25</td></tr>
<tr><td>HD1080i50</td><td>1920</td><td>1080</td><td>true</td><td>25</td></tr>
<tr><td>HD1080p2997</td><td>1920</td><td>1080</td><td>false</td><td>29.97</td></tr>
<tr><td>HD1080i5994</td><td>1920</td><td>1080</td><td>true</td><td>29.97</td></tr>
<tr><td>HD1080p30</td><td>1920</td><td>1080</td><td>false</td><td>30</td></tr>
<tr><td>HD1080i60</td><td>1920</td><td>1080</td><td>true</td><td>30</td></tr>
<tr><td>HD1080p50</td><td>1920</td><td>1080</td><td>false</td><td>50</td></tr>
<tr><td>HD1080p5994</td><td>1920</td><td>1080</td><td>false</td><td>59.94</td></tr>
<tr><td>HD1080p60</td><td>1920</td><td>1080</td><td>false</td><td>60</td></tr>
<tr><td>2k2398</td><td>2048</td><td>1556</td><td>false</td><td>23.98</td></tr>
<tr><td>2k24</td><td>2048</td><td>1556</td><td>false</td><td>24</td></tr>
<tr><td>2k25</td><td>2048</td><td>1556</td><td>false</td><td>25</td></tr>
<tr><td>2kDCI2398</td><td>2048</td><td>1080</td><td>false</td><td>23.98</td></tr>
<tr><td>2kDCI24</td><td>2048</td><td>1080</td><td>false</td><td>24</td></tr>
<tr><td>2kDCI25</td><td>2048</td><td>1080</td><td>false</td><td>25</td></tr>
<tr><td>4kp2398</td><td>3840</td><td>2160</td><td>false</td><td>23.98</td></tr>
<tr><td>4kp24</td><td>3840</td><td>2160</td><td>false</td><td>24</td></tr>
<tr><td>4kp25</td><td>3840</td><td>2160</td><td>false</td><td>25</td></tr>
<tr><td>4kp2997</td><td>3840</td><td>2160</td><td>false</td><td>29.97</td></tr>
<tr><td>4kp30</td><td>3840</td><td>2160</td><td>false</td><td>30</td></tr>
<tr><td>4kDCI2398</td><td>4096</td><td>2160</td><td>false</td><td>23.98</td></tr>
<tr><td>4kDCI24</td><td>4096</td><td>2160</td><td>false</td><td>24</td></tr>
<tr><td>4kDCI25</td><td>4096</td><td>2160</td><td>false</td><td>25</td></tr>
</table>
