# Ximea documentation

`ximea` is a Windows only IO plugin for Vahana VR. It allows Vahana VR to capture video
input from Ximea [xiB PCI Express cameras](http://www.ximea.com/en/products/application-specific-oem-and-custom-cameras/pci-express-high-speed-cameras).

It has been developed and tested with the CB200MU-CM model.


## Set-up
Drivers can be found at this address : http://www.ximea.com/support/wiki/pcie-cam/Wiki.
The xi4API from CB Software Package is actually used by the plugin. It is
actually used because it provides a full xiB camera support.
Please follow the Quick Start Guide from the package to set up the camera and the drivers.

### For multiple camera Set-up

To install a new multiple camera configuration follow this steps:
- Put the first aquisition card into the first pcie port
- Install the driver as the Ximea website
- Shut down the computer
- Put the second card at the same place than the first with the camera plug
- Check if the set-up is working
- If this card is working too you can continue to the last step, otherwise something goes wrong
- Put the first card in an other pcie port, plug with the camera

For more camera just continue to plug the new card alone on the first port for the first launch. Windows will save this card as a Ximea device.

## Configuration
The ximea plugin can be used by Vahana VR through a .vah project file. Please see
the `*.vah file format specification` for additional details.

Define an input for each camera. The `reader_config` member specifies how to read
it.

### Example

    "inputs" : [
    {
      "width" : 5120,
      "height" : 3840,
      ...
      "reader_config" : {
        "type" : "ximea",
        "name" : "0",
        "fps" : 30,
        "exposure" : 80000,
        "gain" : 63,
        "black_level_offset" : -55,
        "f-stop" : 5.6
      }
      ...
    }

### Parameters
<table>
<tr><th>Member</th><th>Type</th><th>Default value</th><th></th></tr>
<tr><td><strong>type</strong></td><td>string</td><td>ximea</td><td><strong>Required</strong>. Defines a Ximea xiB input.</td></tr>
<tr><td><strong>name</strong></td><td>string</td><td>-</td><td><strong>Required</strong>. The device's number (starting from <code>0</code>).</td></tr>
<tr><td><strong>fps</strong></td><td>float</td><td>-</td><td><strong>Required</strong>. The input framerate.</td></tr>
<tr><td><strong>exposure</strong></td><td>int</td><td>-</td><td><strong>Required</strong>. The capturing time of each frame (in microseconds). A higher value will give a darker but sharper image.</td></tr>
<tr><td>frequency</td><td>int</td><td>-</td><td>The bit frequency of the data channels from the sensor in MHz. Calculated as : <code>frequency = width x height x fps x 12 (bytes_per_pixels) x 0.99 (1% margin) / channels</code>. <br> If set manually, a lower frequency can be necessary to avoid dropped or corrupt frames.</td></tr>
<tr><td>channels</td><td>int</td><td>16</td><td>The number of data channels from the sensor. On slower systems, for example when using PCI Express x2, the number of channels can be lowered to <code>8</code>, which results in half speed.</td></tr>
<tr><td>gain</td><td>int</td><td>43</td><td>The frames' saturation level (from <code>0</code> to <code>63</code>).</td></tr>
<tr><td>black_level_offset</td><td>int</td><td>-55</td><td>The frames' black level.</td></tr>
<tr><td>f-stop</td><td>float</td><td>-</td><td>The lens' apperture. The lens configuration can me misdetected, then this parameter will be not used.</td></tr>
</table>

Ximea xiB cameras sensor size are `5120x3840 px`. If set to lower, a region of interest will be used from the top left point with the defined resolution.

Ximea xiB cameras are using Mono12p pixel format. The ximea plugin will unpack it to RGBA.
