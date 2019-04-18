# PTV file format specification

PTV is the file format used by VideoStitch. It uses full [JSON](http://www.json.org/) syntax. Projects are simply defined by populating the root object with a few named variables.

## Objects

VideoStitch will interpret the following objects. An object is of course free to have other members in addition to these.
In the tables below, mandatory members are shown in bold. If not set, optional members take a default value specified in the "Default value" column.

### Root

VideoStitch will interpret the following members of the root object:

<table>
<tr><th>Member</th><th>Type</th><th>Default value</th><th></th></tr>
<tr><th>lib_version</th><td>string</td><td>-</td><td>Version of our core library used to save the project file.</td></tr>
<tr><th>first_frame</th><td>int</td><td>-</td><td>First frame of the project.</td></tr>
<tr><th>last_frame</th><td>int</td><td>-</td><td>Last frame of the project. Set to -1 to try to autodetect the last frame. All readers may not support autodetection.</td></tr>
<tr><th>pano</th><td>object</td><td>-</td><td>Panorama object (see below).</td></tr>
<tr><th>audio_pipe</th><td>object</td><td>-</td><td>Audio pipeline object (see below).</td></tr>
<tr><th>output</th><td>object</td><td>-</td><td>Output object (see below).</td></tr>
<tr><th>merger</th><td>object</td><td>-</td><td>Merger object (see below).</td></tr>
<tr><th>flow</th><td>object</td><td>-</td><td>Flow object (see below).</td></tr>
<tr><th>warper</th><td>object</td><td>-</td><td>Warper object (see below).</td></tr>
<tr><td>buffer_frames</td><td>int</td><td>2</td><td>The number of frames to buffer. If set to > 0, the writer will buffer that many frames. This can improve GPU utilization. It will usually not be interesting to buffer more than a few frames. Memory usage (CPU RAM only) will go up linearly with buffer_frames.</td></tr>
</table>

### Pano

This specifies the geometry and photometry options of the panorama.

<table>
<tr><th>Member</th><th>Type</th><th>Default value</th><th></th></tr>
<tr><th>width</th><td>int</td><td>-</td><td>The output width, in pixels.</td></tr>
<tr><th>height</th><td>int</td><td>-</td><td>The output height, in pixels.</td></tr>
<tr><th>length</th><td>int</td><td>-</td><td>The length of an edge of the cubemap, in pixels.</td></tr>
<tr><th>hfov</th><td>double</td><td>-</td><td>The horizontal field of view, in degrees.</td></tr>
<tr><th>proj</th><td>string</td><td>-</td><td>The projection. See possible values below.</td></tr>
<tr><td>spherescale</td><td>string</td><td>-</td><td>The stitching sphere scale.</td></tr>
<tr><th>precomputed_coordinate_buffer</th><td>bool</td><td>false</td><td>Whether to precompute the coordinate buffer for looking up pixel value. If yes, the program will use more GPU memory and run slightly faster.</td></tr>
<tr><th>precomputed_coordinate_shrink_factor</th><td>bool</td><td>false</td><td>If precomputed_coordinate_buffer is true, the precomputed coordinate buffer will be downsampled by this factor to run faster.</td></tr>
<tr><th>inputs</th><td>array[object]</td><td>-</td><td>The list of input objects (see below).</td></tr>
<tr><th>overlays</th><td>array[object]</td><td>-</td><td>The list of overlay objects (see below).</td></tr>
<tr><td>wrap</td><td>boolean</td><td>true</td><td>If wrap is true, hfov is 360.0, and the projection supports it, then the panorama will wrap seamlessly across 360 border.</td></tr>
<tr><td>ev</td><td>double</td><td>0.0</td><td>Exposure value correction.</td></tr>
<tr><td>global_yaw</td><td>object</td><td>null</td><td>The panorama's global yaw. Can depend on time. Default is no global yaw.</td></tr>
<tr><td>global_pitch</td><td>object</td><td>null</td><td>The panorama's global pitch. Default is no global pitch.</td></tr>
<tr><td>global_roll</td><td>object</td><td>null</td><td>The panorama's global roll. Default is no global roll.</td></tr>
<tr><td>rig</td><td>object</td><td>array[object]</td><td>Rig definition for calibration presets. See the algorithm documentation for full format.</td></tr>
<tr><td>cameras</td><td>object</td><td>array[object]</td><td>Cameras definition for calibration presets. See the algorithm documentation for full format.</td></tr>
</table>

Possible values for (Pano) <i>proj</i> are:

*   "rectilinear"
*   "equirectangular"
*   "ff_fisheye"
*   "stereographic"
*   "cubemap"


### Input

This specifies the geometry and photometry options for each input.

<table>
<tr><th>Member</th><th>Type</th><th>Default value</th><th></th></tr>
<tr><th> width</th><td>int</td><td>-</td><td>The input width, in pixels.</td></tr>
<tr><th> height</th><td>int</td><td>-</td><td>The input height, in pixels.</td></tr>
<tr><td> hfov</td><td>double</td><td>-</td><td>The horizontal field of view, in degrees (old syntax, replaced by "horizontalFocal" in <a href="#input-geom">geometries</a>).</td></tr>
<tr><td> yaw</td><td>double</td><td>-</td><td>The yaw, in degrees (old syntax, now in <a href="#input-geom">geometries</a>).</td></tr>
<tr><td> pitch</td><td>double</td><td>-</td><td>The pitch, in degrees (old syntax, now in <a href="#input-geom">geometries</a>).</td></tr>
<tr><td> roll</td><td>double</td><td>-</td><td>The roll, in degrees (old syntax, now in <a href="#input-geom">geometries</a>).</td></tr>
<tr><th> reader_config (previously filename)</th><td>string or object</td><td>-</td><td>The reader configuration. Usually a filename, but can be more elaborated to enable advanced features. See the Readers section below.</td></tr>
<tr><th> proj</th><td>string</td><td>-</td><td>The projection. <a href="#proj1">Possible values</a></td></tr>
<tr><th> <a href="#input-geom">geometries</a></th><td>curve object</td><td>-</td><td><a href="#geom">The camera geometry curves.</a></td></tr>

<tr><td> crop_left</td><td>int</td><td>0</td><td>Crop that many pixels from the left of the input.</td></tr>
<tr><td> crop_right</td><td>int</td><td>=width</td><td>Crop that many pixels from the right of the input.</td></tr>
<tr><td> crop_top</td><td>int</td><td>0</td><td>Crop that many pixels from the top of the input.</td></tr>
<tr><td> crop_bottom</th><td>int</td><td>=height</td><td>Crop that many pixels from the bottom of the input.</td></tr>
<tr><td> viewpoint_model</td><td>string</td><td>"hugin"</td><td>Viewpoint model: "hugin" or "ptgui".</td></tr>
<tr><td> translation_x</td><td>double</td><td>0.0</td><td>Viewpoint translation along the X axis (old syntax, now in <a href="#input-geom">geometries</a>).</td></tr>
<tr><td> translation_y</td><td>double</td><td>0.0</td><td>Viewpoint translation along the Y axis (old syntax, now in <a href="#input-geom">geometries</a>).</td></tr>
<tr><td> translation_z</td><td>double</td><td>0.0</td><td>Viewpoint translation along the Z axis (old syntax, now in <a href="#input-geom">geometries</a>).</td></tr>
<tr><td> viewpoint_pan</td><td>double</td><td>0.0</td><td>Viewpoint pan (if viewpoint_model=="ptgui").</td></tr>
<tr><td> viewpoint_tilt</td><td>double</td><td>0.0</td><td>Viewpoint tilt (if viewpoint_model=="ptgui").</td></tr>
<tr><td> dist_center_x</td><td>double</td><td>0.0</td><td>Horizontal shift (old syntax, replaced by "center_x" in <a href="#input-geom">geometries</a>).</td></tr>
<tr><td> dist_center_y</td><td>double</td><td>0.0</td><td>Vertical shift (old syntax, replaced by "center_y" in <a href="#input-geom">geometries</a>).</td></tr>
<tr><td> response</td><td>string</td><td>"emor"</td><td>Camera response model. One of "emor", "gamma", "linear", "inverse_emor" or "curve".</td></tr>
<tr><td> gamma</td><td>double</td><td>1.0</td><td>If response is "gamma", the gamma response parameter.</td></tr>
<tr><td> emor_a</td><td>double</td><td>0.0</td><td>If response is "emor" or "inverse_emor", the first emor response parameter.</td></tr>
<tr><td> emor_b</td><td>double</td><td>0.0</td><td>If response is "emor" or "inverse_emor", the second emor response parameter.</td></tr>
<tr><td> emor_c</td><td>double</td><td>0.0</td><td>If response is "emor" or "inverse_emor", the third emor response parameter.</td></tr>
<tr><td> emor_d</td><td>double</td><td>0.0</td><td>If response is "emor" or "inverse_emor", the fourth emor response parameter.</td></tr>
<tr><td> emor_e</td><td>double</td><td>0.0</td><td>If response is "emor" or "inverse_emor", the fifth emor response parameter.</td></tr>
<tr><td> response_curve</td><td>[int]</td><td>[]</td><td>If response is "curve", a list of 1024 integers describing the camera response curve by value.</td></tr>
<tr><td> ev</td><td>double</td><td>0.0</td><td>Exposure value correction.</td></tr>
<tr><td> red_corr</td><td>double</td><td>1.0</td><td>Red white balance multiplier.</td></tr>
<tr><td> blue_corr</td><td>double</td><td>1.0</td><td>Blue white balance multiplier.</td></tr>
<tr><td> lens_dist_a</td><td>double</td><td>0.0</td><td>Lens distortion parameter (degree 0) (old syntax, replaced by "distort_a" in <a href="#input-geom">geometries</a>).</td></tr>
<tr><td> lens_dist_b</td><td>double</td><td>0.0</td><td>Lens distortion parameter (degree 1) (old syntax, replaced by "distort_b" in <a href="#input-geom">geometries</a>).</td></tr>
<tr><td> lens_dist_c</td><td>double</td><td>0.0</td><td>Lens distortion parameter (degree 2) (old syntax, replaced by "distort_c" in <a href="#input-geom">geometries</a>).</td></tr>
<tr><td> vign_a</td><td>double</td><td>1.0</td><td>Vigneting parameter (degree 0).</td></tr>
<tr><td> vign_b</td><td>double</td><td>0.0</td><td>Vigneting parameter (degree 1).</td></tr>
<tr><td> vign_c</td><td>double</td><td>0.0</td><td>Vigneting parameter (degree 2).</td></tr>
<tr><td> vign_d</td><td>double</td><td>0.0</td><td>Vigneting parameter (degree 3).</td></tr>
<tr><td> vign_x</td><td>double</td><td>0.0</td><td>Vigneting center along x axis, relative to image center.</td></tr>
<tr><td> vign_y</td><td>double</td><td>0.0</td><td>Vigneting center along y axis, relative to image center.</td></tr>
<tr><td> frame_offset</td><td>int</td><td>0</td><td>Offset of this input relative to the origin of time, in frames.</td></tr>
<tr><td> preprocessors</td><td>list</td><td>[]</td><td>A list of processors to run before mapping. See below for a list of available preprocessors.</td></tr>
<tr><td> mask_data</th><td>string</td><td>""</td><td>An inline, base64 encoded 2-color colormapped png file, the size of the input. Red pixels are masked out.</td></tr>
<tr><td> no_delete_masked_pixels</td><td>bool</td><td>false</td><td>If true, masked pixels will just have alpha 0. If false, they will also influence how stitching seams are computed. To get smooth blending around masked areas, always disable no_delete_masked_pixels.</td></tr>
</table>

Possible values for (Input) <i><a name="proj1">proj</a></i> are:

*   "rectilinear"
*   "circular_fisheye"
*   "ff_fisheye"
*   "circular\_fisheye_opt"
*   "ff\_fisheye_opt"
*   "equirectangular"

### Geometries <a name="geom"></a>

The <a name="input-geom">"geometries"</a> member inside an Input object is a curve of temporally varying camera geometry parameters.

<table>
<tr><th> horizontalFocal</th><td>double</td><td>1000.0</td><td>The horizontal focal parameter, allowing to transform meters into pixels on the sensor plane. </td></tr>
<tr><td> verticalFocal</td><td>double</td><td>-</td><td>The vertical focal parameter, allowing to transform meters into pixels on the sensor plane. If no value is provided, it is considered equal to the horizontalFocal. </td></tr>
<tr><th> center_x</th><td>double</td><td>0.0</td><td>Principal point / center of distortion horizontal shift in pixels w.r.t. the sensor center.</td></tr>
<tr><th> center_y</th><td>double</td><td>0.0</td><td>Principal point / center of distortion vertical shift in pixels w.r.t. the sensor center.</td></tr>
<tr><th> distort_a</th><td>double</td><td>0.0</td><td>Lens radial distortion parameter (degree 0).</td></tr>
<tr><th> distort_b</th><td>double</td><td>0.0</td><td>Lens radial distortion parameter (degree 1).</td></tr>
<tr><th> distort_c</th><td>double</td><td>0.0</td><td>Lens radial distortion parameter (degree 2).</td></tr>
<tr><td> distort_p1</td><td>double</td><td>0.0</td><td>First lens tangential distortion parameter.</td></tr>
<tr><td> distort_p2</td><td>double</td><td>0.0</td><td>Second lens tangential distortion parameter.</td></tr>
<tr><td> distort_s1</td><td>double</td><td>0.0</td><td>First lens thin-prism distortion parameter.</td></tr>
<tr><td> distort_s2</td><td>double</td><td>0.0</td><td>Second lens thin-prism distortion parameter.</td></tr>
<tr><td> distort_s3</td><td>double</td><td>0.0</td><td>Third lens thin-prism distortion parameter.</td></tr>
<tr><td> distort_s4</td><td>double</td><td>0.0</td><td>Fourth lens thin-prism distortion parameter.</td></tr>
<tr><td> distort_tau1</td><td>double</td><td>0.0</td><td>First lens Scheimpflug distortion angle parameter, in radians.</td></tr>
<tr><td> distort_tau2</td><td>double</td><td>0.0</td><td>Second lens Scheimpflug distortion angle parameter, in radians.</td></tr>
<tr><th> yaw</th><td>double</td><td>0.0</td><td>Camera yaw, in degrees.</td></tr>
<tr><th> pitch</th><td>double</td><td>0.0</td><td>Camera pitch, in degrees.</td></tr>
<tr><th> roll</th><td>double</td><td>0.0</td><td>Camera roll, in degrees.</td></tr>
<tr><td> translation_x</td><td>double</td><td>0.0</td><td>Camera translation along the X axis, in meters.</td></tr>
<tr><td> translation_y</td><td>double</td><td>0.0</td><td>Camera translation along the Y axis, in meters.</td></tr>
<tr><td> translation_z</td><td>double</td><td>0.0</td><td>Camera translation along the Z axis, in meters.</td></tr>
</table>

### Overlay

This specifies the geometry and photometry options for each input.

<table>
<tr><th>Member</th><th>Type</th><th>Default value</th><th></th></tr>
<tr><th> reader_config (previously filename)</th><td>string or object</td><td>-</td><td>The reader configuration. Usually a filename, but can be more elaborated to enable advanced features. See the Readers section below.</td></tr>
<tr><th> width</th><td>int</td><td>-</td><td>The overlay input width, in pixels.</td></tr>
<tr><th> height</th><td>int</td><td>-</td><td>The overlay input height, in pixels.</td></tr>
<tr><td> frame_offset</td><td>int</td><td>0</td><td>Offset of this overlay input relative to the origin of time, in frames.</td></tr>
<tr><td> globalOrientationApplied</td><td>[boolean]</td><td>[]</td><td>Boolean to apply the stitcher orientation to the overlay input.</td></tr>
<tr><td> scaleCurve</td><td>[curve object]</td><td>[]</td><td>The overlay object size scale curve, value should be in the interval [0.0, 1.0].</td></tr>.
<tr><td> alphaCurve</td><td>[curve object]</td><td>[]</td><td>The overlay object alpha blending curve, value should be in the interval [0.0, 1.0].</td></tr>.
<tr><td> transXCurve</td><td>[curve object]</td><td>[]</td><td>The overlay object position X translation curve, in meters.</td></tr>.
<tr><td> transYCurve</td><td>[curve object]</td><td>[]</td><td>The overlay object position Y translation curve, in meters.</td></tr>.
<tr><td> transZCurve</td><td>[curve object]</td><td>[]</td><td>The overlay object position Z translation curve, in meters.</td></tr>.
<tr><td> rotationCurve</td><td>[curve object]</td><td>[]</td><td>The overlay object position Yaw, Pitch, Roll orientation curve.</td></tr>.
</table>

### Mergers

Mergers specify how to blend remapped images.

<table>
<tr><th>Member</th><th>Type</th><th>Default value</th><th></th></tr>
<tr><th> type</th><td>string</td><td>laplacian</td><td>Type of blending. One of the types below.</td></tr>
</table>

#### Mergers for end users

The mergers in this category are available in the products for stitching.

##### "gradient" merger

The gradient merger simply blends images using weights taken from a mask.
A feather parameter is used to control the level of smoothness of the generated mask.
Larger feathers have smoother transition but will make calibration errors more apparent, while smaller feathers make the distorted hard egdes visible.
The special value 100 will make the transition as smooth as possible while never overflowing the overlay zone.

<table>
<tr><th>Member</th><th>Type</th><th>Default value</th><th></th></tr>
<tr><td> mask_merger</td><td>int</td><td>0</td><td>The algorithm used to generate the merger mask. Voronoi mask is used by default.</td></tr>
<tr><td> feather</td><td>int</td><td>100</td><td>The feather parameter [0..100] is used to control the level of smoothness for the generated mask. A low value will produce mask with hard edge while a high value will generate smooth mask.</td></tr>
</table>

##### "laplacian" merger

The laplacian merge blends across multiple spacial frequency bands to hide calibration errors and preserve high frequency details while providing a smooth perceptual blending.
Low frequency signals will always be blended over large area. This property is not sensitive to the feature parameter.
However, blending of high frequency signals (strong edge) is very sensitive to the feature parameter.
It's slower than gradient merging but of much higher quality.

<table>
<tr><th>Member</th><th>Type</th><th>Default value</th><th></th></tr>
<tr><td> base_size</td><td>int</td><td>64</td><td>The size of the base level in the laplacian pyramid. The lower this number, the smoother the output (up to a point).</td></tr>
<tr><td> levels</td><td>int</td><td>5</td><td>[DEPRECATED, please use base_size] The number of levels in the laplacian pyramid.</td></tr>
<tr><td> gaussian_radius</td><td>int</td><td>5</td><td>The radius of the low pass gaussian filter used to build the laplacian pyramid.</td></tr>
<tr><td> filter_passes</td><td>int</td><td>1</td><td>The number of passes for gaussian filter computation. 3 makes up for a 97% accuracy. 1 is enough for plausible blending.</td></tr>
<tr><td> mask_merger</td><td>int</td><td>0</td><td>The algorithm used to generate the merger mask. Voronoi mask is used by default. Currently, box filter of radius 3 is used to construct the gaussian pyramid of the generated mask.</td></tr>
<tr><td> feather</td><td>int</td><td>100</td><td>The feather parameter [0..100] is used to control the level of smoothness for the generated mask. A low value will produce mask with hard edge while a high value will generate smooth mask.</td></tr>
</table>

#### Debug mergers

The mergers in this category are for various debugging purposes, and are usually not available through an app interface, unless debug functionality is enabled.

##### "stack" merger

Inputs are not merged, but simply stacked on top of each other.

##### "noblendv1" merger

For each input, the warped pixel is tranformed to grayscale. The first input which maps to a given panorama pixel will be stored in the R component. The second input which maps to the same panorama pixel will be stored in the G component. The alpha bit is set to 1 if and only if two inputs map to a panorama pixel. This is used to make computations on overlap regions in the panorama output space.

##### "array" merger

The array merger is not a merger per se, and will ignore the actual content of the inputs. Instead, it enables visualizing the overlap between inputs (camera array).

##### "checkerboard" merger

In overlapping areas, the checkerboard merger alternates between contributing inputs in a checkerboard pattern. With that it's possible to compare the differences in the inputs indepedent of the stitching line and without any pseudo-color.

The "feather" parameter can be used to set the size of the checker squares, in pixels.

##### "diff" merger

The diff merger shows how well inputs coincide in overlapping zones. The output will be as usual outside of overlapping zones. In overlapping zones, green indicates a perfect match between inputs, and the error gets bigger as the output moves towards more red.

##### "exposure_diff" merger

The exposure diff merger shows how well the exposure of inputs matches in overlapping zones. The output will be black outside of overlapping zones, signifying no error. In overlapping zones, every pixel contains the absolute difference of the value (per RGB channel) between the twofirst and the second input.

##### "inputidv2" merger

The inputid merger shows the overlap zones of each of the inputs. Each pixel of the ouput is set if the corresponding input contributes to this pixel.

### Flow

Flows specify which optical flow algorithm used for the flow-based blending.

<table>
<tr><th>Member</th><th>Type</th><th>Default value</th><th></th></tr>
<tr><th> type</th><td>string</td><td>no</td><td>Type of optical flow. One of the types below.</td></tr>
<tr><th> leftOffset</th><td>int</td><td>no</td><td>The left-offset of the current frame use to stabilize flow temporally.</td></tr>
<tr><th> rightOffset</th><td>int</td><td>no</td><td>The right-offset of the current frame use to stabilize flow temporally.</td></tr>
</table>

#### "no" flow:

Disable flow-based blending.

#### "simple" flow:

Use SimpleFlow.

### Warper

Warper specify which warp method used for the flow-based blending.

<table>
<tr><th>Member</th><th>Type</th><th>Default value</th><th></th></tr>
<tr><th> type</th><td>string</td><td>no</td><td>Type of image warper. One of the types below.</td></tr>
</table>

#### "no" warper:

Disable flow-based blending. No warper is used.

#### "linearflow" flow:

Use linear warper. Along the image boundary, lookup pixels from the computed optical flow.
As moving futher away from the boundary, use offset from the original mapping.
Smooth transition is used to compute lookup coordinate from using pure optical flow to the pure original mapping.

<table>
<tr><th>Member</th><th>Type</th><th>Default value</th><th></th></tr>
<tr><th> maxTransitionDistance</th><td>int</td><td>100</td><td>Transition distance from the border (using optical flow) to using the original mapping.</td></tr>
</table>

### Audio Pipeline

This specifies the audio pipeline configuration. In a nutshell, it specifies how audio inputs are created from the readers. On these audio inputs simple audio processors can be applied. And with these audio inputs you can create audio mixes.

<table>
<tr><th>Member</th><th>Type</th><th>Default value</th><th></th></tr>
<tr><td> sampling_rate</td><td>int</td><td>44100</td><td>Sampling rate of the audio pipeline in Hz. Available values are: 44100 and 48000.</td></tr>
<tr><td> block_size</td><td>int</td><td>512</td><td>Block size in samples. The audio is processed by block of size block size. Value of power of 2 should be preferred.</td></tr>
<tr><th> audio_selected</th><td>string</td><td>-</td><td>Name of the audio mix selected.</td></tr>
<tr><th> audio_inputs</th><td>array[object]</td><td>-</td><td>The list of audio input objects (see below).</td></tr>
<tr><th> audio_mixes</th><td>array[object]</td><td>-</td><td>The list of audio mix objects (see below).</td></tr>
<tr><td> audio_processors</td><td>array[object]</td><td>-</td><td>The list of audio processor objects (see below).</td></tr>
</table>

#### Audio Input

An audio input is created from several audio sources.

<table>
<tr><th>Member</th><th>Type</th><th>Default value</th><th></th></tr>
<tr><td> name</td><td>string</td><td>-</td><td>Name of the audio input. Arbitrary name.</td></tr>
<tr><th> sources</th><td>array[object]</td><td>-</td><td>The list of audio source objects (see below).</td></tr>
</table>

##### Audio Source

Defines which reader and which channel has to be used to create the corresponding audio input.

<table>
<tr><th>Member</th><th>Type</th><th>Default value</th><th></th></tr>
<tr><td> reader_id</td><td>int</td><td>-</td><td>Index of the reader to select. Warning see below.</td></tr>
<tr><td> channel</td><td>int</td><td>-</td><td>Index of the channel in the reader. -1 means select all the channels of the corresponding reader.</td></tr>
</table>

Warning: the reader_id selected has to correspond to a reader with an audio stream. It is your responsibility to check it. If you set a wrong reader_id, the audio input won't be created.

#### Audio Mix

An audio mix offers the ability to combine several audio inputs. By default one audio mix per audio input is created with the same name.

<table>
<tr><th>Member</th><th>Type</th><th>Default value</th><th></th></tr>
<tr><td> name</td><td>string</td><td>-</td><td>Name of the audio mix. Arbitrary name.</td></tr>
<tr><th> inputs</th><td>array[string]</td><td>-</td><td>The list of audio input names. This list should use the name of the defined audio inputs.</td></tr>
</table>

#### Audio Processor

Defines the processing chain for each audio input.

<table>
<tr><th>Member</th><th>Type</th><th>Default value</th><th></th></tr>
<tr><td> name</td><td>string</td><td>-</td><td>Name of the audio processor to be applied.</td></tr>
<tr><th> params</th><td>array[objects]</td><td>-</td><td>The list of parameter objects. The parameters differs according to the type of audio processor (see below).</td></tr>
</table>

##### Delay processor

This processor delays one audio input.

<table>
<tr><th>Member</th><th>Type</th><th>Default value</th><th></th></tr>
<tr><td> input</td><td>string</td><td>-</td><td>Defines on which input the delay processor will be applied.</td></tr>
<tr><td> delay</td><td>float</td><td>-</td><td>Value of the delay in seconds to apply on this input.</td></tr>
</table>

##### Gain processor

Defines a gain on a specific audio input.

<table>
<tr><th>Member</th><th>Type</th><th>Default value</th><th></th></tr>
<tr><td> input</td><td>string</td><td>-</td><td>Defines on which input the delay processor will be applied.</td></tr>
<tr><td> gain</td><td>float</td><td>0</td><td>Value in dB of the gain to apply. Range[-100, 20].</td></tr>
<tr><td> mute</td><td>boolean</td><td>false</td><td>Boolean to mute this input.</td></tr>
<tr><td> reverse_polarity</td><td>boolean</td><td>false</td><td>Boolean to reverse the polarity of the signal.</td></tr>
</table>

### Output

The output object specifies the format and destination for the stitching output.

<table>
<tr><th>Member</th><th>Type</th><th>Default value</th><th></th></tr>
<tr><th> type</th><td>string</td><td>-</td><td>The output type. See below for supported types.</td></tr>
</table>

The following outputs are supported by default. Each of these have specific options.

#### "null"

Passing null will discard output writing. You have to specify "filename" with a dummy value.

#### "profiling"

An output that will measure the time between stitched frames arriving. Can be used on the command line to determine the peak stitching performance of a hardware setup in a realistic environment, independent of I/O operations. When the stitching is done, it logs the mean and median frame rate, as well as the variance to the error log. You have to specify "filename" with a dummy value.

"mp4", "mov", "rtmp" as primary and secondary types and "flv" as a secondary type only.

#### Video output

<table>
<tr><th>Member</th><th>Type</th><th>Default value</th><th></th></tr>
<tr><th> filename</th><td>string</td><td>-</td><td>Output filename, without extension.</td></tr>
<tr><td> video_codec</td><td>string</td><td>"h264"</td><td>The video codec. Available values are: "mpeg2", "mpeg4" (MPEG4 part2, not h264/AVC), "h264", "mjpeg" (Motion JPEG)</td></tr>
<tr><td> fps</td><td>double</td><td>25</td><td>The framerate. Available values range from 1 to 1000.</td></tr>
<tr><td> bitrate</td><td>int</td><td>15000000</td><td>The bitrate in bits per second. Available values range from 500000 to 110000000.</td></tr>
<tr><td> gop</td><td>int</td><td>25</td><td>Group Of Pictures: set the interval between random-access pictures. Available values range between 1 frame to 10 times the fps value (i.e. 10 seconds).</td></tr>
<tr><td> b_frames</td><td>int</td><td>2</td><td>Specifies the number of B frames in an IP(B)* GOP pattern. Available values ranges from 0 to 5. Ignored with the "mjpeg" video codec.</td></tr>
<tr><td> pass</td><td>int</td><td>1</td><td>The number of pass for video encoding. The higher the better quality. Available values range from 1 to 2.</td></tr>
<tr><td> bitrate_mode</td><td>string</td><td>"VBR"</td><td>The rate-control mode. Available values are "VBR" (Variable Bit Rate) and "CBR" (Constant Bit Rate).</td></tr>
<tr><td> max_video_file_chunk</td><td>int</td><td>-</td><td>When this value in bytes is set, the video writer will split the video file into multiple ones if the file size reaches this limit. Due to buffered frames, headers and trailers, a safety margin is taken, the video files chunks will be approximately 5% below this limit.</td></tr>
<tr><td> max_moov_size</td><td>int</td><td>-</td><td>Reserves at most this space for the moov atom at the beginning of the file instead of placing the moov atom at the end. The file writer will try to reserve dedicated space at the beginning of the files to fill the moov atom. If the estimated space needed is larger than max_moov_size the moov atom will be added at the end of the file and the file will be post processed. Disabled if max_video_file_chunk is not set.</td></tr>
<tr><td> min_moov_size</td><td>int</td><td>-</td><td>Reserves at least this space for the moov atom at the beginning of the file. Disabled if max_video_file_chunk is not set.</td></tr>
<tr><td> extra_params</td><td>string</td><td>""</td><td>Some custom coma-separated parameters to be directly pushed to the libav encoder. Example: "preset=superfast,profile=baseline"</td></tr>
</table>

#### Audio output

<table>
<tr><th>Member</th><th>Type</th><th>Default Value</th><th></th></tr>
<tr><td>audio_codec</td><td>string</td><td>"aac"</td><td>The audio codec. Available values are: "aac" and "mp3". Note: "mp3" supports only a "stereo" layout.</td></tr>
<tr><td>sample_format</td><td>string</td><td>"fltp"</td><td>The sample format. Available values are: "s8", "s16", "s32", "flt", "dbl", "s8p", "s16p", "s32p", "fltp" and "dblp" </td></tr>
<tr><td>sampling_rate</td><td>int</td><td>48000</td><td>The sampling rate in Hz. Available values are: 44100, 48000</td></tr>
<tr><td>channel_layout</td><td>string</td><td>"stereo"</td><td>The channel layout of the output (see below).</td></tr>
<tr><td>audio_bitrate</td><td>int</td><td>128</td><td>The audio bitrate in kilo bits per second (kbps). Available values are: 64, 128 and 192 kbps</td></tr>
</table>

##### "channel_layout"

The only supported and tested value is "stereo" for any audio codec and any output type. Please refer to each plugin documentation to see the supported configurations.

#### "jpg"

Jpeg output. One image per frame ("numbered", see section below).

<table>
<tr><th>Member</th><th>Type</th><th>Default value</th><th></th></tr>
<tr><th> filename</th><td>string</td><td>-</td><td>Output filename prefix.</td></tr>
<tr><td> quality</td><td>int</td><td>90</td><td>JPEG quality. Between 1 and 100.</td></tr>
</table>

#### "png"

PNG output. One image per frame ("numbered", see section below).

<table>
<tr><th>Member</th><th>Type</th><th>Default value</th><th></th></tr>
<tr><th> filename</th><td>string</td><td>-</td><td>Output filename prefix.</td></tr>
<tr><td> alpha</td><td>bool</td><td>false</td><td>Writes RGBA channels when set, otherwise RGB.</td></tr>
</table>

#### "depth-png"

PNG depth output. Depth encoded in millimeters at 16 bits per pixel. One image per frame ("numbered", see section below).

<table>
<tr><th>Member</th><th>Type</th><th>Default value</th><th></th></tr>
<tr><th> filename</th><td>string</td><td>-</td><td>Output filename prefix.</td></tr>
</table>


#### "tiff"

TIFF output. One image per frame ("numbered", see section below).

<table>
<tr><th>Member</th><th>Type</th><th>Default value</th><th></th></tr>
<tr><th> filename</th><td>string</td><td>-</td><td>Output filename prefix.</td></tr>
<tr><td> compression</td><td>string</td><td>"none"</td><td>TIFF compression. One of "none", "lzw", "packbits", "jpeg", "deflate".</td></tr>
</table>

#### "ppm"

PPM output. One image per frame ("numbered", see section below).

<table>
<tr><th>Member</th><th>Type</th><th>Default value</th><th></th></tr>
<tr><th> filename</th><td>string</td><td>-</td><td>Output filename prefix.</td></tr>
</table>

#### "pam"

PAM (PNM + alpha) output. One image per frame ("numbered", see section below).

<table>
<tr><th>Member</th><th>Type</th><th>Default value</th><th></th></tr>
<tr><th> filename</th><td>string</td><td>-</td><td>Output filename prefix.</td></tr>
</table>

#### "raw"

Raw output (raw unencapsulated RGBA data). One image per frame ("numbered", see section below).

<table>
<tr><th>Member</th><th>Type</th><th>Default value</th><th></th></tr>
<tr><th> filename</th><td>string</td><td>-</td><td>Output filename prefix.</td></tr>
</table>

#### "yuv420p"

Planar output. Three images per frame, one per color plane.

<table>
<tr><th>Member</th><th>Type</th><th>Default value</th><th></th></tr>
<tr><th> filename</th><td>string</td><td>-</td><td>Output filename prefix.</td></tr>
</table>

##### "numbered" writers

Numbered writers write the number of the stitched frame between the basename and the extension. It is advised for file based output, such as JPG, PNG, etc.

Here are the optional parameters.
<table>
<tr><th>Member</th><th>Type</th><th>Default value</th><th></th></tr>
<tr><th>numbered_digits</th><td>int</td><td>1</td><td>Number of digits: set 1 for no leading zeros, set 0 to ignore the numbering, any other positive value to get a zero-prefixed number</td></tr>
</table>

### Readers

How to read each input is specified using the reader_config member of the input. If you are reading from videos or image sequences you will most certainly only specify a filename.
If a relative name is given, it starts at the directory where the ptv file is.

However, some configurations need to specify complex configurations. In that case, the configuration is an object with a *type* field. The following types are recognized:

#### "procedural" readers

Procedural readers are used to automatically generate synthetic content, usually for testing. Most procedural readers generate their input directly in device memory using the GPU,
and are therefore extremely efficient. They can be used to assess performance.
The exact procedural reader to use is specified using the "name" field of the config. Then, each reader has specific options.

##### "color"

Fills the input with a single color specified in the 'color' field. The following config fills the input with solid red:

    { "type": "procedural", "name": "color", "color": "ff0000" }

##### "checker"

Fills the input with a color checkerboard of a given size. The background is filled with color1. The checker squares will be painted with a mix between color2 and color3, depending on the coordinates.

The following config fills the input with a red-and-white solid checker of size 32 pixels:

    { "type": "procedural", "name": "checker", "color1": "ff0000", "color2": "ffffff", "color3": "ffffff" "size": 32 }

The following config fills the input with a color gradient checker

    { "type": "procedural", "name": "checker", "color1": "000000", "color2": "eeeeee", "color3": "333333" "size": 32 }

##### "grid"

Fills the input with a wireframe grid of a given size. The following config fills the input with a red-on-transparent grid of size 32 pixels and line width 3 pixels:

    { "type": "procedural", "name": "grid", "color": "ff0000", "bg_color": "00000000", "size": 32, "line_width": 3 }

##### "expr"

Writes the result of evaluating an expression. The expression can be any integer expression using numerical constants and the following variables:
*cFrame* (the current stitcher frame), *rFrame* (The current reader reader frame; this can be different from the stitcher frame if there are temporal offsets), *inputId* (the id of the input, 0 to num_input - 1)

The following config writes the current stitcher frame in red:

    { "type": "procedural", "name": "expr", "color": "ff0000", "bg_color": "00000000", "value": "cFrame" }

##### "movingChecker"

A host-side input that creates a 32x32 black/white checkerboard pattern that moves 2 pixels horizontally and 4 pixels vertically with every frame. As the pattern is changing, it can be used to identify problems such as synchronization issues between readers or incorrectly written buffers / tearing.

The reader does not accept any arguments.

    { "type": "procedural", "name": "movingChecker" }

##### "profiling"

A host-side input that repeats the same frame over and over. The frame is filled with random data in the YV12 color space. Crude simulation of a perfect (0-CPU) host-side video decoder. The main difference to the other procedural readers is that the frame is created in host space in YV12, so it has to be copied to the GPU and be unpacked.

    { "type": "procedural", "name": "profiling" }


#### "shared" readers

Some cameras multiplex their output in a single image. In that case, all inputs must share a common *delegate* input to read the multiplexed stream,
and each input reads a portion of the resulting image. The portion to read is specified by its *offset* with the top-left corner of the delegate.
For example, the configuration below declares two *shared* inputs that read from the same delegate "test". The delegate reads a 1280 x 1024 video ("sphericam.mp4") that multiplexes four
640 x 512 streams. The first input reads the top-left portion (offset is (0,0)), while the second input reads the bottom-left one (offset is (0,512)).

    "inputs" : [
      {
        "width" : 640,
        "height" : 512,
        "reader_config" : {
          "type" : "shared",
          "shared_id" : "test",
          "delegate" : {
            "expected_width" : 1280,
            "expected_height" : 1024,
            "reader_config" : "sphericam.mp4"
          },
          "offset_x" : 0,
          "offset_y" : 0
        }
        (...)
      },
      {
        "width" : 640,
        "height" : 512,
        "reader_config" : {
          "type" : "shared",
          "shared_id" : "test",
          "delegate" : {
            "expected_width" : 1280,
            "expected_height" : 1024,
            "reader_config" : "sphericam.mp4"
          },
          "offset_x" : 0,
          "offset_y" : 512
        }
        (...)
      },
    ]

####  Merger Mask and Blending Order

It is sometimes better to have an optimized mask per input and a blending order for the inputs.
These values can be set manually by users or optimized automatically using our algorithm.
For example, the configuration below show the output pano width ("width") and height ("height") as 2792x1396.
These values should match with the current rig setting for the "merger_mask" to be considered as "valid".
The "enable" field shows whether the mask is being used.
The "interpolationEnabled" shows whether interpolation between different key masks is enabled. The key masks are stored in "masks" described next.

The "masks" stores all the computed seam at different frames. For a certain mask, "frameId" stores the frame index,
the total number of inputs ("input_index_count") at the computation time.
The "input_index_data" field contains the list of masks, one per input, stored as encoded polyline in the input space.
"input_indices" stores the index of the input in "input_index_data".

Finally, the "masks_order" shows the blending order.

    "merger_mask" : {
      "width" : 2792,
      "height" : 1396,
      "enable" : true,
      "interpolationEnabled" : true,
	  "masks" : [
        {
          "input_index_count" : 5,
          "frameId" : 0,
          "input_index_data" : [
            "",
            "",
            "",
            "",
            ""
          ],
          "input_indices" : [
            0,
            1,
            2,
            3,
            4
          ]          
        }
        (...)
      ],
      "masks_order" : [
        1,
        3,
        2,
        0
      ]
    }

### Calibration Control Points

The calibration algorithm returns the list of control points in the "Pano" tree.

    "calibration_control_points" : {
      "matched_control_points" : [
        {
          "frame_number" : 1301,
          "input_index0" : 0,
          "x0" : 1275.88,
          "y0" : 254.024,
          "input_index1" : 2,
          "x1" : 1030.24,
          "y1" : 1394.07,
          "score" : 0.362205
        },
        {
          "frame_number" : 1301,
          "input_index0" : 0,
          "x0" : 1289.4,
          "y0" : 136.963,
          "input_index1" : 2,
          "x1" : 1032.32,
          "y1" : 1311.55,
          "score" : 0.555556
        },
        {
          "frame_number" : 1301,
          "input_index0" : 0,
          "x0" : 1473.11,
          "y0" : 368.84,
          "input_index1" : 2,
          "x1" : 1264.67,
          "y1" : 1406.95,
          "score" : 0.587302
        },
        (...)
        {
          "frame_number" : 9208,
          "input_index0" : 1,
          "x0" : 1408.33,
          "y0" : 52.2616,
          "input_index1" : 3,
          "x1" : 1097.8,
          "y1" : 1227.7,
          "score" : 0.552239
        }
      ]
    }

### Preprocessors

It is sometimes desirable to preprocess the inputs to overlay information or modify the image before mapping.
To do that, you can use one or several optional preprocessors on each input. The processors are executed in the order they are specified.
Each preprocessor is identified by a *type* (and can take option). Here is a list of available types:

##### "tint"

Transforms the input by mapping its luminosity onto a single hue given by *color*. Alpha is ignored.

    "preprocessors" : {"type" : "tint", "color" : "ff3300"}

##### "expr"

Overlays the result of evaluating an expression on the input. Options are similar to the *expr* reader.

##### "grid"

Overlays a grid on the input. Options are similar to the *grid* reader.
