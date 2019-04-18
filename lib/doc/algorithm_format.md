# Algorithms file format specification

Algorithms spec is the file format used by videostitch-cmd. It uses full [JSON](http://www.json.org/) syntax. Algorithms are simply defined by populating the root object with a few named variables.

## Objects

VideoStitch will interpret the following objects. An object is of course free to have other members in addition to these.

### Root
videostitch-cmd will interpret the following optional members of the root object:

<table>
<tr><th>Member</th><th>Type</th><th>Default value</th><th></th></tr>
<tr><th>outputPtv</th><td>list</td><td>Optional</td><td>If provided, this value specifies the output ptv file that will contain the results of the "algorithms" and the settings from the input ptv.</td></tr>
</table>
The tables below show the members of the "outputPtv"
<table>
<tr><th>Member</th><th>Type</th><th>Default value</th><th></th></tr>
<tr><th> name</th><td>string</td><td>No Default</td><td>Names of the output ptv.</td></tr>
<tr><th> outputFile</th><td>list</td><td>Optional</td><td>An optional list to specify the format of the final output in the new ptv file.</td></tr>
</table>

#### outputFile
The "outputFile" contains two optional members.
<table>
<tr><th>Member</th><th>Type</th><th>Default value</th><th></th></tr>
<tr><th> name</th><td>string</td><td>Optional</td><td>An optional string to specify the name of the output file.</td></tr>
<tr><th> type</th><td>string</td><td>Optional</td><td>An optional string to specify the output extension.</td></tr>
</table>

An example setting

	  "outputPtv": [
	    {
	      "name" : "outputFinal.ptv",
	      "outputFile": {
	        "name" : "out-vs",
	        "type" : "jpg"
	      }
	    }
	  ]

### Algorithms

Algorithms specify which algorithms to be called from command-line
In the tables below, mandatory members are shown in bold. If not set, optional members take a default value specified in the "Default value" column.

<table>
<tr><th>Member</th><th>Type</th><th>Default value</th><th></th></tr>
<tr><th> name</th><td>string</td><td>mask</td><td>Type of algorithm. One of the types below.</td></tr>
</table>

#### "mask" algorithm:

The "mask" algorithm optimizes for the blending mask and blending order.

<table>
<tr><th>Member</th><th>Type</th><th>Default value</th><th></th></tr>
<tr><th> list_frames</th><td>list</td><td>[0]</td><td>The list of frames used for the optimization.</td></tr>
<tr><th> blending_order</th><td>bool</td><td>true</td><td>To specify whether the blending order is consider or not. If not, the order of inputs will be used as the blending order.</td></tr>
<tr><th> seam</th><td>bool</th><td>true</td><td>To specify whether the optimal seams are computed.</td></tr>
</table>

An example setting for the "mask" algorithm is:

	   {
	     "algorithms" : [
	     {
	       "name": "mask",
	       "config" : {
	         "list_frames" : [0],
	         "blending_order" : true,
	         "seam" : true
	       }
	     }
	     ]
	   }

#### "autocrop" algorithm:

The "autocrop" algorithm detects the crop circle for the fish-eye images.
The first frame of the inputs is used for detection.

<table>
<tr><th>Member</th><th>Type</th><th>Default value</th><th></th></tr>
<tr><th> neighborThreshold</th><td>int</td><td>40</td><td>A threshold used for the image binarization step. Indicates the similarity of neighboring pixels to be considered as a connected component.</td></tr>
<tr><th> differenceThreshold</th><td>int</th><td>500</td><td>A threshold used for the binarization step. Indicates the similarity of a pixel to the seed pixel to be considered as a connected component.</td></tr>
<tr><th> fineTuneMarginSize</th><td>int</th><td>100</td><td>A pre-defined range around the coarse circle's samples to look for the fine scale samples, e.g. 0 indicates that the fine scale samples are the coarse samples, while 100 indicates that the fine scale samples will be searched in the range [-100..100] px around the coarse samples.</td></tr>
<tr><th> scaleRadius</th><td>double</th><td>0.98</td><td>A parameter used to scale the fine scale circle's radius.</td></tr>
<tr><th> circleImage</th><td>bool</th><td>false</td><td>Whether to dump an image with the crop circle overlays on top of the original image.</td></tr>
</table>

An example setting for the "autocrop" algorithm is:

	   {
	     "algorithms" : [
	     {
	       "name": "autocrop",
	       "config" : {
	         "neighborThreshold" : 40,
	         "differenceThreshold" : 500,
	         "fineTuneMarginSize" : 100,
	         "circleImage" : false,
	         "scaleRadius" : 0.98		 
	       }
	     }
	     ]
	   }

#### "calibration" algorithm:

The "calibration" algorithm runs a calibration with presets for the rig.


<table>
<tr><th>Member</th><th>Type</th><th>Default value</th><th></th></tr>
<tr><th>config</th><td>list</td><td>Optional</td><td>The list of calibration configuration parameters.</td></tr>
<tr><th>apply_presets_only</th><td>bool</td><td>Optional</td><td>If true, applies the rig presets to the PanoDefinition without actually calibrating. Meant to obtain a template project definition out of inputs and presets.</td></tr>
<tr><th>improve_mode</th><td>bool</td><td>Optional</td><td>If true, calibration will reused past control points found in the JSON tree.</td></tr>
<tr><th>auto_iterate_fov</th><td>bool</td><td>Optional</td><td>If true, calibration will try to estimate the FOV of the lens.</td></tr>
<tr><th>single_focal</th><td>bool</td><td>Optional</td><td>If true, the optimizer will estimate a single (fu,fv) pair of parameters for all lenses in the rig.</td></tr>
<tr><th>dump_calibration_snapshots</th><td>bool</td><td>Optional</td><td>If true, calibration will dump pictures with control points at various stages of the procedure.</td></tr>
<tr><th>deshuffle_mode</th><td>bool</td><td>Optional</td><td>If true, calibration will extract key points and reorder the video inputs using the rig presets, before running the full calibration.</td></tr>
<tr><th>deshuffle_mode_only</th><td>bool</td><td>Optional</td><td>If true, calibration will extract key points and reorder the video inputs using the rig presets, without running the full calibration.</td></tr>
<tr><th>deshuffle_mode_preserve_readers_order</th><td>bool</td><td>Optional</td><td>If true, deshuffling will preserve the order of input readers in the returned PanoDefinition and reorder the geometries. Otherwise, the deshuffling will keep the order of geometries and reorder the input readers.</td></tr>
<tr><th>use_synthetic_keypoints</th><td>bool</td><td>false</td><td>If true, the calibration algorithm will generate artificial keypoints from the PanoDefinition geometry, to cover the input areas where no real keypoint was extracted and preserve the PanoDefinition geometry.</td></tr>
<tr><th>synthetic_keypoints_grid_width</th><td>int</td><td>5</td><td>Grid width to generate artificial keypoints in each input picture.</td></tr>
<tr><th>synthetic_keypoints_grid_height</th><td>int</td><td>5</td><td>Grid height to generate artificial keypoints in each input picture.</td></tr>
<tr><th>cp_extractor</th><td>list</td><td>Optional</td><td>The list of parameters for the control point detector and matcher.</td></tr>
<tr><th>extractor</th><td>string</td><td>Optional</td><td>The name of the control point detector.</td></tr>
<tr><th>matcher_norm</th><td>string</td><td>Optional</td><td>The name of the control point matcher.</td></tr>
<tr><th>octaves</th><td>int</td><td>Optional</td><td>The number of octaves used for the detection.</td></tr>
<tr><th>sublevels</th><td>integer</td><td>Optional</td><td>The number of sublevels used for the detection.</td></tr>
<tr><th>threshold</th><td>double</td><td>Optional</td><td>Detection threshold.</td></tr>
<tr><th>nndr_ratio</th><td>double</td><td>Optional</td><td>Ratio between the first and second best score to claim a right match.</td></tr>
<tr><th>cp_filter</th><td>list</td><td>Optional</td><td>Parameters for the control point filter, which uses RANSAC.</td></tr>
<tr><th>angle_threshold</th><td>double</td><td>Optional</td><td>Angle threshold to validate a control point rotated from one lens to the other.</td></tr>
<tr><th>min_ratio_inliers</th><td>double</td><td>Optional</td><td>Minimum ratio of inliers, before RANSAC.</td></tr>
<tr><th>min_samples_for_fit</th><td>integer</td><td>Optional</td><td>Minimum number of control points to run RANSAC.</td></tr>
<tr><th>proba_draw_outlier_free_samples</th><td>double</td><td>Optional</td><td>Target probability to achieve for RANSAC.</td></tr>
<tr><th>decimating_grid_size</th><td>double</td><td>Optional</td><td>Grid size for decimation (w.r.t. picture size).</td></tr>
<tr><th><a name="rig-preset"> rig </a></th><td>list</td><td>Optional</td><td>The list of rig preset parameters.</td></tr>
<tr><th>name</th><td>string</td><td>Mandatory</td><td>The name of the rig presets.</td></tr>
<tr><th>rigcameras</th><td>list</td><td>Mandatory</td><td>The list of camera presets.</td></tr>
<tr><th>yaw_mean</th><td>double</td><td>Mandatory</td><td>Mean value of the yaw rotation angle for the optimizer.</td></tr>
<tr><th>yaw_variance</th><td>double</td><td>Mandatory</td><td>Variance value of the yaw rotation angle for the optimizer.</td></tr>
<tr><th>pitch_mean</th><td>double</td><td>Mandatory</td><td>Mean value of the pitch rotation angle for the optimizer.</td></tr>
<tr><th>pitch_variance</th><td>double</td><td>Mandatory</td><td>Variance value of the pitch rotation angle for the optimizer.</td></tr>
<tr><th>roll_mean</th><td>double</td><td>Mandatory</td><td>Mean value of the roll rotation angle for the optimizer.</td></tr>
<tr><th>roll_variance</th><td>double</td><td>Mandatory</td><td>Variance value of the roll rotation angle for the optimizer.</td></tr>
<tr><th>camera</th><td>string</td><td>Mandatory</td><td>Camera preset name (from the list of cameras).</td></tr>
<tr><th><a name="cameras-presets">cameras</a></th><td>list</td><td>Mandatory</td><td>The list of camera(s) presets.</td></tr>
<tr><th>name</th><td>string</td><td>Mandatory</td><td>The name of the camera presets.</td></tr>
<tr><th>format</th><td>string</td><td>Mandatory</td><td>The format of the lens projection.</td></tr>
<tr><th>width</th><td>integer</td><td>Mandatory</td><td>The width of the camera pictures.</td></tr>
<tr><th>height</th><td>integer</td><td>Mandatory</td><td>The height of the camera pictures.</td></tr>
<tr><th>fu_mean</th><td>double</td><td>Mandatory</td><td>Mean value of the horizontal focal for the optimizer.</td></tr>
<tr><th>fu_variance</th><td>double</td><td>Mandatory</td><td>Variance value of the horizontal focal for the optimizer.</td></tr>
<tr><th>fv_mean</th><td>double</td><td>Mandatory</td><td>Mean value of the vertical focal for the optimizer.</td></tr>
<tr><th>fv_variance</th><td>double</td><td>Mandatory</td><td>Variance value of the vertical focal for the optimizer.</td></tr>
<tr><th>cu_mean</th><td>double</td><td>Mandatory</td><td>Mean value of the horizontal center of projection for the optimizer.</td></tr>
<tr><th>cu_variance</th><td>double</td><td>Mandatory</td><td>Variance value of the horizontal center of projection for the optimizer.</td></tr>
<tr><th>cv_mean</th><td>double</td><td>Mandatory</td><td>Mean value of the vertical center of projection for the optimizer.</td></tr>
<tr><th>cv_variance</th><td>double</td><td>Mandatory</td><td>Variance value of the vertical center of projection for the optimizer.</td></tr>
<tr><th>distorta_mean</th><td>double</td><td>Mandatory</td><td>Mean value of the "a" distortion parameter for the optimizer.</td></tr>
<tr><th>distorta_variance</th><td>double</td><td>Mandatory</td><td>Variance value of "a" distortion parameter for the optimizer.</td></tr>
<tr><th>distortb_mean</th><td>double</td><td>Mandatory</td><td>Mean value of the "b" distortion parameter for the optimizer.</td></tr>
<tr><th>distortb_variance</th><td>double</td><td>Mandatory</td><td>Variance value of "b" distortion parameter for the optimizer.</td></tr>
<tr><th>distortc_mean</th><td>double</td><td>Mandatory</td><td>Mean value of the "c" distortion parameter for the optimizer.</td></tr>
<tr><th>distortc_variance</th><td>double</td><td>Mandatory</td><td>Variance value of "c" distortion parameter for the optimizer.</td></tr>
<tr><th>list_frames</th><td>list</td><td>Optional</td><td>The list of frames used for the optimization.</td></tr>
</table>

### "calibration\_presets\_maker" algorithm:

The "calibration\_presets\_maker" algorithm takes an input PanoDefinition, and creates calibration presets (i.e. <a href="#rig-preset">"rig"</a> and <a href="#cameras-presets">"cameras"</a> JSON objects) from it, to be used by the "calibration" algorithm. 

The returned PanoDefinition also contains the calibration presets, and its control point list is cleared. 

<table>
<tr><th>Member</th><th>Type</th><th>Default value</th><th></th></tr>
<tr><th>output</th><td>string</td><td>none</td><td>The name of the JSON file receiving the results of the algorithm.</td></tr>
<tr><th>name</th><td>string</td><td>Mandatory</td><td>The name of the rig presets.</td></tr>
<tr><th>focal_std_dev_value_percentage</th><td>double</td><td>5.0</td><td>Standard Deviation of focal parameters, expressed in terms of percentage of the input values.</td></tr>
<tr><th>center_std_dev_width_percentage</th><td>double</td><td>10.0</td><td>Standard Deviation of center parameters, expressed in terms of percentage of the width of the input.</td></tr>
<tr><th>distort_std_dev_value_percentage</th><td>double</td><td>50.0</td><td>Standard Deviation of distortion parameters, expressed in terms of percentage of the input values.</td></tr>
<tr><th>yaw_std_dev</th><td>double</td><td>5.0</td><td>Standard Deviation of yaw angles, in degrees.</td></tr>
<tr><th>pitch_std_dev</th><td>double</td><td>5.0</td><td>Standard Deviation of pitch angles, in degrees.</td></tr>
<tr><th>roll_std_dev</th><td>double</td><td>5.0</td><td>Standard Deviation of roll angles, in degrees.</td></tr>
<tr><th>translation_x_std_dev</th><td>double</td><td>0.0</td><td>Standard Deviation of X translations, in meters.</td></tr>
<tr><th>translation_y_std_dev</th><td>double</td><td>0.0</td><td>Standard Deviation of Y translations, in meters.</td></tr>
<tr><th>translation_z_std_dev</th><td>double</td><td>0.0</td><td>Standard Deviation of Z translations, in meters.</td></tr>
</table>


An example setting for the "calibration\_presets\_maker" algorithm is:

	   {
	     "algorithms" : [
	       {
	         "name": "calibration_presets_maker",
	         "output": "output_presets.json",
	         "config" : {
	           "name" : "Orah Tetra 4i",
	           "focal_std_dev_value_percentage" : 15.0,
	           "center_std_dev_width_percentage" : 10.0, 
	           "distort_std_dev_value_percentage" : 50,
	           "yaw_std_dev" : 1.0, 
	           "pitch_std_dev" : 1.0, 
	           "roll_std_dev" : 1.0,
	           "translation_x_std_dev" : 0.07,
	           "translation_y_std_dev" : 0.07,
	           "translation_z_std_dev" : 0.
	         }
	       }
	     ]
	   }

An example output of the "calibration\_presets\_maker" algorithm is:

	{
	  "rig" : {
	    "name" : "Orah Tetra", 
	    "rigcameras" : [
	      {
	        "angle_unit" : "degrees", 
	        "yaw_mean" : 0, 
	        "pitch_mean" : 0, 
	        "roll_mean" : 0, 
	        "yaw_variance" : 16414, 
	        "pitch_variance" : 16414, 
	        "roll_variance" : 16414, 
	        "camera" : "camera_0"
	      },
	      ...
	      {
	        "angle_unit" : "degrees", 
	        "yaw_mean" : 93.068, 
	        "pitch_mean" : -67.2862, 
	        "roll_mean" : -68.2285, 
	        "yaw_variance" : 16414, 
	        "pitch_variance" : 16414, 
	        "roll_variance" : 16414, 
	        "camera" : "camera_3"
	      }
	    ]
	  }, 
	  "cameras" : [
	    {
	      "name" : "camera_0", 
	      "format" : "circular_fisheye_opt", 
	      "width" : 1920, 
	      "height" : 1440, 
	      "fu_mean" : 1035.6, 
	      "fu_variance" : 24130.5, 
	      "fv_mean" : 1046.15, 
	      "fv_variance" : 24624.7, 
	      "cu_mean" : 885.214, 
	      "cu_variance" : 36864, 
	      "cv_mean" : 814.362, 
	      "cv_variance" : 36864, 
	      "distorta_mean" : 0, 
	      "distorta_variance" : 0, 
	      "distortb_mean" : -0.401758, 
	      "distortb_variance" : 0.0403524, 
	      "distortc_mean" : 0, 
	      "distortc_variance" : 0
	    },
	    ...
	    {
	      "name" : "camera_3", 
	      "format" : "circular_fisheye_opt", 
	      "width" : 1920, 
	      "height" : 1440, 
	      "fu_mean" : 1035.6, 
	      "fu_variance" : 24130.5, 
	      "fv_mean" : 1046.15, 
	      "fv_variance" : 24624.7, 
	      "cu_mean" : 974.67, 
	      "cu_variance" : 36864, 
	      "cv_mean" : 690.774, 
	      "cv_variance" : 36864, 
	      "distorta_mean" : 0, 
	      "distorta_variance" : 0, 
	      "distortb_mean" : -0.311981, 
	      "distortb_variance" : 0.024333, 
	      "distortc_mean" : 0, 
	      "distortc_variance" : 0
	    }
	  ]
	}

#### "epipolar" algorithm (experimental, used for debugging purposes):

The "epipolar" algorithm takes a calibrated pano definition and a list of frames, of input points and/or matched input points to produce pictures showing:

* a spherical grid of points (optional)
* the location of points or point pairs
* the epipolar curves corresponding to the points or point pairs
* the estimated depth of point pairs

and outputs on the "Info" logger level the estimated error-free stitching distance. 
The point pairs can be automatically detected and matched. 

It also computes the rig minimum stitching distance (i.e. sphere scale) where points seen by at least 2 cameras become visible by only one as a floating metric point value in the output JSON file, and saves a equirectangular gray-level 8 bits picture "output\_min\_stitching\_distance.png" (which size is given by the output panorama size of the input panorama definition) where one intensity value is 1 cm. This scaling can be controlled by the **image\_max\_output\_depth** parameter below. 

<table>
<tr><th>Member</th><th>Type</th><th>Default value</th><th></th></tr>
<tr><th> list_frames</th><td>list</td><td>[0]</td><td>The list of frames used for the optimization.</td></tr>
<tr><th> spherical_grid_radius</th><td>double</th><td>0</td><td>If non-zero, generates a spherical grid of points with this radius.</td></tr>
<tr><th> auto_point_matching</th><td>bool</th><td>true</td><td>If true, automatically matching points, in addition to the ones provided.</td></tr>
<tr><th>decimating_grid_size</th><td>double</td><td>0.04</td><td>Grid size for decimation of matched points (w.r.t. picture size).</td></tr>
<tr><th> single_points</th><td>list</th><td>empty</td><td>List of single points.</td></tr>
<tr><th> input_index</th><td>int</th><td>mandatory</td><td>Element of single_points list: camera input index of a single point.</td></tr>
<tr><th> x</th><td>float</th><td>mandatory</td><td>Element of single_points list: "x" coordinate of a single point in its input.</td></tr>
<tr><th> y</th><td>float</th><td>mandatory</td><td>Element of single_points list: "y" coordinate of a single point in its input.</td></tr>
<tr><th> matched_points</th><td>list</th><td>empty</td><td>List of matched pairs of points.</td></tr>
<tr><th> input_index0</th><td>int</th><td>mandatory</td><td>Element of matched_points list: camera input index of the first point in pair.</td></tr>
<tr><th> x0</th><td>float</th><td>mandatory</td><td>Element of matched_points list: "x" coordinate of the first point in a pair, in its input.</td></tr>
<tr><th> y0</th><td>float</th><td>mandatory</td><td>Element of matched_points list: "y" coordinate of the first point in a pair, in its input.</td></tr>
<tr><th> input_index1</th><td>int</th><td>mandatory</td><td>Element of matched_points list: camera input index of the second point in a pair.</td></tr>
<tr><th> x1</th><td>float</th><td>mandatory</td><td>Element of matched_points list: "x" coordinate of the second point in a pair, in its input.</td></tr>
<tr><th> y1</th><td>float</th><td>mandatory</td><td>Element of matched_points list: "y" coordinate of the second point in a pair, in its input.</td></tr>
<tr><th> image_max_output_depth</th><td>float</th><td>2.55</td><td>Depth value mapped to 255 in the output depth image. Leaving it to 2.55 means that 255 will be mapped to 2.55 meters, i.e. one gray-level is 1 cm. </td></tr>
</table>

An example setting for the "epipolar" algorithm is:

	{
	  "algorithms" : [
	    {
	      "name": "epipolar",
	      "config" : {
	        "auto_point_selection" : false,
	        "single_points" : [
	          {
	            "input_index" : 0,
	            "x" : 1000,
	            "y" : 720
	          },
	          {
	            "input_index" : 1,
	            "x" : 600,
	            "y" : 600
	          }
	        ],
	        "matched_points" : [
	          {
	            "input_index0" : 3,
	            "x0" : 144,
	            "y0" : 445,
	            "input_index1" : 2,
	            "x1" : 76,
	            "y1" : 931
	          }
	        ],
	        "list_frames" : [
	          1,
	          200
	        ]
	      }
	    }
	  ]
	}

An example JSON output is:

	{
	  "minStitchingDistance" : 0.073859989643096924
	}

#### "scoring" algorithm:

The "scoring" algorithm takes a calibrated pano definition and produces a list of scores per frame: a score between 0 and 1, based on the normalized cross correlation of input pairs in overlap in the equirectangular projection, and the ratio of uncovered pixels (i.e. holes in the projection).

<table>
<tr><th>Member</th><th>Type</th><th>Default value</th><th></th></tr>
<tr><th>output</th><td>string</td><td>none</td><td>The name of the JSON file receiving the results of the algorithm.</td></tr>
<tr><th> first_frame</th><td>int</th><td>0</td><td>First frame number for the scoring.</td></tr>
<tr><th> last_frame</th><td>int</th><td>0</td><td>Last frame number for the scoring.</td></tr>
</table>

An example setting for the "scoring" algorithm is:

	   {
	     "algorithms" : [
	       {
	         "name": "scoring",
	         "output": "output_scoring.ptv",
	         "config" : {
	           "first_frame" : 0,
	           "last_frame" : 0
	         }
	       }
	     ]
	   }

An example output of the "scoring" algorithm is:

	  [
	    {
	      "frame_number" : 0,
	      "score" : 0.609535,
	      "uncovered_ratio" : 0.0705967
	    },
	    {
	      "frame_number" : 1,
	      "score" : 0.609598,
	      "uncovered_ratio" : 0.0705967
	    },
	    {
	      "frame_number" : 2,
	      "score" : 0.609391,
	      "uncovered_ratio" : 0.0705967
	    }
	  ]

