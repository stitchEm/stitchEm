**VideoStitch Studio (stitchEm) v2.4**

**User Guide**

<br>

**Table of contents**

[**Getting started with VideoStitch Studio**](#getting-started-with-videostitch-studio)<br>
​    [1. Starting a new project](#1-starting-a-new-project)

[**VideoStitch Studio user interface**](#videostitch-studio-user-interface)<br>
​    [1. VideoStitch Studio panels](#1-videostitch-studio-panels)<br>
​    [2. Working with the timeline](#2-working-with-the-timeline)<br>
​    [3. Project and system information](#3-project-and-system-information)

[**VideoStitch Studio workflow**](#videostitch-studio-workflow)<br>
​    [**1. Synchronization**](#1-synchronization)<br>
​        [1.1 Automatic synchronization](#11-automatic-synchronization)<br>
​            [1.1.1 Audio based synchronization](#111-audio-based-synchronization)<br>
​            [1.1.2 Motion based synchronization](#112-motion-based-synchronization)<br>
​            [1.1.3 Flash based synchronization](#113-flash-based-synchronization)<br>
​        [1.2 Manual synchronization](#12-manual-synchronization)

​    [**2. Calibration**](#2-calibration)<br>
​        [2.1 Automatic calibration](#21-automatic-calibration)<br>
​        [2.2 Manual calibration](#22-manual-calibration)

​    [**3. Color Correction**](#3-color-correction)<br>
​        [3.1 Photometric parameters](#31-photometric-parameters)<br>
​        [3.2 Exposure compensation](#32-exposure-compensation)<br>
​        [3.3 Manual Adjustments](#33-manual-adjustments)

​    [**4. Stabilization & Orientation**](#4-stabilization---orientation)<br>
​        [4.1 Stabilization](#41-stabilization)<br>
​        [4.2 Orientation](#42-orientation)

​    [**5. Output configuration**](#5-output-configuration)

​    [**6. Output rendering**](#6-output-rendering)<br>
​        [6.1 General settings](#61-general-settings)<br>
​        [6.2 Encoder settings](#62-encoder-settings)<br>
​        [6.3 Audio settings](#63-audio-settings)<br>
​        [6.4 Batch stitcher](#64-batch-stitcher)

[**Useful Tips & Troubleshooting**](#useful-tips--troubleshooting)<br>
​    [1. Building your own 360° camera rig](#1-building-your-own-360-camera-rig)<br>
​    [2. Filming with a 360° camera rig](#2-filming-with-a-360-camera-rig)<br>
​    [3. Working with VideoStitch Studio - Troubleshooting](#3-working-with-videostitch-studio---troubleshooting)<br>
​    [4. Keyboard shortcuts](#4-keyboard-shortcuts)

[**Have fun creating breathtaking 360° content!**](#have-fun-creating-breathtaking-360-content)

<br>

<br>
<br>

# **Getting started with VideoStitch Studio**

<br>

## 1. Starting a new project

<br>

To start a new project you need to import the videos you want to stitch.
There are two ways to do so:

<br>

● Add media by clicking on "File" > "Open media"

![Open Media](https://github.com/stitchEm/stitchEm/blob/master/User%20Guide/Images/Open%20Media.jpg)**&nbsp;** 

You then have to select the videos you want to stitch in the file explorer.



● Alternatively you can drag the videos you want to stitch from your file explorer and drop into VideoStitch Studio user interface.

![Drag Media](https://github.com/stitchEm/stitchEm/blob/master/User%20Guide/Images/Drag%20Media.jpg)

<br>




After importing your media it will look like this and is listed in the "Source" panel:

![Media in Source](https://github.com/stitchEm/stitchEm/blob/master/User%20Guide/Images/Media%20in%20Source.jpg)

<br><br><br>

# **VideoStitch Studio user interface**

<br>

## 1. VideoStitch Studio panels

![Panels](https://github.com/stitchEm/stitchEm/blob/master/User%20Guide/Images/Panels.jpg)

<br>

There are four panels in VideoStitch Studio

1. **Source:** Displays the input videos

2. **Output\*:** Shows the stitched result in realtime

3. **Interactive\*:** Shows thestitched result in realtime from an interactive perspective

4. **Process\*:** Video and Output settings can be adjusted in this panel

*Note: You can only see “Output“, “Interactive“ & “Process“ after importing media.*



<img src="https://github.com/stitchEm/stitchEm/blob/master/User%20Guide/Images/Output.jpg" alt="Output"  />

*Output panel*



![Interactive](https://github.com/stitchEm/stitchEm/blob/master/User%20Guide/Images/Interactive.jpg)

*Interactive Panel*

<br>

## 2. Working with the timeline

The timeline of VideoStitch Studio allows you to preview your source videos and stitched result.
You can play, pause, select specific frames & time and set a working area.

![Timeline](https://github.com/stitchEm/stitchEm/blob/master/User%20Guide/Images/Timeline.jpg)



**Play button**<br>
On the top left of the timeline you can find the Play button. Depending on the panel you are in you can either view the source videos or preview the stitched result in real time.



**Select working area**<br>
Below the Play button you will find two timecodes. By default the first one is 00:00:00 and the second is the total time of the video. Changing these values will limit the working area which allows you to start a video later or end earlier than the original source. You can also grab the grey markers and drag them to the desired position in the timeline.
Synchronization, calibration, exposure will only be calculated for the working area.

![Timeline Limiter](https://github.com/stitchEm/stitchEm/blob/master/User%20Guide/Images/Timeline%20Limiter.jpg)



**Zoom in and out the timeline**<br>
For precise navigating in the timeline you can zoom in or out using the slider on the bottom right.
Zooming in also helps seeing the values for algorithms, like color correction more precisely.

![Zoom timeline](https://github.com/stitchEm/stitchEm/blob/master/User%20Guide/Images/Zoom%20timeline.jpg)



**Navigate to the desired frame**<br>
If you want to view a specific frame you can use the Shortcut "Ctrl+J". You also zoom in as mentioned above and then drag the orange marker to the desired time and use the arrow keys to jump one frame forward or backward. "Shift+Home" and "Shift+End" will select the first / last frame of your current working area.



**Keyframes** <br>
Keyframes for Stabilization, Orientation & Color Correction are automatically created by VideoStitch Studio.
If you click on a keyframe marker on the timeline and drag it up or down this will affect the corresponding function (e.g. increasing or decreasing exposure).
If you want to add a new keyframe you can do so with the shortcut "Ctrl+K". You can jump between keyframes with the shortcuts "K" (next keyframe) and "J" (previous keyframe).

<br>

## 3. Project and system information

At the bottom right of the interface, you will find more useful information:



1. Current output and preview size (2048*1024 by default when creating a project)	
2. GPU (Graphics Processing Unit) memory currently used by VideoStitch Studio
3. Graphics card model and its GPU memory
4. Rendered frame: The current frame selected in the timelime

![Bottom Information](https://github.com/stitchEm/stitchEm/blob/master/User%20Guide/Images/Bottom%20Information.jpg)



<br>

<br>

<br>

# **VideoStitch Studio workflow**

<br>

## **1. Synchronization**

While most 360° camera setups are synchronized hardware-wise nowadays, synchronization still is needed for DIY rigs. If you are using a DIY rig and not a 360° camera, synchronization is the first step in the stitching process otherwise you can skip this step.


*For optimal synchronization keep the following in mind:*

- *Select the highest possible framerate (fps) in camera (60 should be the minimum if there are moving objects in the scene)*

- *Rolling shutter can create issues when dealing with scenes with fast moving objects.*

- *Make sure you think about syncronisation while shooting; a strong sound (e.g. clapping) or turning the whole camera 360° for motion based sync.*

   

To open the synchronization tool, navigate to the bar at the top, click on "Window" > "Synchronization". The synchronization tool appears on the top left of the user interface.


![Synchronization Bar](https://github.com/stitchEm/stitchEm/blob/master/User%20Guide/Images/Synchronization%20Bar.jpg)**&nbsp;** 

<br>

### 1.1 Automatic synchronization

![Synchronizaition Auto](https://github.com/stitchEm/stitchEm/blob/master/User%20Guide/Images/Synchronizaition%20Auto.jpg)**&nbsp;** 

<br>

*Note: For optimal results make sure to select an appropriate work area which includes either a specific sound, a motion, like turning the camera or the strong change of lighting in the scene depending on the syncronizing mehtod you want to use.*



There are three different approches to synchronizing your source videos.



#### 1.1.1 Audio based synchronization

Videos are being automatically synchronized based on a loud sound, that stands out from the background noise. A loud clap or a dog training clicker can create such sounds, but it is still not recommended for noisy environments. Make sure the sound is within the working area!



#### 1.1.2 Motion based synchronization

Videos are being automatically synchronized based on a strong motion of the camera rig itself.
If your rig allows it, a short, but strong spin would be ideal for this synchronization method.
Make sure both, the start and end of the spin are within the working area.



#### 1.1.3 Flash based synchronization

Videos are being automatically synchronized based on a sudden change in lighting that needs to be visible from all cameras.
Turning on bright lights in a room or using synchronized professional flashes would work for this method.
Make sure the change of lighting is within the working area.

<br>

### 1.2 Manual synchronization

![Synchronizaition Manual](https://github.com/stitchEm/stitchEm/blob/master/User%20Guide/Images/Synchronizaition%20Manual.jpg)**&nbsp;** 

<br>

If your footage wasn't optimized for synchronization and it doesn't synchronize good enough with any of the automatic methods (ghosting on moving objects in the calibrated 360° panorama can be a sign for this) you can adjust the frame offset manually.
By clicking on "Manual" you will reach the Manual tab where you see the source video names and frame offset.



<img src="https://github.com/stitchEm/stitchEm/blob/master/User%20Guide/Images/ghosting.jpg" alt="ghosting" style="zoom: 67%;" />**&nbsp;** 

*badly synchronized / ghosting example* <br>

To get rid of the ghosting of moving objects in the 360° panorama go to a frame where you can clearly see the ghosting in the "Output" panel, enable input numbers ("Window" > "Output configuration" > "Show input numbers"), locate the source file that isn't synchronized well and the input number assigned to it and manually change the offset until no ghosting occurs. You can see the changes live in the "Output" and "Interactive" panel. Click play and see if there is any ghosting remaining – repeat if necessary.
You can also link source videos by clicking the checkbox on the right if you are certain some source videos are already synchronized. VideoStitch Studio will use this information to enhance the automatic synchronization.

<br>

## **2. Calibration**

**What is a calibration?**<br>
Calibration is essentially how the videos have to be moved around and stitched together to get a full 360° panorama. Make sure to synchronize your videos before heading to the "Calibration" tab.

VideoStitch Studio provides you an automatic calibration tool, but it is also possible to import a stitching template from PTGui or Hugin.



![Before the stitch](https://github.com/stitchEm/stitchEm/blob/master/User%20Guide/Images/Before%20the%20stitch.jpg)

*before calibration (source videos)*



![Stitched](https://github.com/stitchEm/stitchEm/blob/master/User%20Guide/Images/Stitched.jpg)

*stitched result after calibration (shown in "Output" panel)*

<br>

To open the calibration tool, navigate to the bar at the top, click on "Window" > "Calibration". The calibration tool appears on the top left of the user interface.

![Calibration Tab](https://github.com/stitchEm/stitchEm/blob/master/User%20Guide/Images/Calibration%20Tab.jpg)

<br>

### 2.1 Automatic calibration

VideoStitch Studio is able to automatically calibrate your footage using custom camera settings.

![Calibration Window](https://github.com/stitchEm/stitchEm/blob/master/User%20Guide/Images/Calibration%20Window.jpg)

<br>

**Custom parameters calibration**<br>
You can always try to keep everything on auto first. 
If that doesn't give you a good result you should create a custom calibration:

1. Select the used lens type (e.g Fullframe fisheye for GoPro)

2. Specify the FOV of your cameras (You'll most likely find this in your cameras manual or online!)

3. Click on "Calibrate on sequence"



**Crop source videos**<br>
You can crop  source videos. 
This is especially useful for when you are filming with high FOV lenses. 
Our example project uses 3 cams with a FOV of ~200°. You can adjust by grabbing the orange circle and drag it to the desired position. In our example we can easily crop out the parts we don't want to use for our 360° video.



![Inputs crop2](https://github.com/stitchEm/stitchEm/blob/master/User%20Guide/Images/Inputs%20crop2.jpg)

*Source Video*



![Inputs crop](https://github.com/stitchEm/stitchEm/blob/master/User%20Guide/Images/Inputs%20crop.jpg)

*Cropped video*

<br>

**Manual frame selection**<br>
VideoStitch Studio will automatically choose frames from the calibration sequence you defined as reference for the calibration. If you want more control, you can calibrate on frames you select yourself.



![Frame selection](https://github.com/stitchEm/stitchEm/blob/master/User%20Guide/Images/Frame%20selection.jpg)

<br>

*To calibrate using specified frames:*

1. Click on the "Manual frame selection" checkbox
2. Select a frame by placing the orange marker on the desired frame
3. Click on "Add frame"
4. Repeat steps 2 and 3 until you are satisfied with your frame list
5. Launch the calibration ("Calibrate")



You can remove one or all frames by clicking on "Remove selected" and "Clear all".

<br>

### 2.2 Manual calibration

In some cases the automatic calibration might not be precise enough or doesn't work at all. There are multiple reasons for this; Objects close to the rig, not enough detail (e.g. blue sky) or not enough overlap between images.
To solve this VideoStitch Studio is compatible with templates from PTGui and Hugin *–* both are thrid party software solutions made specifically for stitching panoramic images.

***Note: VideoStitch Studio will import the exact position of every source file in the 360° sphere and all red masks. Most other adjustments will not be imported!
 If you want to use masks make sure to only use the red masks.***

If you want to work with PTGui or Hugin you need to create the template using still images. VideoStitch Studio helps you with exporting images.
Go to "Edit" > "Extract stills to..." OR "Extract stills" and it will export image files you can use. Make sure to keep the names and order the same to be able to quickly apply the template in VideoStitch Studio once it is done.

![Calibration Extract Stills](https://github.com/stitchEm/stitchEm/blob/master/User%20Guide/Images/Calibration%20Extract%20Stills.jpg)

<br>

In the Calibration window select "Import a template" > "Browse calibration..." and select the file in the file browser. If you have used a template in the past and want to use it again simply click on "Recent calibrations" and apply the desired one.

![Import Calibration](https://github.com/stitchEm/stitchEm/blob/master/User%20Guide/Images/Import%20Calibration.jpg)

<br>

## **3. Color Correction**

Color correction optimizes the stitched 360° video by automatically adjusting vignetting, camera response curves, exposure and white balance. Depending on the videos you want to stitch you can also choose to only adjust some of the settings.

To open the color correction tool, navigate to the bar at the top, click on "Window" > "Color correction". The color correction tool appears on the top left of the user interface.

![Color correction](https://github.com/stitchEm/stitchEm/blob/master/User%20Guide/Images/Color%20correction.jpg)**&nbsp;** 

<br>

### 3.1 Photometric parameters

![Color Corretion base tab](https://github.com/stitchEm/stitchEm/blob/master/User%20Guide/Images/Color%20Corretion%20base%20tab.jpg)**&nbsp;** **&nbsp;** 

By clicking "Calibrate photometry" under "Photometric parameters" VideoStitch Studio will automatically calculate the response curves and vignetting of the used cameras to improve the blending of the stitched 360° panorama.

***Vignetting*** describes the light fall-off in the frame; corners usually appear darker than the center. Darker edges and corners are sometimes wanted in photography, but it is a problem for stitching and blending videos together.

Explaining ***camera response*** would take to much time here, but is basically the relation between the amount of real world light coming in and the video pixel values the camera takes. 

"Vignette" and "Camera response" tabs will appear under "Photometric parameters".

![Color Correction window](https://github.com/stitchEm/stitchEm/blob/master/User%20Guide/Images/Color%20Correction%20window.jpg)

![ColorCorrection Camera response](https://github.com/stitchEm/stitchEm/blob/master/User%20Guide/Images/ColorCorrection%20Camera%20response.jpg)

<br>

### 3.2 Exposure compensation

VideoStitch Studio automatically analyzes the input videos and calculates exposure and white balance adjustments. 
You can choose to either "Adjust on current frame" or "Adjust sequence" which calculates a new exposure (and white balance if selected) at a specific frame interval and applies an interpolated exposure (and white balance) to frames in between.



**Example**<br>![Color Correction Examples](https://github.com/stitchEm/stitchEm/blob/master/User%20Guide/Images/Color%20Correction%20Examples.jpg)



**Advanced parameters**<br>
You can set advanced parameters by ticking the box next to "Advanced parameters".

![Color Correction advanced parameters](https://github.com/stitchEm/stitchEm/blob/master/User%20Guide/Images/Color%20Correction%20advanced%20parameters.jpg)



**Adjust every **<br>
Allows you to change the interval between each exposure (& white balance) adjustment, frames in between will be interpolated.

Depending on how the light changes in you scene a lower value will deliver better results, if the lighting doesn't change in your scene, setting a higher frame interval is okay, too.



**Anchor**<br>
Allows you to anchor the exposure to a specified source video that has the correct color (even lighting – not pointing to a bright light source or too dark area).
This specified source video will be used as a reference for the color correction. 
You can also pick "All" and VideoStitch Studio will compare colors between source videos automatically.

<br>

### 3.3 Manual Adjustments

You can manually adjust the exposure as well as the Red and  Blue correction by heading to the timeline, expanding the "Exposure compensation" window and manually dragging the keyframe markers up (increase exposure/correction value) or down (decrease exposure/correction value).

![keyframes exposure](https://github.com/stitchEm/stitchEm/blob/master/User%20Guide/Images/keyframes%20exposure.jpg)

<br>

## **4. Stabilization & Orientation**

**Stabilization** is useful if your videos were not shot on a solid tripod. Drone or handheld footage without a gimbal are good examples, but sometimes even a tripod can shake if the environment is rather windy. Stabilizing will smooth out the movements and vertical bumping.

**Orientation** is used if the horizon in your stitched panorama isn't right. Straight vertical lines that look tilted in your stitched panorama are often a sign that you need to set the horizon right. You can also choose the starting perspective by rotating your panorama without adjusting the horizon.
Depending on how the cameras are arranged in your camera rig your panorama orientation might be completely wrong like in the panorama below.

*![After Stitch](https://github.com/stitchEm/stitchEm/blob/master/User%20Guide/Images/Orientation.jpg)*

*Wrong Orientation*

![Orientation2](https://github.com/stitchEm/stitchEm/blob/master/User%20Guide/Images/Orientation2.jpg)

*Wrong Orientation, too*

<br>

To open the stabilization & orientation tool, navigate to the bar at the top, click on "Window" > "Stabilization & Orientation". The stabilization & orientation tool appears on the top left of the user interface.

![Stabilization & Orientation](https://github.com/stitchEm/stitchEm/blob/master/User%20Guide/Images/Stabilization%20Orientation.jpg)**&nbsp;** **&nbsp;** 

![Stabilization & Orientation Window](https://github.com/stitchEm/stitchEm/blob/master/User%20Guide/Images/Stabilization%20Orientation%20Window.jpg)**&nbsp;** **&nbsp;** 

<br>

### 4.1 Stabilization

To stabilize the footage:

1. Set the working area you want to stabilize

2. Click on "Stabilize sequence"



*If you are not happy with the result you can set a different working area, click on "Reset sequence" and "Stabilize sequence" again.*

<br>

### 4.2 Orientation

To change the orientation:

1. Select the frame on which you want to change the orientation

2. Click on "Adjust on current frame"

3. Click and drag your mouse on the output to straighten out the horizon and change orientation 


*Select any frame for stationary footage, otherwise you have to make sure to select a frame where you can easily see certain points as references. (e.g. straight lines, trees, the ocean)*

<br>

## **5. Output configuration**

To open the output configuration tool, navigate to the bar at the top, click on "Window" > "Output configuration". The output configuration tool appears on the top left of the user interface.

![Output Configuration](https://github.com/stitchEm/stitchEm/blob/master/User%20Guide/Images/Output%20Configuration.jpg)**&nbsp;** **&nbsp;** 


There are two options in the output configuration window:

![Output Configuration Window](https://github.com/stitchEm/stitchEm/blob/master/User%20Guide/Images/Output%20Configuration%20Window.jpg)

<br>

**Show input numbers** overlays the assigned input number of each source video on the stitched panorama. Very helpful if you have to locate a certain source video for optimizing syncronization and problems that might occur.



**Blending** offers two options.
You can select the *blending method* and set it to either linear or multiband.
Linear will render faster and is especially usefull for previewing your stitched footage live, multiband offers a higher quality and will take longer to compute.

The *feathering* slider allows you to control how sharp or smooth the blending between source videos will be.
While a smoother blending is useful for scenes without sharp detail (e.g. blending blue sky) it might create some ghosting where parallax is visible. Sharp blending is useful for keeping all the detail (e.g. text or mosaic), some seames might look too hard.
Try to find the perfect level of sharpness for you scene by trying and checking in the live preview of the "Output" panel.

<br>

## **6. Output rendering**

![Process Panel](https://github.com/stitchEm/stitchEm/blob/master/User%20Guide/Images/Process%20Panel.jpg)

<br>
To export your 360° video or change the preview resolution you have to go to the "Process“ panel.

<br>

### 6.1 General settings



**File**<br>
Change the output filename and path. The file is created in the same directory as your input videos by default. Click on "Browse" and you can select thelocation in the file explorer.



**Size**<br>
Resolution of the live-preview and output. Remember the output will always automatically be a 2:1 format as VideoStitch Studio will render a 360*180 output.

By clicking "Set optimal size" VideoStitch Studio will pick the maximal true resolution for the panorama video without interpolation. 
The optimal size may differ even if you always use the same rig. The reason for this is the difference in overlap for scenes with objects close by and open environments.



**Length**<br>
You can select if you want to output the whole video or just the current working area ("Process selected sequence") .

<br>

### 6.2 Encoder settings

![Advanced settings](https://github.com/stitchEm/stitchEm/blob/master/User%20Guide/Images/Advanced%20settings.jpg)

<br>

**Format**<br>
Switch between ***MP4*** or ***MOV*** Video formats or export an image sequence as ***JPEG***, ***PNG*** or ***TIFF.***



**Codec**<br>
Choose your desired codec: ***ProRes***, ***H264***, ***H264 (Nvidia NVENC), HEVC (Nvidia NVENC)*** ***MPEG2*** and ***Motion JPEG***.

**Profile** (ProRes)<br>
You can choose between Proxy, LT, Standard & High Quality.
If you want to continue working on your 360° video in an editing software like Adobe Premiere or Davinci Resolve you usually want to export „High Quality“.



**Bitrate mode** (H*264*, *H264 (Nvidia NVENC), HEVC (Nvidia NVENC) & MPEG2*)<br>
You can choose between VBR (variable bitrate) and CBR(constant bitrate).



**Bitrate** (H*264*, *H264 (Nvidia NVENC), HEVC (Nvidia NVENC) & MPEG2*)<br>
*File size = bitrate (kilobits per second) \* duration*<br>
Higher bitrate means higher quality and bigger filesize. The value heavily depends on your desired platform; if you are exporting specifically for a certain VR headset you can usually find the right values online. 



**Quality Scale** (Motion JPEG)<br>
You can drag the slider to choose between more quailty vs more compression.



**Advanced settings**<br>
If you enable advanced settings you will be able to customize the GOP and B-Frames number.
If you don't know what this is either look it up or keep as is, as explaining it here would take (p)ages.



**Which encoding setting to pick?**<br>
Ask yourself what you are using your panorama video for; if you want to render a quick preview a small H264 is enough, if you want to edit your footage in an editing software like Adobe Premiere Pro or Davinci Resolve you need to export either a High Quality MOV or an image sequence for maximum quality.

<br>

### 6.3 Audio settings

![Audio Settings](https://github.com/stitchEm/stitchEm/blob/master/User%20Guide/Images/Audio%20Settings.jpg)

<br>

By default the audio settings are disabled, to activate the settings, click on the "Audio settings" checkbox.



**Source**<br>
Select which source videos audio you want to use.



**Codec**<br>
Choose if your audio should be in ***MP3*** or ***AAC***.



**Bitrate**<br>
Choose the bitrate of the Audio in kbps: **64**, **96**, **128**, **192**, **256**, **512**



**Sample rate**<br>
Sample rate is fixed to ***44100Hz*** for ***MP3*** output and ***48000Hz*** for ***AAC***



**Channel layout**<br>
VideoStitch Studio outputs Stereo audio.

<br>

### 6.4 Batch stitcher

![Send to batch](https://github.com/stitchEm/stitchEm/blob/master/User%20Guide/Images/Send%20to%20batch.jpg)

<br>

If you want to prepare multiple videos for stitching and stitch them all later you can select "Send to batch". You will then be asked to create a copy of your project which is send to the VideoStitch Studio batch stitcher. You can close the current project and open the next, and also send it to the batch stitcher once done.

You can remove, reset or edit a project by right clicking on the project name in the batch stitcher.



![Batch Stitcher](https://github.com/stitchEm/stitchEm/blob/master/User%20Guide/Images/Batch%20Stitcher.jpg)



<br>

<br>

<br>

# **Useful Tips & Troubleshooting**



### 1. Building your own 360° camera rig



- Make sure the FOV is big enough to cover the whole sphere and there is enough overlap between videos.

   

- If possible it is advisable to build a synchronized rig, because even if your synchonization in VideoStich Studio is perfect there still is the offset between frames.

   

- Keep the desired output resolution in mind.

  

   <br>

### 2. Filming with a 360° camera rig



- Make sure to select the highest possible framerate for the resolution if your rig isn't synchronized in itself. This minimizes the offset between movements that happen in between frames.

   

- Make sure to look around and try to see exactly what your camera sees. Remember everything is visible!

   

- Hiding lighting is hard in 360° videos, but you can place a cylindrical light below your camera (at the tripod).
  
  <br>

### 3. Working with VideoStitch Studio - Troubleshooting



- **Live preview in Output and Interactive panel is very slow.**<br>
  	Try setting a lower "Size" in the "Process" panel.

  

- **I see ghosting, but can't figure out which source videos are affected.**<br>
   	Enable „Show input numbers“ in the „Output configuration“.



- **Clicking „Set optimal size“ gives a different result for different scenes shot with the same camera rig.**<br>
  This is to be expected, as the stitching is dependend on the distance of objects to the camera because of  the overlap between videos. A shot on the ocean would 	deliver a smaller resolution than a shot in a small room, because there is less 	overlap in the small room.



- **The stitched 360° video is too bright / dark!**<br>
  While this can obviously be fixed in any video editing software you can also 	manually drag the exposure keyframes down / up for the stitched panorama to 	increase / decrese exposure.
  
  

- **I created a template in PTGui / Hugin, but my stitched video looks different in Videostitch Studio!**<br>
  Make sure to keep the import order just the same as in VideoStitch Studio. 
  Keep in mind VideoStitch Studio imports the position of every source video and the red masks (mask out) from PTGui or Hugin, most other adjustments are not imported.



- **Stitchlines between images are looking to soft / hard!**<br>
  Head to the "Output configuration" window, go to "Blending" and adjust the "Feather" slider.

<br>

### 4. Keyboard shortcuts

![Shortcuts](https://github.com/stitchEm/stitchEm/blob/master/User%20Guide/Images/Shortcuts.jpg)**&nbsp;** **&nbsp;** 

<br>

**Check out the project on Github:<br>**https://github.com/stitchEm/stitchEm



**Join the VideoStitch Studio (2020) Users group on Facebook:**<br>
https://www.facebook.com/groups/VideoStitch



<br>

<br>

<br>

# **Have fun creating breathtaking 360° content!**