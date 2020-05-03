[license-badge]: https://img.shields.io/badge/license-MIT-green.svg
[license-link]: https://opensource.org/licenses/MIT

[actions-badge]: https://github.com/stitchEm/stitchEm/workflows/Build/badge.svg
[actions-link]: https://github.com/stitchEm/stitchEm/actions

[![License][license-badge]][license-link]
[![Build][actions-badge]][actions-link]

Vahana VR & VideoStitch Studio: software to create immersive 360° VR video, live and in post-production.

The software was originally developed by VideoStitch SAS. After the company folded in early 2018, the source code was acquired by the newly founded non-profit organization stitchEm, to publish it under a free software license.

### Vahana VR
Vahana VR is a camera-rig independent real-time 360° VR video stitching software.

With Vahana VR you can capture multiple video streams, stitch them instantly into a single 360°×180° video file, preview the result in real-time and stream it to any 360° video player or compatible platform.

Vahana VR supports range of video capture and output hardware, including Decklink SDI cards from BlackMagic Design, and HDMI capture cards from Magewell, Ximea cameras and RTMP network broadcasting.

![Screenshot Vahana VR](https://stitchEm.github.io/images/Screen-Shot-Vahana-VR.png)

### VideoStitch Studio

VideoStitch Studio is a post-production video stitching software that enables you to create immersive 360° VR videos. It takes video files and stitches them into a standard 360° video file automatically.

VideoStitch Studio supports input synchronization, automatic calibration, exposure and color correction, video stabilisation and a range of different output formats.

![Screenshot VideoStitch Studio](https://stitchEm.github.io/images/Screen-Shot-VideoStitch-Studio.png)

### Orah 4i

The Orah 4i is a camera with 4 fisheye lenses designed for live 4K 360° Video streaming. Information on setting up the camera is found in a the dedicated [camorah](https://github.com/stitchEm/camorah) respository.

## Download

Builds of Vahana VR and VideoStitch Studio for Windows and macOS can be downloaded in the [release section](https://github.com/stitchEm/stitchEm/releases/latest).

## Build

The software can be built on Windows, macOS and Linux. Not all features are available on all platforms. See [doc/BUILD.md](doc/BUILD.md) for details.

## License

The stitchEm project is licensed under the terms of the [MIT License](LICENSE.md).

The repository includes several third-party open source files, which are licensed under their own respective licenses. They are listed in [doc/LICENSE-3RD-PARTY-SOURCES.md](doc/LICENSE-3RD-PARTY-SOURCES.md).

The stitchEm software uses code from third-party libraries, listed in [doc/LICENSE-3RD-PARTY-LIBRARIES.md](doc/LICENSE-3RD-PARTY-LIBRARIES.md).
