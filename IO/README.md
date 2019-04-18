# IO

This folder contains the IO plugins for VideoStitch applications. A plugin is an IO library that can be used by all the applications using the VideoStitch library.
Interfaces are defined in the VideoStitch library. Each plugin declares the used interfaces in its export.cpp file.

I/O plugins link against libvideostitch. They are loaded at runtime from libvideostitch.

A plugin is most the time an adapter using a third party library.

Two types of plugins are used:

 * Core plugins: I/O from files, including raw and network streams. Used in Studio and Vahana VR.
 * Vahana plugins: I/O from external hardware devices, trough acquisition cards. Used in Vahana VR.

The plugin documentation can be found on its folder `src/<plugin>/README.md`.

# Build

Building any I/O plugin can be turned off by globally by setting `BUILD_IO_PLUGINS=OFF`.

Indidividual plugins are turned off on some systems where 3rd party libraries may not be available, and can be disabled with these CMake flags:

| Option            | Default                 | Comments                |
|:------------------|:------------------------|:------------------------|
| DISABLE_AV        | OFF                     |                         |
| DISABLE_BMP       | ON                      | Used for debugging only |
| DISABLE_JPEG      | ${ANDROID}              |                         |
| DISABLE_TIFF      | ${ANDROID}              |                         |
| DISABLE_MP4       | ${NANDROID}             | Uses Android Media SDK  |
| DISABLE_RTMP      | OFF                     |                         |
| DISABLE_PORTAUDIO | ${CMAKE_CROSSCOMPILING} |                         |
