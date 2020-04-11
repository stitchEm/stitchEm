#define qt_dir_v "D:\dev\Qt\5.12.5\msvc2017_64"
#define qt_bin qt_dir_v + "\bin"
#define qt_platform qt_dir_v + "\plugins\platforms"
#define qt_format qt_dir_v + "\plugins\imageformats"
#define qt_audio qt_dir_v + "\plugins\audio"
#define vcpkg_dir "D:\dev\vcpkg\installed\x64-windows\bin"

; other I/O dependencies
Source: "{#vcpkg_dir}\tiff.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#vcpkg_dir}\libpng16.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#vcpkg_dir}\jpeg62.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#vcpkg_dir}\turbojpeg.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#vcpkg_dir}\openvr_api.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#vcpkg_dir}\portaudio_x64.dll"; DestDir: "{app}"; Flags: ignoreversion

; qt dependencies
Source: "{#qt_bin}\Qt5Core.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#qt_bin}\Qt5Gui.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#qt_bin}\Qt5Widgets.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#qt_bin}\Qt5Network.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#qt_bin}\Qt5OpenGL.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#qt_bin}\Qt5Multimedia.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#qt_audio}\qtaudio_windows.dll"; DestDir: "{app}\audio"; Flags: recursesubdirs ignoreversion
Source: "{#qt_format}\qgif.dll"; DestDir: "{app}\imageformats"; Flags: recursesubdirs ignoreversion
Source: "{#qt_format}\qico.dll"; DestDir: "{app}\imageformats"; Flags: recursesubdirs ignoreversion
Source: "{#qt_format}\qjpeg.dll"; DestDir: "{app}\imageformats"; Flags: recursesubdirs ignoreversion
Source: "{#qt_format}\qsvg.dll"; DestDir: "{app}\imageformats"; Flags: recursesubdirs ignoreversion
Source: "{#qt_format}\qtga.dll"; DestDir: "{app}\imageformats"; Flags: recursesubdirs ignoreversion
Source: "{#qt_format}\qtiff.dll"; DestDir: "{app}\imageformats"; Flags: recursesubdirs ignoreversion
Source: "{#qt_format}\qwbmp.dll"; DestDir: "{app}\imageformats"; Flags: recursesubdirs ignoreversion
Source: "{#qt_platform}\qminimal.dll"; DestDir: "{app}\platforms"; Flags: recursesubdirs ignoreversion
Source: "{#qt_platform}\qoffscreen.dll"; DestDir: "{app}\platforms"; Flags: recursesubdirs ignoreversion
Source: "{#qt_platform}\qwindows.dll"; DestDir: "{app}\platforms"; Flags: recursesubdirs ignoreversion

; Ceres dependencies
Source: "{#vcpkg_dir}\ceres.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#vcpkg_dir}\glog.dll"; DestDir: "{app}"; Flags: ignoreversion

; OpenCV dependencies
Source: "{#vcpkg_dir}\opencv_calib3d.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#vcpkg_dir}\opencv_core.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#vcpkg_dir}\opencv_features2d.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#vcpkg_dir}\opencv_imgproc.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#vcpkg_dir}\opencv_flann.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#vcpkg_dir}\opencv_video.dll"; DestDir: "{app}"; Flags: ignoreversion

; VideoStitch internals
Source: "{#bin_dir}\libvideostitch-gpudiscovery.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#bin_dir}\libvideostitch.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#bin_dir}\libvideostitch-gui.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#bin_dir}\libvideostitch-base.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#bin_dir}\{#ExeName}"; DestDir: "{app}"; Flags: ignoreversion

; rtmp dependencies
Source: "{#vcpkg_dir}\librtmp.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#vcpkg_dir}\zlib1.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#vcpkg_dir}\ssleay32.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#vcpkg_dir}\libeay32.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#vcpkg_dir}\libx264-157.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#vcpkg_dir}\libmp3lame.dll"; DestDir: "{app}"; Flags: ignoreversion

; ffmpeg dependencies
Source: "{#vcpkg_dir}\avcodec-58.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#vcpkg_dir}\avformat-58.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#vcpkg_dir}\avresample-4.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#vcpkg_dir}\avutil-56.dll"; DestDir: "{app}"; Flags: ignoreversion

Source: "D:\dev\Visual Studio 2017\VC\Tools\MSVC\14.16.27023\bin\Hostx64\x64\vcruntime140.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "D:\dev\Visual Studio 2017\VC\Tools\MSVC\14.16.27023\bin\Hostx64\x64\msvcp140.dll"; DestDir: "{app}"; Flags: ignoreversion