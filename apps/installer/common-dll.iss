#define qt_version GetEnv('QT_VERSION')
#define qt_dir GetEnv('QT_INSTALL')
#define qt_dir_v qt_dir + "\" + qt_version + "\msvc2013_64"
#define qt_bin qt_dir_v + "\bin"
#define qt_platform qt_dir_v + "\plugins\platforms"
#define qt_format qt_dir_v + "\plugins\imageformats"
#define qt_audio qt_dir_v + "\plugins\audio"

; rtmp dependencies
Source: "{#bin_dir}\libwinpthread-1.dll"; DestDir: "{app}"; Flags: ignoreversion

; other I/O dependencies
Source: "{#bin_dir}\libtiff-5.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#bin_dir}\libpng16.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#bin_dir}\jpeg62.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#bin_dir}\turbojpeg.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#bin_dir}\openvr_api.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#bin_dir}\portaudio_x64.dll"; DestDir: "{app}"; Flags: ignoreversion

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
Source: "{#bin_dir}\ceres.dll"; DestDir: "{app}"; Flags: ignoreversion

; OpenCV dependencies
Source: "{#bin_dir}\opencv_calib3d310.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#bin_dir}\opencv_core310.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#bin_dir}\opencv_features2d310.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#bin_dir}\opencv_imgproc310.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#bin_dir}\opencv_flann310.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#bin_dir}\opencv_video310.dll"; DestDir: "{app}"; Flags: ignoreversion

; VideoStitch internals
Source: "{#bin_dir}\libvideostitch-gpudiscovery.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#bin_dir}\libvideostitch.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#bin_dir}\libvideostitch-gui.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#bin_dir}\libvideostitch-base.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#bin_dir}\{#ExeName}"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#bin_dir}\vcredist_x64_2013.exe"; DestDir: "{app}"; Flags: deleteafterinstall ignoreversion

; rtmp dependencies
Source: "{#bin_dir}\librtmp.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#bin_dir}\librtmp-1.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#bin_dir}\zlib1.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#bin_dir}\ssleay32.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#bin_dir}\libeay32.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#bin_dir}\libx264-148.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#bin_dir}\libmp3lame.dll"; DestDir: "{app}"; Flags: ignoreversion

; ffmpeg dependencies
Source: "{#bin_dir}\avcodec-57.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#bin_dir}\avformat-57.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#bin_dir}\avresample-3.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#bin_dir}\avutil-55.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#bin_dir}\libx264-146.dll"; DestDir: "{app}"; Flags: ignoreversion
