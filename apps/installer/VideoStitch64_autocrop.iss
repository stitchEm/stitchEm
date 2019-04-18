#define MyAppName "VideoStitch Autocrop"
#define VersionName "Autocrop-v1"
#define ExeName "autocrop-cmd.exe"
#define bin_dir "..\bin\x64\release"

[Setup]
AppID=E336E779-59CA-44FA-872A-835FE9D4A7E8
SetupIconFile="src\resources\icons\vs\videostitch.ico"
AppMutex==VideoStitch-A269AB24-4B56-4EF4-A108-CF5D78C5026A
AppVersion=__APPVERSION__
OutputBaseFileName=VideoStitch-Autocrop-Setup
VersionInfoVersion=__INFOVERSION__
WizardImageAlphaFormat = defined
WizardImageFile="src\resources\installer\orah_side.bmp"
WizardSmallImageFile = "src\resources\installer\orah_small.bmp"
#include "common-setup.iss"

[Dirs]
Name: "{app}"; Flags: uninsalwaysuninstall

[Files]
; Ceres dependencies
Source: "{#bin_dir}\ceres.dll"; DestDir: "{app}"; Flags: ignoreversion

; OpenCV dependencies
Source: "{#bin_dir}\opencv_core310.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#bin_dir}\opencv_imgproc310.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#bin_dir}\opencv_imgcodecs310.dll"; DestDir: "{app}"; Flags: ignoreversion

; VideoStitch autocrop
Source: "{#bin_dir}\{#ExeName}"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#bin_dir}\{#ExeName}"; DestDir: "{app}"; Flags: ignoreversion
Source: "installer\docs\AutocropReadMe.txt"; DestDir: "{app}"; Flags: ignoreversion

; VideoStitch external deps
Source: "{#bin_dir}\libpng16.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#bin_dir}\libeay32.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "installer\vcredist_x64_2013.exe"; DestDir: "{app}"; Flags: deleteafterinstall ignoreversion

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#ExeName}"; IconFilename: "{app}\{#ExeName}"
Name: "{commondesktop}\{#MyAppName}"; Filename: "{app}\{#ExeName}"; WorkingDir: {app}; Tasks: desktopicon

[Run]
Filename: "{app}\vcredist_x64_2013.exe"; Parameters: "/q /norestart"; WorkingDir: "{app}"; StatusMsg: "Installing Microsoft Visual C++ 2013 Redistributable Package (x64)"

[Tasks]
Name: desktopicon; Description: "Create a &desktop icon"; GroupDescription: "Additional icons:"

[InstallDelete]
Type: filesandordirs; Name: "{commondesktop}\{#MyAppName}"
Type: filesandordirs; Name: "{commonprograms}\{#MyAppName}"
Type: filesandordirs; Name: "{group}"

#include "common-uninstall.iss"
