#define MyAppName "VideoStitch Studio"
#define VersionName "Studio"
#define ExeName "videostitch-studio.exe"
#define batch "batchstitcher.exe"
#define batchName "VideoStitch Batch"
#define bin_dir "..\bin\x64\release"

[Setup]
AppID=49F4C00B-EB5D-4709-AB5B-9BFC8DD35AE2
SetupIconFile="src\resources\icons\vs\videostitch.ico"
AppMutex==VideoStitch-C46F88B1-1FBD-4280-AA22-F358AFB019BD
AppVersion=__APPVERSION__
OutputBaseFileName=VideoStitch-Studio-Setup
VersionInfoVersion=__INFOVERSION__
WizardImageAlphaFormat = defined
WizardImageFile="src\resources\installer\orah_side.bmp"
WizardSmallImageFile = "src\resources\installer\orah_small.bmp"
#include "common-setup.iss"

[REGISTRY]
Root: HKLM; Subkey: "System\CurrentControlSet\Control\GraphicsDrivers"; ValueType: dword; ValueName: "TdrDelay"; ValueData: "10"; Flags: uninsdeletevalue
Root: HKCR; Subkey: ".ptv"; ValueType: string; ValueName: ""; ValueData: {#MyAppName}; Flags: uninsdeletevalue
Root: HKCR; Subkey: ".ptvb"; ValueType: string; ValueName: ""; ValueData: {#MyAppName}; Flags: uninsdeletevalue
Root: HKCR; Subkey: "{#MyAppName}"; ValueType: string; ValueName: ""; ValueData: {#MyAppName}; Flags: uninsdeletekey
Root: HKCR; Subkey: "{#MyAppName}\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\{#ExeName},0"
Root: HKCR; Subkey: "{#MyAppName}\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#ExeName}"" ""%1"""

[Dirs]
Name: "{app}"; Flags: uninsalwaysuninstall
Name: "{app}\platforms"; Flags: uninsalwaysuninstall
Name: "{app}\imageformats"; Flags: uninsalwaysuninstall

[Files]
#include "common-dll.iss"
#include "core-plugins-dll.iss"
#include "__GPU__-dll.iss"
__OTHER_GPU__

Source: "{#bin_dir}\videostitch-cmd.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#bin_dir}\batchstitcher.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#bin_dir}\calibrationimport.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#bin_dir}\ptvb2ptv.exe"; DestDir: "{app}"; Flags: ignoreversion

Source: "installer\droplets\1-calibrationimport.bat"; DestDir: "{app}"; Flags: ignoreversion
Source: "installer\droplets\2-process-ptv.bat"; DestDir: "{app}"; Flags: ignoreversion
Source: "installer\droplets\3-batch-process-ptv.bat"; DestDir: "{app}"; Flags: ignoreversion
Source: "installer\droplets\dropletusage.txt"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#ExeName}"; IconFilename: "{app}\{#ExeName}"
Name: "{group}\{#batchName}"; Filename: "{app}\{#batch}"; IconFilename: "{app}\{#batch}"
;Name: "{group}\Uninstall {#MyAppName}"; Filename: "{uninstallexe}" ; create a shortcut in the startmenu
Name: "{commondesktop}\{#MyAppName}"; Filename: "{app}\{#ExeName}"; WorkingDir: {app}; Tasks: desktopicon

[Tasks]
Name: desktopicon; Description: "Create a &desktop icon"; GroupDescription: "Additional icons:"

[InstallDelete]
Type: filesandordirs; Name: "{commondesktop}\{#MyAppName}"
Type: filesandordirs; Name: "{commonprograms}\{#MyAppName}"
Type: filesandordirs; Name: "{group}"

#include "common-uninstall.iss"
