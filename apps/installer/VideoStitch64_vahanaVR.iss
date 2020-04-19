#define MyAppName "Vahana VR"
#define VersionName "VahanaVR"
#define ExeName VersionName + ".exe"
#define bin_dir "..\bin\x64\release"
#define vah_dir "..\bin\x64\release\vahana_plugins"
#define vah_plugin_dir "vahana_plugins"

[Setup]
AppID=52CB03D5-4269-487C-A4E6-CA059D4C1BF0
SetupIconFile="src\resources\icons\vs\vahana.ico"
AppMutex==VideoStitch-52CB03D5-4269-487C-A4E6-CA059D4C1BF0
AppVersion=__APPVERSION__
OutputBaseFileName=VideoStitch-VahanaVR-Setup
VersionInfoVersion=__INFOVERSION__
WizardImageAlphaFormat = defined
WizardImageFile="src\resources\installer\orah_side.bmp"
WizardSmallImageFile = "src\resources\installer\orah_small.bmp"
#include "common-setup.iss"

[REGISTRY]
Root: HKCR; Subkey: ".vah"; ValueType: string; ValueName: ""; ValueData: {#MyAppName}; Flags: uninsdeletevalue
Root: HKCR; Subkey: "{#MyAppName}"; ValueType: string; ValueName: ""; ValueData: {#MyAppName}; Flags: uninsdeletekey
Root: HKCR; Subkey: "{#MyAppName}\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\{#ExeName},0"
Root: HKCR; Subkey: "{#MyAppName}\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#ExeName}"" ""%1"""

[Dirs]
Name: "{app}"; Flags: uninsalwaysuninstall
Name: "{app}\platforms"; Flags: uninsalwaysuninstall
Name: "{app}\imageformats"; Flags: uninsalwaysuninstall
Name: "{app}\vahana_plugins"; Flags: uninsalwaysuninstall
Name: "{%HOMEPATH}\{#MyAppName}\Projects"; Flags: uninsneveruninstall
Name: "{%HOMEPATH}\{#MyAppName}\Recordings"; Flags: uninsneveruninstall
Name: "{%HOMEPATH}\{#MyAppName}\Snapshots"; Flags: uninsneveruninstall

[Files]
#include "common-dll.iss"
#include "core-plugins-dll.iss"
#include "__GPU__-dll.iss"
__OTHER_GPU__

Source: "installer\examples\*.vah"; DestDir: "{%HOMEPATH}\{#MyAppName}\Projects"; Flags: confirmoverwrite uninsneveruninstall

Source: "{#vah_dir}\decklink.dll"; DestDir: "{app}\{#vah_plugin_dir}"; Flags: ignoreversion
Source: "{#vah_dir}\magewell.dll"; DestDir: "{app}\{#vah_plugin_dir}"; Flags: ignoreversion
Source: "{#vah_dir}\magewellpro.dll"; DestDir: "{app}\{#vah_plugin_dir}"; Flags: ignoreversion
Source: "{#vah_dir}\rtmp.dll"; DestDir: "{app}\{#vah_plugin_dir}"; Flags: ignoreversion
Source: "{#vah_dir}\portaudio.dll"; DestDir: "{app}\{#vah_plugin_dir}"; Flags: ignoreversion
Source: "{#vah_dir}\ximea_64.dll"; DestDir: "{app}\{#vah_plugin_dir}"; Flags: ignoreversion

; Magewell dependencies
Source: "C:\Windows\System32\LibXIProperty.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "C:\Windows\System32\LibXIStream2.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "C:\Windows\System32\LibMWCapture.dll"; DestDir: "{app}"; Flags: ignoreversion

; Ximea dependencies
Source: "D:\dev\XIMEA\vcredist_2013_x64.exe"; DestDir: "{app}"; Flags: deleteafterinstall ignoreversion
Source: "D:\dev\XIMEA\API\x64\xiapi64.dll"; DestDir: "{app}"; Flags: ignoreversion


[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#ExeName}"; IconFilename: "{app}\{#ExeName}"
Name: "{commondesktop}\{#MyAppName}"; Filename: "{app}\{#ExeName}"; Tasks: desktopicon

[Run]
Filename: {app}\{#ExeName}; Description: "Launch {#MyAppName}"; Flags: nowait postinstall skipifsilent runascurrentuser

[Tasks]
Name: desktopicon; Description: "Create a &desktop icon"; GroupDescription: "Additional icons:"

[InstallDelete]
Type: filesandordirs; Name: "{commondesktop}\{#MyAppName}"
Type: filesandordirs; Name: "{commonprograms}\{#MyAppName}"
Type: filesandordirs; Name: "{group}"

#include "common-uninstall.iss"
