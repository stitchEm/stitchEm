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
Source: "{#vah_dir}\aja_64.dll"; DestDir: "{app}\{#vah_plugin_dir}"; Flags: ignoreversion
Source: "{#vah_dir}\rtmp.dll"; DestDir: "{app}\{#vah_plugin_dir}"; Flags: ignoreversion
Source: "{#vah_dir}\portaudio.dll"; DestDir: "{app}\{#vah_plugin_dir}"; Flags: ignoreversion
Source: "{#vah_dir}\ximea_64.dll"; DestDir: "{app}\{#vah_plugin_dir}"; Flags: ignoreversion

; Magewell dependencies
Source: "{#bin_dir}\LibXIProperty.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#bin_dir}\LibXIStream2.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#bin_dir}\LibMWCapture.dll"; DestDir: "{app}"; Flags: ignoreversion

;AJA CORVID88 NTV2 dependencies
Source: "{#bin_dir}\ajastuffdll_64.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#bin_dir}\classesDLL_64.dll"; DestDir: "{app}"; Flags: ignoreversion

; Point Gray & Ladybug dependencies
;Source: "installer\vcredist_x64_2010.exe"; DestDir: "{app}"; Flags: deleteafterinstall ignoreversion
;Source: "{#bin_dir}\ActiveFlyCap_v100.dll"; DestDir: "{app}"; Flags: ignoreversion
;Source: "{#bin_dir}\FlyCapture2_v100.dll"; DestDir: "{app}"; Flags: ignoreversion
;Source: "{#bin_dir}\libiomp5md.dll"; DestDir: "{app}"; Flags: ignoreversion
;Source: "{#bin_dir}\ladybug.dll"; DestDir: "{app}"; Flags: ignoreversion

; Ximea dependencies
Source: "{#bin_dir}\vcredist_x64_2012.exe"; DestDir: "{app}"; Flags: deleteafterinstall ignoreversion
;Source: "{#bin_dir}\xi4Api_x64.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#bin_dir}\xiapi64.dll"; DestDir: "{app}"; Flags: ignoreversion

; rtmp dependencies
Source: "{#bin_dir}\librtmp.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#bin_dir}\librtmp-1.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#bin_dir}\zlib1.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#bin_dir}\libx264-148.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#bin_dir}\libmp3lame-0.dll"; DestDir: "{app}"; Flags: ignoreversion

; rtmp and Youtube output dependencies
Source: "{#bin_dir}\ssleay32.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#bin_dir}\libeay32.dll"; DestDir: "{app}"; Flags: ignoreversion

; Youtube output dependencies
Source: "{#bin_dir}\roots.pem"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#bin_dir}\libglog.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#bin_dir}\libcurl.dll"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#ExeName}"; IconFilename: "{app}\{#ExeName}"
;Name: "{group}\Uninstall {#MyAppName}"; Filename: "{uninstallexe}" ; create a shortcut in the startmenu
Name: "{commondesktop}\{#MyAppName}"; Filename: "{app}\{#ExeName}"; Tasks: desktopicon

[Run]
;Filename: "{app}\vcredist_x64_2010.exe"; Parameters: "/q /norestart"; StatusMsg: "Installing Microsoft Visual C++ 2010 Redistributable Package (x64)"
Filename: "{app}\vcredist_x64_2012.exe"; Parameters: "/q /norestart"; StatusMsg: "Installing Microsoft Visual C++ 2012 Redistributable Package (x64)"
Filename: "{app}\vcredist_x64_2013.exe"; Parameters: "/q /norestart"; StatusMsg: "Installing Microsoft Visual C++ 2013 Redistributable Package (x64)"
Filename: {app}\{#ExeName}; Description: "Launch {#MyAppName}"; Flags: nowait postinstall skipifsilent runascurrentuser

[Tasks]
Name: desktopicon; Description: "Create a &desktop icon"; GroupDescription: "Additional icons:"

[InstallDelete]
Type: filesandordirs; Name: "{commondesktop}\{#MyAppName}"
Type: filesandordirs; Name: "{commonprograms}\{#MyAppName}"
Type: filesandordirs; Name: "{group}"

#include "common-uninstall.iss"
