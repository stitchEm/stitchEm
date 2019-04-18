#define io_dir "..\bin\x64\release\core_plugins"
#define plugin_dir "core_plugins"

Source: "{#io_dir}\jpg.dll"; DestDir: "{app}\{#plugin_dir}"; Flags: ignoreversion
Source: "{#io_dir}\png.dll"; DestDir: "{app}\{#plugin_dir}"; Flags: ignoreversion
Source: "{#io_dir}\tiff.dll"; DestDir: "{app}\{#plugin_dir}"; Flags: ignoreversion
