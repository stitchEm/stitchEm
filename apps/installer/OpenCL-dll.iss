#define opencl_io_dir "..\bin\x64\release\core_plugins_opencl"
#define opencl_plugin_dir "core_plugins_opencl"

; VideoStitch internals
Source: "{#bin_dir}\libvideostitch_opencl.dll"; DestDir: "{app}"; Flags: ignoreversion

; av plugin
Source: "{#opencl_io_dir}\av_opencl.dll"; DestDir: "{app}\{#opencl_plugin_dir}"; Flags: ignoreversion