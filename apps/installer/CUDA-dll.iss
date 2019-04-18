#define cuda_io_dir "..\bin\x64\release\core_plugins_cuda"
#define cuda_plugin_dir "core_plugins_cuda"

#define cuda_version GetEnv('CUDA_VERSION')
#define cuda_dir GetEnv('CUDA_INSTALL')
#define cuda_path cuda_dir + "\v" + cuda_version

#define programfiles GetEnv('ProgramW6432')
#define nvsmi_dir programfiles + "\NVIDIA CORPORATION\NVSMI"

; CUDA dependencies
Source: "{#cuda_path}\bin\cudart64_80.dll"; DestDir: "{app}"; Flags: ignoreversion

; NVML dependency
Source: "{#nvsmi_dir}\nvml.dll"; DestDir: "{app}"; Flags: ignoreversion

; VideoStitch internals
Source: "{#bin_dir}\libvideostitch_cuda.dll"; DestDir: "{app}"; Flags: ignoreversion

; av plugin
Source: "{#cuda_io_dir}\av_cuda.dll"; DestDir: "{app}\{#cuda_plugin_dir}"; Flags: ignoreversion
