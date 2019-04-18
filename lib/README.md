libvideostitch is the core library of VideoStitch.

The library can be compiled with a CUDA or an OpenCL backend. The GPU backend is tied to the library, and in both cases a self-contained library is created. Thus `libvideostitch_cuda` and `libvideostitch_opencl` share the same API and ABI.
