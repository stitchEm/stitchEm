## OpenCL GPGPU backend

OpenCL programs are written in OpenCL-C, compatible with OpenCL 1.2. Each program is stored in a .cl file, which can contain several kernel definitions and utility functions.

Code shared by OpenCL programs is put in header files (.h) and declared as inline functions. There is currently no linking of OpenCL programs against each other, every OpenCL program stands for itself.

Since OpenCL can target many different platforms, it is usually compiled just-in-time. We compile the .cl files into an intermediate representation (SPIR) to catch errors at compile time.

This is done in the compilation process (see backend/cl/CMakeLists.txt), creating kernel.spir from kernel.cl. The next step is to bake the binary file directly into libvideostitch, as not to drag around the binary kernels next to it. The tool `xxd` is used to create a file containing C code that defines the binary data as a C char array (kernel.xxd).

This xxd file can be included in the code (in the same file that calls the kernel to keep the dependencies) thus baking the binary representation directly into libvideostitch. On runtime, the binary will be read and compiled from the SPIR state directly into code the current GPU can run.

To add the program foo.cl to the compilation process, it has to be registered in this folder's CMakeLists.txt, and in the corresponding foo.cpp (that's launching the foo kernels), the following code has to be added, to make the OpenCL SPIR binary available:

```C++
namespace {
#include "foo.xxd"
}
INDIRECT_REGISTER_OPENCL_PROGRAM(foo, true);
```
