The stitchEm project is built with CMake. The user interface is developed with Qt. The stitching is done on the GPU and requires either CUDA or OpenCL.

Several 3rd party libraries are required to build the project. On Linux and macOS these can be installed with a package manager. Windows currently requires a manual installation and building of the required libraries.

Using ccache to speed up recompilation is recommended, but not required, on Linux and macOS. Using ninja as the build system for quick builds is recommended, but not recquired. A list of IDEs including Qt Creator, Xcode and Visual Studio are supported through their CMake support or CMake generators.

##  Building with macOS:

To install the dependencies on macOS, use either the MacPorts or Homebrew package manager.

### Using MacPorts

```
# Install build tools
sudo port install CMake bison doxygen yasm ninja ccache

# Install library dependencies
sudo port install opencv glew gsed jpeg libpng \
          tiff faac faad2 ceres-solver glfw glm OpenEXR \
          ffmpeg +gpl2 +librtmp +nonfree
```

```
# Configure build
cmake -DCREATE_BOX_PACKAGE=OFF \
      -DGPU_BACKEND_CUDA=ON -DGPU_BACKEND_OPENCL=ON \
      -DQt5_DIR=~/Qt/5.9.6/clang_64/lib/cmake/Qt5 \
      -DMACPORTS=ON \
      -G Ninja \
      stitchEm

# Build!
ninja
```

### Using Homebrew

See the `run` instructions in the MacOpenCL builder in [../.github/workflows/build.yml](build.yml) for a full OpenCL-build with Homebrew dependencies.


##  Building with Linux:

```
# Installing the dependencies on Ubuntu:
sudo apt install
  bison \
  clang-tools \
  cmake \
  doxygen \
  flex \
  libceres-dev \
  libeigen3-dev \
  libfaac-dev \
  libfaad-dev \
  libglew-dev \
  libglfw3-dev \
  libglm-dev \
  libmp3lame-dev \
  libopencv-dev \
  libopenexr-dev \
  libpng-dev \
  libportaudio-ocaml-dev \
  librtmp-dev \
  libturbojpeg0-dev \
  libx264-dev \
  ninja-build \
  ocl-icd-opencl-dev \
  opencl-headers \
  portaudio19-dev \
  qt5-default \
  qtmultimedia5-dev \
  qttools5-dev \
  swig \
  wget \
  xxd \


# Install CUDA: https://developer.nvidia.com/cuda-downloads

# Configure build
cmake -DGPU_BACKEND_CUDA=ON -DGPU_BACKEND_OPENCL=OFF -DRTMP_NVENC=OFF \
      -G Ninja \
      stitchEm

ninja
```

## Building on windows

You need visual studio 2017, QT >= 5.9 and CUDA 10
Install [vcpkg](https://github.com/microsoft/vcpkg)
Then installs all of this:
```
./vcpkg install --triplet x64 ceres eigen3 ffmpeg[avresample,core,gpl,x264,opencl] gflags glfw3 glog libjpeg-turbo liblzma libpng librtmp libwebp mp3lame opencl opencv3 openexr opengl openssl openvr portaudio protobuf tiff x264 zlib glm
./vcpkg install glew:x64-windows-static
```
* install manually [bison/flex](https://sourceforge.net/projects/winflexbison/files) and put the executables in the PATH
* install manually [Intel SDK 2017](https://software.intel.com/en-us/media-sdk)
* install manually [Magewell SDK](http://www.magewell.com/files/sdk/Magewell_Capture_SDK_3.3.1.1004.zip)
* install manually [Oculus SDK (1.4.0)](https://developer.oculus.com/downloads/package/oculus-sdk-for-windows/1.4.0)
* install manually [Decklink SDK](https://www.blackmagicdesign.com/developer/product/capture-and-playback) v10.9.12
* install manually [XIMEA SDK](https://www.ximea.com/support/documents/4)

clone the repo and create a directory next to it, and configure with cmake:
```
git clone https://github.com/stitchEm/stitchEm.git
mkdir build
cd build
cmake -G "Visual Studio 15 2017 Win64" \
    -DCMAKE_TOOLCHAIN_FILE=PATH_TO_YOUR_vcpkg_REPOSITORY\scripts\buildsystems\vcpkg.cmake \
    -DQt5_DIR=PATH_TO_QT_5\msvc2017\lib\cmake\Qt5 \
    -DGPU_BACKEND_CUDA=ON \
    -DGPU_BACKEND_OPENCL=OFF \
    -DCUDA_LOCAL_ARCH_ONLY=ON \
    -DVCPKG_ROOT="PATH_TO_YOUR_vcpkg_REPOSITORY" \
    -DVCPKG_TARGET_TRIPLET="x64-windows" \
    -DMAGEWELL_PATH="PATH_TO_MAGEWELL_SDK" \
    -DXIMEA_PATH="PATH_TO_XIMEA_API" \
    -DDECKLINK_PATH="PATH_TO_DECK_LINK_OVR" \
    -DINTEL_MEDIA_SDK_PATH="PATH_TO_INTEL_SDK" \
    ..\stitchEm\
```

Then open the generated project with visual studio and build it.

## CMake flags

### Global options

| Option                         | Default               | Usage                                                               |
|:-------------------------------|:----------------------|:--------------------------------------------------------------------|
| BUILD_STATIC_LIB               | OFF<br>ON for Windows | Static library for unit tests                                       |
| BUILD_IO_PLUGINS               | ON                    | I/O plugins                                                         |
| BUILD_APPS                     | ON                    | VideoStitch Studio, Vahana VR                                       |
| GENERATE_BINDINGS              | OFF                   | libvideostitch Python bindings                                      |
| CMAKE_CXX_INCLUDE_WHAT_YOU_USE | undefined             | Linux/macOS. Specifies the path to the "include what you use" tool. |

### libvideostitch options

| Option              | Default | Usage                                                                              |
|:--------------------|:--------|:-----------------------------------------------------------------------------------|
| GPU_BACKEND_CUDA    | ON      | Build CUDA backend                                                                 |
| GPU_BACKEND_OPENCL  | OFF     | Build OpenCL backend                                                               |
| GPU_BACKEND_DEFAULT | CUDA    | If both backend builds are enabled, set the default one (for tests and static lib) |
| AUTOCROP_CMD        | ON      |                                                                                    |

### OpenCL backend

| Option              | Default | Usage                                                           |
|:--------------------|:--------|:----------------------------------------------------------------|
| DISABLE_OPENCL_SPIR | OFF     | Turn off SPIR compilation, put OpenCL source code in lib binary |

### CUDA backend

| Option               | Default   | Usage                                                                                                                |
|:---------------------|:----------|:---------------------------------------------------------------------------------------------------------------------|
| FASTMATH             | ON        | CUDA compilation flag                                                                                                |
| LINEINFO             | OFF       | CUDA compilation flag                                                                                                |
| CUDA_LOCAL_ARCH_ONLY | OFF       | Recommended to turn ON for local dev builds to reduce build time                                                     |
| CUDA_TARGET_ARCH     | undefined | Build for specific GPU arch that is different from local GPU. A list of target capabilies can be set (e.g. "52,50"). |


### UI (APPS)
| Option               | Default   | Usage                                                                                                                |
|:---------------------|:----------|:---------------------------------------------------------------------------------------------------------------------|
| Qt5_DIR              | undefined | Qt5 install path                                                                                                |
### macOS
| Option               | Default   | Usage                                                                                                                |
|:---------------------|:----------|:---------------------------------------------------------------------------------------------------------------------|
| MACPORTS       | ON | ON if you want to use MacPorts, OFF to use Homebrew |
