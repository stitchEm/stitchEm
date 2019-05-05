The stitchEm project is built with CMake. The user interface is developed with Qt. The stitching is done on the GPU and requires either CUDA or OpenCL.

Several 3rd party libraries are required to build the project. On Linux and macOS these can be installed with a package manager. Windows currently requires a manual installation and building of the required libraries.

Using ccache to speed up recompilation is recommended, but not required, on Linux and macOS. Using ninja as the build system for quick builds is recommended, but not recquired. A list of IDEs including Qt Creator, Xcode and Visual Studio are supported through their CMake support or CMake generators.

## Building with Windows:

```
# Build tools

- Install Visual Studio 2013
- run cmd.exe as admin
- install chocolatey (https://chocolatey.org/install)

- choco install cmake -y
- choco install ninja -y

- choco install winflexbison3 -y

# TODO doesn't work - mscv10/12 only - msvc version mismatch?
- choco install glfw3 -y


# Install conan package manager?
- choco install conan -y

# problem: installs OpenCV 4 vc14_vc15 - we need vc12?
- choco install opencv -y




- mkdir build
- cd build

# Visual Studio 2013
- "C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\vcvarsall.bat" x64

# Visual Studio 2017
- "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build\vcvarsall.bat" x64


- cmake -G Ninja -DDISABLE_EXR=ON -DWIN_CHOCO=ON -DBUILD_IO_PLUGINS=OFF -DBUILD_APPS=OFF ../stitchEm
```

##  Building with macOS:

```
# Build tools
sudo port install CMake bison doxygen yasm ninja ccache

# Libraries
sudo port install opencv glew gsed jpeg libpng openal \
          tiff faac faad2 ceres-solver glfw ffmpeg glm OpenEXR
```

```
cmake -DCREATE_BOX_PACKAGE=OFF \
      -DGPU_BACKEND_CUDA=ON -DGPU_BACKEND_OPENCL=ON \
      -DQt5_DIR=~/Qt/5.9.6/clang_64/lib/cmake/Qt5 \
      -G Ninja \
      stitchEm
```


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
  portaudio19-dev \
  qt5-default \
  qtmultimedia5-dev \
  qttools5-dev \
  swig

# Set up gcc-6 and g++-6 as your compiler
sudo apt-get install gcc-6 g++-6


# As system-wide configuration
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-6 20
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-6 20

# Or for the local build
export CC=gcc-6
export CXX=g++-6

# Install CUDA: https://developer.nvidia.com/cuda-90-download-archive

cmake -DGPU_BACKEND_CUDA=ON -DGPU_BACKEND_OPENCL=ON \
      -G Ninja \
      stitchEm
```

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
