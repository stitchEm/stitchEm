// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "cmdUtils.hpp"

#include <libvideostitch/logging.hpp>
#include <libvideostitch/gpu_device.hpp>
#include <libvideostitch/inputController.hpp>
#include <libvideostitch/panoDef.hpp>
#include <libvideostitch/parse.hpp>
#include <libvideostitch/status.hpp>

#include "libgpudiscovery/backendLibHelper.hpp"

// System-dependant filesystem stuff.
#ifdef _MSC_VER
#include <direct.h>
#define chdir _chdir
#define snprintf _snprintf
#define getcwd _getcwd
#include <io.h>
#include <sys/types.h>
#include <sys/stat.h>
#else
#include <unistd.h>
#endif

#ifdef _MSC_VER
#include <libgpudiscovery/delayLoad.hpp>
SET_DELAY_LOAD_HOOK
FARPROC WINAPI delayFailureHook(unsigned dliNotify, PDelayLoadInfo pdli) {
  switch (dliNotify) {
    case dliFailLoadLib:
      throw std::runtime_error(std::string(pdli->szDll) + " could not be loaded");
    case dliFailGetProc:
      throw std::runtime_error("Could not find procedure in module " + std::string(pdli->szDll));
    default:
      assert(false);
      throw std::runtime_error("Unknown error code");
  };
}
#endif  // _MSC_VER

namespace VideoStitch {
namespace Cmd {

namespace {
/**
 * Give a pointer on the file component of a full input filename.
 * @param input The input full filename.
 * @return the filename, belonging to the same char* as @input.
 */
const char* extractFilename(const char* input) {
  const char* file = input;
#ifdef _MSC_VER
  // separator is '\' or '/'
  for (const char* p = input; *p != 0; ++p) {
    if (*p == '/' || *p == '\\') {
      file = p + 1;
    }
  }
#else
  // separator is '/'
  for (const char* p = input; *p != 0; ++p) {
    if (*p == '/') {
      file = p + 1;
    }
  }
#endif
  return file;
}

}  // namespace

bool loadGPUBackend(const int deviceId, int& returnCode) {
  // select backend
  VideoStitch::Discovery::Framework selectedFramework = VideoStitch::Discovery::Framework::Unknown;
  Discovery::DeviceProperties prop;
  prop.supportedFramework = VideoStitch::Discovery::Framework::Unknown;

  // if a device is selected, select its backend
  if (deviceId != -1) {
    if (Discovery::getDeviceProperties(deviceId, prop) &&
        BackendLibHelper::isBackendAvailable(prop.supportedFramework)) {
      selectedFramework = prop.supportedFramework;
      std::cout << "Selected device " << deviceId << std::endl;
    }
  }

  // if no device is selected, try select best available backend
  if (selectedFramework == VideoStitch::Discovery::Framework::Unknown) {
    std::cout << "No device selected, selecting default device" << std::endl;
    selectedFramework = BackendLibHelper::getBestFrameworkAndBackend();
  }

  bool needToRestart = false;
  if (BackendLibHelper::selectBackend(selectedFramework, &needToRestart)) {
    std::cout << "[videostitch-cmd] " << VideoStitch::Discovery::getFrameworkName(selectedFramework)
              << " backend selected" << std::endl;
  } else {
    std::cerr << "No CUDA nor OpenCL capable GPU detected on your system. "
              << "A Nvidia card with CUDA capability and CUDA drivers or "
              << "a graphics card with OpenCL capability and drivers installed are mandatory to run the software. "
              << std::endl;
    returnCode = 1;
    return false;
  }

  if (needToRestart) {
    std::cout << "Closing application to finalize setup " << std::endl;
#ifdef __APPLE__
    BackendLibHelper::forceUpdateSymlink();
#endif
    returnCode = 0;
    return false;
  }

#ifdef DELAY_LOAD_ENABLED
  // failure hook, to prevent crash on delay load failures
  PfnDliHook oldFailureHook = __pfnDliFailureHook2;
  __pfnDliFailureHook2 = &delayFailureHook;
  try {
    // call any lib function
    Status::OK();
  } catch (std::exception& e) {
    std::cerr << "Error using backend library: " << e.what() << std::endl;
  }
  __pfnDliFailureHook2 = oldFailureHook;
#endif
  return true;
}

const char* changeWorkingPathToPtvFolder(char* ptvPath) {
  // Change directory to the project directory, so that all paths are relative.
  const char* ptvFile = extractFilename(ptvPath);
  if (ptvFile != ptvPath) {
    // if ptvPath is "/"
    if (ptvFile - ptvPath == 1) {
      if (chdir("/")) {
        Logger::get(Logger::Error) << "Incorrect input file: " << ptvFile << std::endl;
        return nullptr;
      }
    } else {
      ptvPath[ptvFile - ptvPath - 1] = 0;  // trim the path
      if (chdir(ptvPath)) {
        Logger::get(Logger::Error) << "Incorrect input file: " << ptvFile << std::endl;
        return nullptr;
      }
    }
  }
  return ptvFile;
}

bool parseInputPath(int argc, char** argv, int index, char** ptvPath) {
  if (index >= argc - 1 || (argv[index + 1][0] == '-' && argv[index + 1][1] != '\0')) {
    Logger::get(Logger::Error) << "The -i option takes a parameter." << std::endl;
    return false;
  }
  if (*ptvPath) {
    Logger::get(Logger::Error) << "Several input files: \"" << *ptvPath << "\" and \"" << argv[index + 1] << "\""
                               << std::endl;
    return false;
  }
  *ptvPath = argv[index + 1];
  return true;
}

bool parseOutputPath(int argc, char** argv, int index, char** ptvPath) {
  if (index >= argc - 1 || (argv[index + 1][0] == '-' && argv[index + 1][1] != '\0')) {
    Logger::get(Logger::Error) << "The -o option takes a parameter." << std::endl;
    return false;
  }
  if (*ptvPath) {
    Logger::get(Logger::Error) << "Several output files: \"" << *ptvPath << "\" and \"" << argv[index + 1] << "\""
                               << std::endl;
    return false;
  }
  *ptvPath = argv[index + 1];
  return true;
}

bool parseFirstFrame(int argc, char** argv, int index, int* firstFrame) {
  if (index >= argc - 1 || (argv[index + 1][0] == '-' && argv[index + 1][1] != '\0')) {
    Logger::get(Logger::Error) << "The -f option takes a parameter." << std::endl;
    return false;
  }
  *firstFrame = atoi(argv[index + 1]);
  return true;
}

bool parseLastFrame(int argc, char** argv, int index, int* lastFrame) {
  if (index >= argc - 1 || (argv[index + 1][0] == '-' && argv[index + 1][1] != '\0')) {
    Logger::get(Logger::Error) << "The -l option takes a parameter." << std::endl;
    return false;
  }
  *lastFrame = atoi(argv[index + 1]);
  return true;
}

Status checkDevice(int vsDeviceID) {
  FAIL_RETURN(GPU::setDefaultBackendDeviceVS(vsDeviceID));
  return GPU::checkDefaultBackendDeviceInitialization();
}

bool parseDeviceId(int argc, char** argv, int& deviceId, int& returnCode) {
  deviceId = -1;
  for (int i = 1; i < argc; ++i) {
    if (argv[i][0] != '\0' && argv[i][1] != '\0' && argv[i][0] == '-') {
      switch (argv[i][1]) {
        case 'd':
          if (i >= argc - 1 || argv[i + 1][0] == '-') {
            std::cout << "-d option used without parameter, list compatible GPU devices:" << std::endl;
            int numDevices = Discovery::getNumberOfDevices();
            if (numDevices == 0) {
              std::cerr << "Could not find any compatible GPU device on this computer." << std::endl;
              returnCode = 1;
              return false;
            }
            for (int i = 0; i < numDevices; ++i) {
              Discovery::DeviceProperties prop;
              if (Discovery::getDeviceProperties(i, prop)) {
                std::cout << "Device " << i << ": " << prop << std::endl;
              } else {
                std::cerr << "Could not query device properties for device " << i << std::endl;
              }
            }
            returnCode = 0;
            return false;
          }
          // else
          std::stringstream ss(argv[++i]);
          if (!(ss >> deviceId)) {
            std::cerr << "Malformed device id: '" << argv[i] << "'" << std::endl;
            returnCode = 1;
            return false;
          }
          break;
      }
    }
  }
  return true;
}

/**
 * Detects GPU devices and returns true on success.
 * No check is done for GPU usability
 */
bool selectGPUDevice(int argc, char** argv, int& deviceId, int& returnCode) {
  if (!parseDeviceId(argc, argv, deviceId, returnCode)) {
    return false;
  }

  int defaultGpu = 0;
  int devCount = Discovery::getNumberOfDevices();

  if (devCount == 0) {
    std::cerr << "Error: No GPU found!" << std::endl;
    returnCode = 1;
    return false;
  } else if (devCount == 1) {
    if (deviceId != -1) {
      std::cout << "Warning: Only one GPU found, ignoring -d." << std::endl;
    }
    deviceId = defaultGpu;
  } else {
    if (deviceId == -1) {
      std::cerr << "Error: More than 1 device and no '-d' option found. Falling back to the default device ('-d 0')."
                << std::endl;
      deviceId = defaultGpu;
    } else {
      if (deviceId >= devCount) {
        std::cerr << "Error: No such device " << deviceId << ". Only " << devCount << " GPUs found." << std::endl;
        returnCode = 1;
        return false;
      }
    }
  }
  return true;
}

/**
 * @brief Prints properties of a given GPU device.
 * @param os Output stream to print to.
 * @param vsGPUDeviceID GPU device (libvideostitch internal ID)
 */
void printDeviceProperties(ThreadSafeOstream& os, int vsGPUDeviceID) {
  Discovery::DeviceProperties prop;
  if (VideoStitch::Discovery::getDeviceProperties(vsGPUDeviceID, prop)) {
    os << prop << std::endl;
  } else {
    os << "Error when trying to access device properties of GPU device #" << vsGPUDeviceID << std::endl;
  }
}

/**
 * Checks the usable GPU devices and returns true on success.
 */
bool checkGPUDevice(Core::PanoDeviceDefinition& dev) {
  auto& errLog = Logger::get(Logger::Error);

  if (!checkDevice(dev.device).ok()) {
    errLog << "Cannot use device " << dev.device << std::endl;
    return false;
  }
  printDeviceProperties(Logger::get(Logger::Verbose), dev.device);
  return true;
}

std::unique_ptr<Core::PanoDefinition> parsePanoDef(const Ptv::Value& ptvRoot, const char* ptvFile) {
  if (!ptvRoot.has("pano")) {
    Logger::get(Logger::Error) << "Error: No 'pano' entry in file: " << ptvFile << std::endl;
    return nullptr;
  }

  // Create a runtime panorama from the parsed project.
  Core::PanoDefinition* panoDef = Core::PanoDefinition::create(*ptvRoot.has("pano"));
  if (!panoDef) {
    Logger::get(Logger::Error) << "Error: Invalid panorama definition in " << ptvFile << std::endl;
    return nullptr;
  }

  return std::unique_ptr<Core::PanoDefinition>{panoDef};
}

bool parsePtvFile(Ptv::Parser& parser, const char* ptvFile) {
  // Load the project and parse it.
  if (!parser.parse(ptvFile)) {
    Logger::get(Logger::Error) << "Error: Cannot parse PTV file: " << ptvFile << std::endl;
    Logger::get(Logger::Error) << parser.getErrorMessage() << std::endl;
    return false;
  }

  return true;
}

std::unique_ptr<Core::PanoDefinition> parsePanoDef(Ptv::Parser& parser, const char* ptvFile) {
  if (parsePtvFile(parser, ptvFile)) {
    return parsePanoDef(parser.getRoot(), ptvFile);
  }
  return nullptr;
}

bool normalizeFrameBoundaries(const Core::InputController& controller, const frameid_t firstFrame,
                              frameid_t& lastFrame) {
  lastFrame =
      lastFrame < 0 ? controller.getLastStitchableFrame() : std::min(lastFrame, controller.getLastStitchableFrame());
  if (lastFrame == NO_LAST_FRAME) {
    Logger::get(Logger::Error) << "Last frame auto detection was enabled, but all the readers are unbounded (Are you "
                                  "using only procedural readers ?)."
                               << std::endl;
    lastFrame = firstFrame;
  }

  if (lastFrame < firstFrame) {
    Logger::get(Logger::Error) << "Nothing to stitch: last_frame = " << lastFrame << " < first_frame = " << firstFrame
                               << "." << std::endl;
    return false;
  }

  Logger::get(Logger::Info) << "Will stitch " << lastFrame - firstFrame + 1 << " images." << std::endl;
  return true;
}

}  // namespace Cmd
}  // namespace VideoStitch
