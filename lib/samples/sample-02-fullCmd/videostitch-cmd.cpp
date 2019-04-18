// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm
// Delay load helper
#include "libgpudiscovery/backendLibHelper.hpp"
#include "libgpudiscovery/fileHelper.hpp"
// VideoStitch SDK
#include <libvideostitch/allocator.hpp>
#include <libvideostitch/algorithm.hpp>
#include <libvideostitch/audio.hpp>
#include <libvideostitch/audioPipeDef.hpp>
#include <libvideostitch/controller.hpp>
#include <libvideostitch/context.hpp>
#include <libvideostitch/imageMergerFactory.hpp>
#include <libvideostitch/imageWarperFactory.hpp>
#include <libvideostitch/imageFlowFactory.hpp>
#include <libvideostitch/inputFactory.hpp>
#include <libvideostitch/logging.hpp>
#include <libvideostitch/gpu_device.hpp>
#include <libvideostitch/output.hpp>
#include <libvideostitch/overlay.hpp>
#include <libvideostitch/panoDef.hpp>
#include <libvideostitch/parse.hpp>
#include <libvideostitch/profile.hpp>
#include <libvideostitch/ptv.hpp>
#include <libvideostitch/stereoRigDef.hpp>
#include <libvideostitch/stitchOutput.hpp>
#include "version.hpp"

#include "../common/cmdUtils.hpp"

#ifndef GLFWLIB_UNSUPPORTED
#ifdef __linux__
#include "GLFW/glfw3.h"
#else
#include "glfw/glfw3.h"
#endif
#else
typedef void* GLFWwindow;
#define glfwMakeContextCurrent(a) \
  {}
#endif

#include <atomic>
#include <csignal>
#include <future>
#include <iomanip>
#include <memory>

// System-dependant filesystem stuff.
#ifdef _MSC_VER
#include <direct.h>
#define chdir _chdir
#define snprintf _snprintf
#define getcwd _getcwd
#include <io.h>
#include <sys/types.h>
#include <sys/stat.h>
const char dirSep = '\\';
#else
#include <unistd.h>
const char dirSep = '/';
#endif

enum option { apply_algos, check_gpu_compatibility, invalid_option };

using namespace VideoStitch;

namespace {

/**
 * @brief Prints the executable usage.
 * @param execName Name of the executable (argv[0]).
 * @param os Output stream: e.g; std::cerr to output in the standard error stream.
 */
void printUsage(const char* execName, ThreadSafeOstream& os) {
  os << "Usage: " << execName << " -i <template.ptv> [options]" << std::endl;
  os << "<template.ptv> can be '-' to read from standard input." << std::endl;
  os << "[options] are:" << std::endl;
  os << "  -d <int>: Used with no argument, -d lists the compatible GPU devices found on this system and shadows all "
        "other options. Otherwise use the specified device id for stitching."
     << std::endl;
  os << "  -f <int>: First frame of the project. Use the first stitchable frame if absent." << std::endl;
  os << "  -l <int>: Last frame of the project. Use the last stitchable frame if absent." << std::endl;
  os << "  -p <directory>: scans directory and loads found plug-ins."
     << " This option can be used several times." << std::endl;
  os << "  -w : Enable OpenGL window for display" << std::endl;
  os << "  -v <q|0|1|2|3|4> : Log level: quiet, error, warning, info (default), verbose, debug." << std::endl;
  os << "  --apply_algos [filename] : applies a list of algorithms. See the documentation for the format of this file."
     << std::endl;
  os << "  --check_gpu_compatibility : check compatibility of the device given to -d option" << std::endl;
  os << std::endl;
  os << "Use '" << execName << " --version' to display version information." << std::endl;
}

/**
 * Extract the path, including the final separator, from a full file name (path/basename.extension).
 * @param infn the full input filename.
 * @param outfn the extract path/basename.
 *              Set to "" if a @infn is a filename with no path indication (i.e. basename.extension).
 */
void extractPath(const std::string& infn, std::string& outfn) {
  for (int i = (int)infn.size() - 1; i > 0; --i) {
    if (infn[i] == dirSep) {
      outfn = infn.substr(0, i + 1);
      return;
    }
  }
  outfn = "";
}

/**
 * Returns the size of a full input filename.
 * @param filepath The input full filename.
 * @return the size of @filepath when it exists, -1 otherwise.
 */
std::streamoff getFileSize(const std::string& filepath) {
  std::ifstream ifile(filepath, std::ios::binary | std::ios::ate);
  if (!ifile.is_open()) {
    return -1;
  }
  ifile.seekg(0, std::ios_base::end);
  std::streamoff size = ifile.tellg();
  ifile.close();
  return size;
}

/**
 * Checks that the current user has the Read & Write permissions on the given path.
 * Note from http://msdn.microsoft.com/en-us/library/1w06ktdy.aspx
 * "in Windows 2000 and later operating systems, all directories have read and write access."
 * @param path Path on which the permissions should be checked.
 */
bool hasRWPermissions(const std::string& path) {
#ifdef _MSC_VER
  struct _stat buf;
  int result;
  char* filename = "crt_stat.c";
  // Get data associated with "crt_stat.c":
  result = _stat(filename, &buf);
  if (result == 0) {
    if (buf.st_mode & S_IFREG) {
      return _access(path.c_str(), 06) == 0;
    }
  }
  return true;
#else
  return access(path.c_str(), W_OK | R_OK) == 0;
#endif
}

/**
 * Checks the file permissions and returns true if OK.
 */
bool maybeCheckFilePermissions(const Ptv::Value* outputConfig) {
  if (!outputConfig || !outputConfig->has("filename") || !outputConfig->has("type") ||
      !outputConfig->has("type")->asString().compare("rtmp")) {
    return true;
  }
  std::stringstream outputFile;
  outputFile << outputConfig->has("filename")->asString() << "." << outputConfig->has("type")->asString();
  // Check if we have the permission to write on the file if it exists
  if (FileHelper::fileExists(outputFile.str())) {
    if (!hasRWPermissions(outputFile.str())) {
      Logger::get(Logger::Error) << "You don't have the permission to write in the directory " << outputFile.str()
                                 << std::endl;
      return false;
    }
  } else {
    std::string directory;
    extractPath(outputFile.str(), directory);

    if (directory == "") {
      directory = ".";
    }
    // Check if we have the permissions to write in the output directory.
    if (!hasRWPermissions(directory)) {
      Logger::get(Logger::Error) << "You don't have the permission to write in the directory " << directory
                                 << std::endl;
      return false;
    }
  }
  return true;
}

std::atomic<bool> threadsShallDie;

#ifdef _MSC_VER
/**
 * @breif Catches any stop signal and stops the processing.
 * @return True: close the app, False: do not close it.
 */
BOOL WINAPI signalHandler(_In_ DWORD dwCtrlType) {
  switch (dwCtrlType) {
    case CTRL_C_EVENT:
    case CTRL_BREAK_EVENT:
    case CTRL_CLOSE_EVENT:
    case CTRL_LOGOFF_EVENT:
    case CTRL_SHUTDOWN_EVENT:
      Logger::get(Logger::Warning) << "Caught signal(" << dwCtrlType << "), trying to close gracefully..." << std::endl;
      break;
    default:
      return FALSE;
  }

  threadsShallDie = true;
  return TRUE;
}
#else
void sigTermHandler(int sig) {
  if (sig == SIGTERM) {
    Logger::get(Logger::Warning) << "Caught SIGTERM, trying to close gracefully..." << std::endl;
    threadsShallDie = true;
  }
}
#endif

class SimpleProgressReporter : public Util::Algorithm::ProgressReporter {
 public:
  bool notify(const std::string& message, double percent) {
    Logger::get(Logger::Info) << "[" << percent << "%] " << message << std::endl;
    return false;
  }
};

Potential<Core::StitchOutput> makeOutput(Core::Controller* controller, const Core::PanoDefinition& pano,
                                         std::shared_ptr<Output::VideoWriter> writer,
                                         GLFWwindow* offscreenContext = nullptr) {
  std::vector<std::shared_ptr<Core::PanoSurface>> surfs;

  for (int i = 0; i < 2; ++i) {
    if (pano.getProjection() == VideoStitch::Core::PanoProjection::Cubemap ||
        pano.getProjection() == VideoStitch::Core::PanoProjection::EquiangularCubemap) {
      Potential<VideoStitch::Core::CubemapSurface> potSurf = Core::OffscreenAllocator::createCubemapSurface(
          pano.getLength(), "StitchOutput", pano.getProjection() == Core::PanoProjection::EquiangularCubemap);
      FAIL_RETURN(potSurf.status());
      surfs.push_back(std::shared_ptr<Core::PanoSurface>(potSurf.release()));
    } else {
      if (offscreenContext) {
        // Make the offscreen context active for the main thread memory allocations
        glfwMakeContextCurrent(offscreenContext);
        auto panoSurface = Core::OpenGLAllocator::createPanoSurface(pano.getWidth(), pano.getHeight());
        FAIL_RETURN(panoSurface.status());
        surfs.push_back(std::shared_ptr<Core::PanoSurface>(panoSurface.release()));
      } else {
        auto panoSurface =
            Core::OffscreenAllocator::createPanoSurface(pano.getWidth(), pano.getHeight(), "StitchOutput");
        FAIL_RETURN(panoSurface.status());
        surfs.push_back(std::shared_ptr<Core::PanoSurface>(panoSurface.release()));
      }
    }
  }

  // TODO: allow only one thread for this one; this is not thread-safe.
  return controller->createAsyncStitchOutput(surfs, writer);
}

Potential<Core::StereoOutput> makeOutput(Core::StereoController* controller, const Core::PanoDefinition& pano,
                                         std::shared_ptr<Output::StereoWriter> writer) {
  std::vector<std::shared_ptr<Core::PanoSurface>> surfs;
  for (int i = 0; i < 2; ++i) {
    auto panoSurface = Core::OffscreenAllocator::createPanoSurface(pano.getWidth(), pano.getHeight(), "StitchOutput");
    surfs.push_back(std::shared_ptr<Core::PanoSurface>(panoSurface.release()));
  }
  Logger::get(Logger::Verbose) << "Output memory: 2x" << (4 * pano.getWidth() * pano.getHeight()) / (1024 * 1024)
                               << "MB" << std::endl;
  return controller->createAsyncStitchOutput(surfs, writer);
}

template <typename Controller>
Output::Output* makeCallback(Controller* controller, Core::PanoDefinition& pano, Ptv::Value* outputConfig,
                             Audio::SamplingRate outRate, Audio::SamplingDepth outDepth,
                             Audio::ChannelLayout outLayout) {
  size_t width = pano.getWidth(), height = pano.getHeight();
  if (pano.getProjection() == VideoStitch::Core::PanoProjection::Cubemap ||
      pano.getProjection() == VideoStitch::Core::PanoProjection::EquiangularCubemap) {
    width = 3 * pano.getLength();
    height = 2 * pano.getLength();
  } else {
    width = pano.getWidth();
    height = pano.getHeight();
  }

  Potential<Output::Output> pot =
      Output::create(*outputConfig, outputConfig->has("filename")->asString(), (unsigned)width, (unsigned)height,
                     controller->getFrameRate(), outRate, outDepth, outLayout);
  if (!pot.ok()) {
    Logger::get(Logger::Error) << "Output writer creation failed!" << std::endl;
    return nullptr;
  }
  return pot.release();
}

/**
 * The stitcher loop for one device.
 * @param controller Stitching controller.
 * @param stitchOutput The output to use. Must remain alive.
 * @param curFrame The current frame.
 * @param lastFrame The last frame to stitch.
 * @param pleaseDie The thread will die if this becomes nonzero.
 * @param blocker If not NULL, the thread will block on this before stitching a frame.
 */
template <typename Controller>
int stitchLoop(GLFWwindow* offscreenContext, Controller* controller, typename Controller::Output* stitchOutput,
               std::atomic<int>* curFrame, const int lastFrame) {
  // Setup stitcher.
  Util::SimpleProfiler* prof = new Util::SimpleProfiler("Stitcher setup", false, Logger::get(Logger::Info));
  Status status = controller->createStitcher();
  delete prof;
  if (!status.ok()) {
    Logger::get(Logger::Error) << "Could not create stitcher." << std::endl;
    return 1;
  }

  // Activate offscreen graphics context
  if (offscreenContext) {
    glfwMakeContextCurrent(offscreenContext);
  }

  // Run it.
  while (*curFrame <= lastFrame && !threadsShallDie) {
    int frame = ++(*curFrame) - 1;
    Core::ControllerStatus status = controller->stitch(stitchOutput);
    switch (status.getCode()) {
      case Core::ControllerStatusCode::Ok:
        break;
      case Core::ControllerStatusCode::EndOfStream:
        Logger::get(Logger::Info) << "No more input frames available, reached end of stream. Cannot stitch frame "
                                  << frame << ", shutting down." << std::endl;
        return 1;
      case Core::ControllerStatusCode::ErrorWithStatus:
        Logger::get(Logger::Error) << "Failed to stitch frame " << frame << std::endl;
        return 1;
    }
  }
  return 0;
}

typedef std::packaged_task<int(Core::PanoDeviceDefinition&, GLFWwindow*, Core::Controller*, Core::StitchOutput*,
                               std::atomic<int>*, const int)>
    stitch_task;

bool stitchMono(GLFWwindow* offscreenContext, Core::Controller* controller, Core::StitchOutput* stitchOutput,
                int firstFrame, int lastFrame) {
  Util::SimpleProfiler globalComputeProf("Global computation time", false, Logger::get(Logger::Info));
  std::atomic<int> ff(firstFrame);
  return stitchLoop(offscreenContext, controller, stitchOutput, &ff, lastFrame) == 0;
}

bool stitchStereo(GLFWwindow* offscreenContext, Core::StereoController* controller, Core::StereoOutput* stitchOutput,
                  int firstFrame, int lastFrame) {
  Util::SimpleProfiler globalComputeProf("Global computation time", false, Logger::get(Logger::Info));
  std::atomic<int> ff(firstFrame);
  return stitchLoop(offscreenContext, controller, stitchOutput, &ff, lastFrame) == 0;
}

option getOption(const std::string& s) {
  if (s == "--apply_algos") {
    return apply_algos;
  } else if (s == "--check_gpu_compatibility") {
    return check_gpu_compatibility;
  } else {
    return invalid_option;
  }
}
}  // namespace

#ifndef GLFWLIB_UNSUPPORTED
void glfwErrorCallback(int error, const char* description) {
  std::cerr << "GLFW Error " << error << ": " << description << std::endl;
  assert(false);
}

struct LockableContext {
  GLFWwindow* window;
  std::mutex mutex;
  std::condition_variable cv;
};

class Compositor : public VideoStitch::GPU::Overlayer {
 public:
  explicit Compositor(LockableContext& ctx) : Overlayer(), ctx(ctx) {}

  void attachContext() {
    ctx.mutex.lock();
    glfwMakeContextCurrent(ctx.window);
  }
  void detachContext() {
    glfwMakeContextCurrent(nullptr);
    ctx.mutex.unlock();
    ctx.cv.notify_one();
  }

 private:
  LockableContext& ctx;
};

void* renderThreadLoop(LockableContext& ctx, std::shared_ptr<VideoStitch::GPU::Overlayer> overlay) {
  GLFWwindow* win = ctx.window;
  int winWidth, winHeight;

  if (!ctx.window) {
    return 0;
  }

  while (!glfwWindowShouldClose(ctx.window) && overlay) {
    std::unique_lock<std::mutex> lk(ctx.mutex);
    ctx.cv.wait_for(lk, std::chrono::milliseconds(16));
    glfwMakeContextCurrent(win);
    glfwGetWindowSize(ctx.window, &winWidth, &winHeight);
    overlay->renderOverlay(winWidth, winHeight);
    glfwSwapBuffers(ctx.window);
    glfwMakeContextCurrent(nullptr);
  }

  return 0;
}
#endif

int main(int argc, char** argv) {
  // Prints the SDK version
  if (argc == 2 && !strcmp(argv[1], "--version")) {
    std::cout << "videostitch-cmd example, copyright (c) 2018 stitchEm" << std::endl;
    std::cout << "SDK version: " << LIB_VIDEOSTITCH_VERSION << "-" << LIB_VIDEOSTITCH_BRANCH << std::endl;
    return 0;
  }

  int deviceId;
  {
    int returnCode;
    // Get the GPU device
    if (!Cmd::selectGPUDevice(argc, argv, deviceId, returnCode)) {
      return returnCode;
    }

    if (!Cmd::loadGPUBackend(deviceId, returnCode)) {
      return returnCode;
    }
  }

  // DON'T USE LIBVIDEOSTITCH BEFORE THIS LINE
  // Reason: libvideostitch is delay loaded on Windows. Before loading it (thus to use any symbol of it),
  // we should check that we have a usable GPU (this is the job of checkGPUFrameworkAvailable)

  // Set the log level
  Logger::readLevelFromArgv(argc, argv);

  Logger::get(Logger::Info) << "Running on " << VideoStitch::Discovery::getFrameworkName(GPU::getFramework())
                            << " backend" << std::endl;

  auto& errLog = Logger::get(Logger::Error);

  // Check if the output folder/file is writable
  const std::string program = std::string(argv[0]);
  const size_t found = program.find_last_of("/\\");
  std::string executableDirectory = program.substr(0, found);

  char currentDir[1024];
  if (getcwd(currentDir, 1024)) {
    executableDirectory =
        (executableDirectory.empty() || executableDirectory == std::string(argv[0]) || executableDirectory == ".")
            ? std::string(currentDir)
            : executableDirectory;
  }

  // Parse command line
  Core::PanoDeviceDefinition deviceDef;
  Discovery::getBackendDeviceIndex(deviceId, deviceDef.device);

  char* ptvPath = NULL;
  int firstFrame = 0, lastFrame = NO_LAST_FRAME;
  std::string algoFile;
  std::vector<std::string> pluginDirs;
#ifndef GLFWLIB_UNSUPPORTED
  bool enableDebugWindow = false;
#endif
  for (int i = 1; i < argc; ++i) {
    if (argv[i][0] != '\0' && argv[i][1] != '\0' && argv[i][0] == '-') {
      switch (argv[i][1]) {
        case 'i':
          if (!Cmd::parseInputPath(argc, argv, i++, &ptvPath)) {
            printUsage(argv[0], errLog);
            return 1;
          }
          break;
        case 'f':
          if (!Cmd::parseFirstFrame(argc, argv, i++, &firstFrame)) {
            printUsage(argv[0], errLog);
            return 1;
          }
          break;
        case 'l':
          if (!Cmd::parseLastFrame(argc, argv, i++, &lastFrame)) {
            printUsage(argv[0], errLog);
            return 1;
          }
          break;
        case 'd':
          // already parsed with `Cmd::selectGPUDevice`
          ++i;
          break;
        case 'p': /* plugin-directory */
          if (i >= argc - 1) {
            errLog << "The -p option takes a parameter." << std::endl;
            printUsage(argv[0], errLog);
            return 1;
          }
          ++i;
          pluginDirs.push_back(argv[i]);
          break;
#ifndef GLFWLIB_UNSUPPORTED
        case 'w':
          enableDebugWindow = true;
          break;
#endif
        case '-': {
          option flag = getOption(std::string(argv[i]));
          // Long options.
          switch (flag) {
            case apply_algos:
              if (i >= argc - 1 || argv[i + 1][0] == '-') {
                Logger::get(Logger::Info)
                    << "--apply_algos option used without parameter, listing available algorithms:" << std::endl;
                std::vector<std::string> algos;
                Util::Algorithm::list(algos);
                for (size_t i = 0; i < algos.size(); ++i) {
                  Logger::get(Logger::Info) << std::endl << algos[i] << std::endl;
                  Logger::get(Logger::Info) << Util::Algorithm::getDocString(algos[i]) << std::endl;
                }
                return 0;
              } else {
                algoFile = argv[++i];
              }
              break;
            case check_gpu_compatibility:
              if (!Cmd::checkGPUDevice(deviceDef)) {
                return 1;
              }
              // checking program compilation for the device
              if (!(VideoStitch::GPU::Context::compileAllKernelsOnSelectedDevice(deviceDef.device, false).ok())) {
                errLog << "Could not build all OpenCL programs for device " << deviceDef.device << std::endl;
                return 1;
              }
              if (!VideoStitch::GPU::Context::destroy().ok()) {
                errLog << "Could not release OpenCL context " << std::endl;
                return 1;
              }
              return 0;
            default:
              errLog << "No such option: " << argv[i] << std::endl << std::endl;
              printUsage(argv[0], errLog);
              return 1;
          }
          break;
        }
        default:
          errLog << "No such option: " << argv[i] << std::endl << std::endl;
          printUsage(argv[0], errLog);
          return 1;
      }
    } else {
      errLog << "No such option: " << argv[i] << std::endl;
      printUsage(argv[0], errLog);
      return 1;
    }
  }

  if (pluginDirs.empty()) {
    Logger::get(Logger::Warning) << "Warning: no plugins directory has been specified. ";
    Logger::get(Logger::Warning) << "Use option -p to set the plugins directory" << std::endl;
  }
  for (std::vector<std::string>::const_iterator l_it = pluginDirs.begin(), l_last = pluginDirs.end(); l_it != l_last;
       ++l_it) {
    VideoStitch::Plugin::loadPlugins(*l_it);
  }

  if (!ptvPath) {
    errLog << "Missing input file. Use -i. Use '-i -' for stdin." << std::endl;
    printUsage(argv[0], errLog);
    return 1;
  }

  // Change directory to the project directory, so that all paths are relative.
  const char* ptvFile = Cmd::changeWorkingPathToPtvFolder(ptvPath);
  if (!ptvFile) {
    return 1;
  }

  // Instantiate a parser.
  Potential<Ptv::Parser> parser = Ptv::Parser::create();
  if (!parser.ok()) {
    errLog << "Error: Cannot create parser." << std::endl;
    return 1;
  }

  // Create a runtime panorama from the parsed project.
  std::unique_ptr<Core::PanoDefinition> pano = Cmd::parsePanoDef(*parser.object(), ptvFile);

  // Init GLFW
  GLFWwindow* offscreenContext = nullptr;

#ifndef GLFWLIB_UNSUPPORTED
  GLFWwindow* overlayDebugContext = nullptr;
  if (parser->getRoot().has("pano") && parser->getRoot().has("pano")->has("overlays")) {
    if (!glfwInit()) {
      errLog << "glfw init failed, will disable overlay" << std::endl;
    } else {
      glfwSetErrorCallback(glfwErrorCallback);
      glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
      glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
      glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
      glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
      // Create the offscreen opengl context before create the opencl context
      glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
      GLFWwindow* window = glfwCreateWindow(1, 1, "", nullptr, nullptr);
      offscreenContext = window;
      glfwMakeContextCurrent(offscreenContext);
    }
  }
#endif

  // check the GPU devices
  if (!Cmd::checkGPUDevice(deviceDef)) {
    return 1;
  }

  if (offscreenContext) {
    glfwMakeContextCurrent(nullptr);
  }

  Potential<Core::ImageMergerFactory> imageMergerFactory =
      parser->getRoot().has("merger")
          ? Core::ImageMergerFactory::createMergerFactory(*parser->getRoot().has("merger"))
          : Status{Origin::PanoramaConfiguration, ErrType::InvalidConfiguration, "Could not find 'merger' field"};

  Potential<Core::ImageWarperFactory> imageWarperFactory =
      Core::ImageWarperFactory::createWarperFactory(parser->getRoot().has("warper"));

  Potential<Core::ImageFlowFactory> imageFlowFactory =
      Core::ImageFlowFactory::createFlowFactory(parser->getRoot().has("flow"));

  // Check if we have an algorithm to process, or if we stitch.
  if (!algoFile.empty()) {
    Potential<Ptv::Parser> algoParser = Ptv::Parser::create();
    if (!algoParser->parse(algoFile)) {
      errLog << "Error: Cannot parse algos config PTV file: " << algoFile << std::endl;
      errLog << algoParser->getErrorMessage() << std::endl;
      return 1;
    }
    const Ptv::Value& algos = algoParser->getRoot();
    if (!(algos.has("algorithms") && algos.has("algorithms")->getType() == Ptv::Value::LIST)) {
      errLog << "Invalid algorithms file." << std::endl;
      return 1;
    }
    const std::vector<Ptv::Value*>& algoDefs = algos.has("algorithms")->asList();
    for (size_t i = 0; i < algoDefs.size(); ++i) {
      const Ptv::Value* algoDef = algoDefs[i];
      Potential<Util::Algorithm> algo =
          Util::Algorithm::create(algoDef->has("name")->asString(), algoDef->has("config"));
      if (!algo.ok()) {
        errLog << "Could not create algo '" << algoDef->has("name")->asString() << "'" << std::endl;
        return 1;
      }
      SimpleProgressReporter reporter;

      /*Specific calibration code*/
      if (algoDef->has("name")->asString() == "calibration") {
        for (auto videoInput : pano->getVideoInputs()) {
          VideoStitch::Core::InputDefinition& inputDef = videoInput.get();
          /*Calibration is looking for a constant value*/
          VideoStitch::Core::GeometryDefinition g = inputDef.getGeometries().at(0);
          auto geometries = inputDef.getGeometries().clone();
          geometries->setConstantValue(g);
          inputDef.replaceGeometries(geometries);
        }
      }

      Potential<Ptv::Value> result = algo->apply(pano.get(), &reporter);
      if (!result.ok()) {
        errLog << "Could not apply algo '" << algoDef->has("name")->asString() << "'" << std::endl;
        return 1;
      }

      if (algoDef->has("output") && algoDef->has("output")->getType() == Ptv::Value::STRING) {
        std::ofstream ofs(algoDef->has("output")->asString());
        if (result.object()) {
          result->printJson(ofs);
          ofs.close();
        }
      }
    }
    if (algos.has("output") && algos.has("output")->getType() == Ptv::Value::STRING) {
      std::ofstream ofs(algos.has("output")->asString());
      std::unique_ptr<Ptv::Value> result(pano->serialize());
      result->printJson(ofs);
      ofs.close();
    }
    if (algos.has("outputPtv") && algos.has("outputPtv")->getType() == Ptv::Value::LIST) {
      const std::vector<Ptv::Value*>& outputPtvDefs = algos.has("outputPtv")->asList();
      for (auto outputPtvDef : outputPtvDefs) {
        // If the output ptv filename is defined
        if (outputPtvDef->has("name")) {
          std::ofstream ofs(outputPtvDef->has("name")->asString());
          std::string outputFilename("");
          std::string outputType("");
          if (outputPtvDef->has("outputFile")) {
            auto outputFile = outputPtvDef->has("outputFile");
            if (outputFile->has("name") && (outputFile->has("name")->getType() == Ptv::Value::STRING)) {
              outputFilename = outputFile->has("name")->asString();
            }

            if (outputFile->has("type") && (outputFile->has("type")->getType() == Ptv::Value::STRING)) {
              outputType = outputFile->has("type")->asString();
            }
          }

          std::unique_ptr<Ptv::Value> result(Ptv::Value::emptyObject());
          for (int i = 0; i < parser->getRoot().size(); i++) {
            if (*parser->getRoot().get(i).first == std::string("pano")) {
              result->push("pano", pano->serialize());
            } else if (*parser->getRoot().get(i).first == std::string("output")) {
              if (outputFilename.length() > 0 || outputType.length() > 0) {
                Ptv::Value* res = Ptv::Value::emptyObject();
                for (int j = 0; j < parser->getRoot().get(i).second->size(); j++) {
                  if (*parser->getRoot().get(i).second->get(j).first == std::string("type") &&
                      outputType.length() > 0) {
                    res->push("type", Ptv::Value::stringObject(outputType));
                  } else if (*parser->getRoot().get(i).second->get(j).first == std::string("filename") &&
                             outputFilename.length() > 0) {
                    res->push("filename", Ptv::Value::stringObject(outputFilename));
                  } else {
                    res->push(*parser->getRoot().get(i).second->get(j).first,
                              parser->getRoot().get(i).second->get(j).second->clone());
                  }
                }
                result->push("output", res);
              } else {
                result->push(*parser->getRoot().get(i).first, parser->getRoot().get(i).second->clone());
              }
            } else {
              result->push(*parser->getRoot().get(i).first, parser->getRoot().get(i).second->clone());
            }
            // Overwrite the output format
          }
          result->printJson(ofs);
          ofs.close();
        }
      }
    }
    if (!(algos.has("proceed") && algos.has("proceed")->asBool())) {
      VideoStitch::GPU::Context::destroy();
      return 0;
    }
  }

  Logger::get(Logger::Info) << "Panorama size: " << pano->getWidth() << "x" << pano->getHeight() << std::endl;

#ifdef _MSC_VER
  SetConsoleCtrlHandler(signalHandler, TRUE);
#else
  std::signal(SIGTERM, sigTermHandler);
#endif

  if (!parser->getRoot().has("output")) {
    Logger::get(Logger::Error) << "Missing output config" << std::endl;
    return 1;
  }
  std::unique_ptr<Ptv::Value> outputConfig(parser->getRoot().has("output")->clone());
  if (!maybeCheckFilePermissions(outputConfig.get())) {
    return 1;
  }

  // Audio output characteristics
  Audio::SamplingRate outRate = Audio::SamplingRate::SR_48000;
  Audio::SamplingDepth outDepth = Audio::SamplingDepth::FLT_P;
  Audio::ChannelLayout outLayout = Audio::STEREO;
  if (outputConfig->has("audio_codec")) {
    if (!outputConfig->has("sample_format")) {
      Logger::get(Logger::Error) << "Output audio: missing sample format" << std::endl;
      return 1;
    }
    const std::string sample_format = parser->getRoot().has("output")->has("sample_format")->asString();
    if (!sample_format.compare("s8")) {
      outDepth = Audio::SamplingDepth::UINT8;
    } else if (!sample_format.compare("s16")) {
      outDepth = Audio::SamplingDepth::INT16;
    } else if (!sample_format.compare("s32")) {
      outDepth = Audio::SamplingDepth::INT32;
    } else if (!sample_format.compare("flt")) {
      outDepth = Audio::SamplingDepth::FLT;
    } else if (!sample_format.compare("dbl")) {
      outDepth = Audio::SamplingDepth::DBL;
    } else if (!sample_format.compare("s8p")) {
      outDepth = Audio::SamplingDepth::UINT8_P;
    } else if (!sample_format.compare("s16p")) {
      outDepth = Audio::SamplingDepth::INT16_P;
    } else if (!sample_format.compare("s32p")) {
      outDepth = Audio::SamplingDepth::INT32_P;
    } else if (!sample_format.compare("fltp")) {
      outDepth = Audio::SamplingDepth::FLT_P;
    } else if (!sample_format.compare("dblp")) {
      outDepth = Audio::SamplingDepth::DBL_P;
    } else {
      Logger::get(Logger::Error) << "Output audio: invalid sample format" << std::endl;
      return 1;
    }

    if (!outputConfig->has("sampling_rate") ||
        (outputConfig->has("sampling_rate")->asInt() != 32000 && outputConfig->has("sampling_rate")->asInt() != 44100 &&
         outputConfig->has("sampling_rate")->asInt() != 48000)) {
      Logger::get(Logger::Error) << "Output audio: invalid sample rate" << std::endl;
      return 1;
    }
    outRate = Audio::getSamplingRateFromInt(static_cast<int>(outputConfig->has("sampling_rate")->asInt()));

    if (!outputConfig->has("channel_layout")) {
      Logger::get(Logger::Error) << "Output audio: invalid channel layout" << std::endl;
      return 1;
    }
    std::string channel_layout = outputConfig->has("channel_layout")->asString();
    outLayout = Audio::getChannelLayoutFromString(channel_layout.c_str());
  }

  Core::StereoRigDefinition* rigDef = nullptr;
  if (parser->getRoot().has("rig")) {
    rigDef = Core::StereoRigDefinition::create(*parser->getRoot().has("rig"));
  }

  std::unique_ptr<Core::AudioPipeDefinition> audioPipeDef;
  if (parser->getRoot().has("audio_pipe")) {
    audioPipeDef = std::unique_ptr<Core::AudioPipeDefinition>(
        Core::AudioPipeDefinition::create(*parser->getRoot().has("audio_pipe")));
  } else {
    audioPipeDef = std::unique_ptr<Core::AudioPipeDefinition>(
        Core::AudioPipeDefinition::createAudioPipeFromPanoInputs(pano.get()));
  }

  Input::ReaderFactory* readerFactory = new Input::DefaultReaderFactory(firstFrame, lastFrame);
  if (!readerFactory) {
    Logger::get(Logger::Error) << "Reader factory creation failed!" << std::endl;
    return 1;
  }

#ifndef GLFWLIB_UNSUPPORTED
  // Create Overlay object
  std::shared_ptr<VideoStitch::GPU::Overlayer> overlay;
  LockableContext data;
  std::thread renderThread;

  if (offscreenContext) {
    if (pano->numOverlays()) {
      if (enableDebugWindow) {
        glfwWindowHint(GLFW_VISIBLE, GLFW_TRUE);
      }
      GLFWwindow* window = glfwCreateWindow((int)pano->getWidth(), (int)pano->getHeight(), "overlay renderer", nullptr,
                                            offscreenContext);
      overlayDebugContext = window;
      data.window = overlayDebugContext;

      overlay = std::shared_ptr<VideoStitch::GPU::Overlayer>(new Compositor(data));
    } else {
      glfwSetWindowShouldClose(offscreenContext, GLFW_TRUE);
      glfwTerminate();
      offscreenContext = nullptr;
      data.window = overlayDebugContext;
    }
  }
#endif

  bool success = false;
  if (imageMergerFactory.ok()) {
    if (rigDef != nullptr) {
      Core::PotentialStereoController controller =
          Core::createController(*pano, *rigDef, *imageMergerFactory.object(), *imageWarperFactory.object(),
                                 *imageFlowFactory.object(), readerFactory);
      if (!controller.ok()) {
        Logger::get(Logger::Error) << "Controller creation failed!" << std::endl;
        return 1;
      }
      // Check the export configuration
      if (!Cmd::normalizeFrameBoundaries(*controller.object(), firstFrame, lastFrame)) {
        return 1;
      }

      Output::Output* writer =
          makeCallback(controller.object(), *pano, outputConfig.get(), outRate, outDepth, outLayout);
      if (!writer) {
        return 1;
      }

      std::shared_ptr<Output::StereoWriter> stereoWriter(
          Output::StereoWriter::createComposition(writer->getVideoWriter(), Output::StereoWriter::VerticalLayout, Host)
              .release());
      Potential<Core::StereoOutput> stitchOutput = makeOutput(controller.object(), *pano, stereoWriter);
      if (!stitchOutput.ok()) {
        return 1;
      }
      success = stitchStereo(offscreenContext, controller.object(), stitchOutput.object(), firstFrame, lastFrame);
    } else {
      Core::PotentialController controller =
          Core::createController(*pano, *imageMergerFactory.object(), *imageWarperFactory.object(),
                                 *imageFlowFactory.object(), readerFactory, *audioPipeDef.get());
      if (!controller.ok()) {
        Logger::get(Logger::Error) << "Controller creation failed! " << controller.status().getErrorMessage()
                                   << std::endl;
        return 1;
      }
      // Check the export configuration
      if (!Cmd::normalizeFrameBoundaries(*controller.object(), firstFrame, lastFrame)) {
        return 1;
      }

      std::shared_ptr<Output::Output> sharedWriter(
          makeCallback(controller.object(), *pano, outputConfig.get(), outRate, outDepth, outLayout));
      if (!sharedWriter) {
        return 1;
      }

      {
        Potential<Core::StitchOutput> stitchOutput =
            makeOutput(controller.object(), *pano,
                       std::dynamic_pointer_cast<VideoStitch::Output::VideoWriter>(sharedWriter), offscreenContext);
        if (!stitchOutput.ok()) {
          return 1;
        }
        if (sharedWriter->getAudioWriter() != nullptr) {
          controller->addAudioOutput(std::dynamic_pointer_cast<VideoStitch::Output::AudioWriter>(sharedWriter));
        }
#ifndef GLFWLIB_UNSUPPORTED
        if (overlay) {
          // activate the main context to create the framebuffer and texture
          glfwMakeContextCurrent(overlayDebugContext);
          overlay->initialize(pano.get(), controller->getFrameRate());
          glfwMakeContextCurrent(nullptr);
          stitchOutput->setCompositor(overlay);
          if (overlayDebugContext && enableDebugWindow) {
            renderThread = std::thread(renderThreadLoop, std::ref(data), overlay);
          }
        }
#endif

        success = stitchMono(offscreenContext, controller.object(), stitchOutput.object(), firstFrame, lastFrame);
      }
      controller->removeAudioOutput(sharedWriter->getName());
    }
  }

  // Display the output file size
  if (outputConfig->has("filename") && outputConfig->has("type")) {
    std::stringstream outputFile;
    outputFile << outputConfig->has("filename")->asString() << "." << outputConfig->has("type")->asString();
    const std::streamoff fileSize = getFileSize(outputFile.str());
    if (fileSize != -1) {
      Logger::get(Logger::Info) << "Output size: " << std::fixed << std::setprecision(2)
                                << static_cast<double>(fileSize) / (1024.0 * 1024.0) << " MiB" << std::endl;
    }
  }

#ifndef GLFWLIB_UNSUPPORTED
  if (overlayDebugContext) {
    glfwSetWindowShouldClose(overlayDebugContext, GLFW_TRUE);
    if (renderThread.joinable()) {
      renderThread.join();
    }
    glfwTerminate();
  }
#endif
  VideoStitch::GPU::Context::destroy();

  return success ? 0 : 1;
}
