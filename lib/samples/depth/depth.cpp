// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "libgpudiscovery/backendLibHelper.hpp"
#include "libgpudiscovery/fileHelper.hpp"

#include <libvideostitch/depthController.hpp>
#include <libvideostitch/depthDef.hpp>
#include <libvideostitch/logging.hpp>
#include <libvideostitch/ptv.hpp>
#include <libvideostitch/inputDef.hpp>
#include <libvideostitch/panoDef.hpp>
#include <libvideostitch/parse.hpp>
#include <libvideostitch/inputFactory.hpp>
#include <libvideostitch/audioPipeDef.hpp>
#include <libvideostitch/profile.hpp>
#include <libvideostitch/stitchOutput.hpp>
#include <libvideostitch/allocator.hpp>
#include <libvideostitch/gpu_device.hpp>
#include "version.hpp"

#include "../common/cmdUtils.hpp"

#include <fstream>
#include <iostream>
#include <cstring>
#include <sstream>

using namespace VideoStitch;
using VideoStitch::Logger;

namespace {

void printUsage(const char* execName, VideoStitch::ThreadSafeOstream& os) {
  os << "Usage: " << execName << " [options] -i <input.ptv>" << std::endl;
  os << "Options are:" << std::endl;
  os << "  -f <int>: First frame of the project. Use the first stitchable frame if absent." << std::endl;
  os << "  -l <int>: Last frame of the project. Use the last stitchable frame if absent." << std::endl;
  os << "  -s <int>: Number of levels for pyramid computation. Enables multi-scale processing." << std::endl;
  os << "  -p <directory>: scans directory and loads found plug-ins."
     << " This option can be used several times." << std::endl;
  os << "  -v <q|0|1|2|3|4> : Log level: quiet, error, warning, info (default), verbose, debug." << std::endl;
}

bool parsePyramidLevels(int argc, char** argv, int index, int* numLevels) {
  if (index >= argc - 1 || (argv[index + 1][0] == '-' && argv[index + 1][1] != '\0')) {
    Logger::get(Logger::Error) << "The -s option takes a parameter." << std::endl;
    return false;
  }
  *numLevels = atoi(argv[index + 1]);
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  // Prints the SDK version
  if (argc == 2 && !strcmp(argv[1], "--version")) {
    std::cout << "depth, copyright (c) 2018 stitchEm" << std::endl;
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

  // Parse command line
  Core::PanoDeviceDefinition deviceDef;
  Discovery::getBackendDeviceIndex(deviceId, deviceDef.device);

  char* ptvInPath = nullptr;

  std::string outfn;
  std::vector<std::string> pluginDirs;

  int firstFrame = 0;
  int lastFrame = NO_LAST_FRAME;

  int numPyramidLevels = 0;

  for (int i = 1; i < argc; ++i) {
    if (argv[i][0] != '\0' && argv[i][1] != '\0' && argv[i][0] == '-') {
      switch (argv[i][1]) {
        case 'i':
          if (!Cmd::parseInputPath(argc, argv, i, &ptvInPath)) {
            printUsage(argv[0], errLog);
            return 1;
          }
          i++;
          break;
        case 'f':
          if (!Cmd::parseFirstFrame(argc, argv, i, &firstFrame)) {
            printUsage(argv[0], errLog);
            return 1;
          }
          i++;
          break;
        case 'l':
          if (!Cmd::parseLastFrame(argc, argv, i, &lastFrame)) {
            printUsage(argv[0], errLog);
            return 1;
          }
          i++;
          break;
        case 's':
          if (!parsePyramidLevels(argc, argv, i, &numPyramidLevels)) {
            printUsage(argv[0], errLog);
            return 1;
          }
          i++;
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
        default:
          errLog << "Argument `" << argv[i] << "` not understood." << std::endl;
          printUsage(argv[0], Logger::get(Logger::Error));
          return 1;
      }
    } else if (outfn.empty()) {
      outfn = argv[i];
    } else {
      printUsage(argv[0], Logger::get(Logger::Error));
      return 1;
    }
  }

  if (!ptvInPath) {
    Logger::get(Logger::Error) << "Error: no input file." << std::endl;
    printUsage(argv[0], Logger::get(Logger::Error));
    return 1;
  }

  if (pluginDirs.empty()) {
    Logger::get(Logger::Warning) << "Warning: no plugins directory has been specified. ";
    Logger::get(Logger::Warning) << "Use option -p to set the plugins directory" << std::endl;
  }
  for (const std::string& pluginDir : pluginDirs) {
    VideoStitch::Plugin::loadPlugins(pluginDir);
  }

  // check the GPU devices
  if (!Cmd::checkGPUDevice(deviceDef)) {
    return 1;
  }

  const char* ptvFile = Cmd::changeWorkingPathToPtvFolder(ptvInPath);
  if (!ptvFile) {
    printUsage(argv[0], errLog);
    return 1;
  }

  std::unique_ptr<Input::ReaderFactory> readerFactory(new Input::DefaultReaderFactory(firstFrame, lastFrame));
  if (!readerFactory) {
    Logger::get(Logger::Error) << "Reader factory creation failed!" << std::endl;
    return 1;
  }

  // Instantiate a parser.
  Potential<Ptv::Parser> parser = Ptv::Parser::create();
  if (!parser.ok()) {
    Logger::get(Logger::Error) << "Error: Cannot create parser." << std::endl;
    return 1;
  }

  std::unique_ptr<Ptv::Value> ptvRoot;
  std::unique_ptr<Core::PanoDefinition> pano;
  if (Cmd::parsePtvFile(*parser.object(), ptvFile)) {
    ptvRoot.reset(parser->getRoot().clone());
    pano = Cmd::parsePanoDef(*ptvRoot, ptvFile);
  }

  if (!pano) {
    return 1;
  }

  // force sphereScale to be 1.0
  if (pano->getSphereScale() != 1.0) {
    Logger::get(Logger::Info) << "Info: overrriding sphere scale to be 1.0." << std::endl;
    pano->setSphereScale(1.0);
  }

  std::unique_ptr<Core::AudioPipeDefinition> audioPipeDef;
  if (parser->getRoot().has("audio_pipe")) {
    audioPipeDef = std::unique_ptr<Core::AudioPipeDefinition>(
        Core::AudioPipeDefinition::create(*parser->getRoot().has("audio_pipe")));
  } else {
    audioPipeDef = std::unique_ptr<Core::AudioPipeDefinition>(
        Core::AudioPipeDefinition::createAudioPipeFromPanoInputs(pano.get()));
  }

  Core::DepthDefinition depthDef{};
  depthDef.setNumPyramidLevels(numPyramidLevels);

  Potential<Core::DepthController> ctrl =
      Core::createDepthController(*pano, depthDef, *audioPipeDef, readerFactory.release());
  if (!ctrl.ok()) {
    errLog << "DepthController creation failed!" << std::endl;
    return 1;
  }

  if (!Cmd::normalizeFrameBoundaries(*ctrl.object(), firstFrame, lastFrame)) {
    return 1;
  }

  std::vector<std::unique_ptr<Core::ExtractOutput>> extractsOutputs;
  std::vector<std::shared_ptr<Core::SourceSurface>> surfs;

  for (int id = 0; id < pano->numInputs(); ++id) {
    const VideoStitch::Core::InputDefinition& inputDef = pano->getInput(id);
    if (inputDef.getIsVideoEnabled()) {
      VideoStitch::Input::VideoReader::Spec spec = ctrl->getReaderSpec(id);
      auto potSurf = Core::OffscreenAllocator::createDepthSurface(spec.width, spec.height, "Input extract surface");
      if (!potSurf.ok()) {
        errLog << "Failed to allocate GPU surfaces to process the inputs" << std::endl;
        return 1;
      }
      potSurf.object()->sourceId = id;
      surfs.push_back({std::shared_ptr<Core::SourceSurface>(potSurf.release())});
    }
  }

  if (!parser->getRoot().has("output")) {
    Logger::get(Logger::Error) << "Missing output config" << std::endl;
    return 1;
  }

  // TODO RES-639 replace with .json file?
  std::unique_ptr<Ptv::Value> panoOutputConfig(parser->getRoot().has("output")->clone());
  // TODO warn if we can't write in output dir
  //  if (!maybeCheckFilePermissions(outputConfig.get())) {
  //    return 1;
  //  }

  int i = 0;
  for (int id = 0; id < pano->numInputs(); ++id) {
    const VideoStitch::Core::InputDefinition& inputDef = pano->getInput(id);
    if (inputDef.getIsVideoEnabled()) {
      std::unique_ptr<Ptv::Value> extractOutputConfig(panoOutputConfig->clone());

      // TODO RES-639 error if !->has("filename")

      std::string& filename = extractOutputConfig->get("filename")->asString();

      // TODO RES-639 filename pattern
      filename += "-depth-" + std::to_string(id);
      VideoStitch::Input::VideoReader::Spec spec = ctrl->getReaderSpec(id);

      Potential<Output::Output> potOutput = Output::create(*extractOutputConfig, filename, (unsigned)spec.width,
                                                           (unsigned)spec.height, ctrl->getFrameRate());

      if (!potOutput.ok()) {
        Logger::get(Logger::Error) << "Output writer creation failed!" << std::endl;
        return 1;
      }

      std::shared_ptr<Output::Output> output(potOutput.release());
      Potential<Core::ExtractOutput> potExtractOutput = ctrl->createAsyncExtractOutput(
          id, surfs[i], std::dynamic_pointer_cast<VideoStitch::Output::VideoWriter>(output));
      if (!potExtractOutput.ok()) {
        Logger::get(Logger::Error) << "ExtractOutput creation failed!" << std::endl;
        return 1;
      }
      extractsOutputs.emplace_back(potExtractOutput.release());
      ++i;
    }
  }

  {
    Util::SimpleProfiler globalComputeProf("Global computation time", false, Logger::get(Logger::Info));
    frameid_t curFrame = firstFrame;
    std::vector<Core::ExtractOutput*> rawExtractOutputPtrs;
    auto getRawPtr = [](const std::unique_ptr<Core::ExtractOutput>& ptr) -> Core::ExtractOutput* { return ptr.get(); };
    std::transform(extractsOutputs.begin(), extractsOutputs.end(), std::back_inserter(rawExtractOutputPtrs), getRawPtr);
    while (curFrame <= lastFrame) {
      switch (ctrl->estimateDepth(rawExtractOutputPtrs).getCode()) {
        case Core::ControllerStatusCode::EndOfStream:
          Logger::get(Logger::Info) << "No more input frames available, reached end of stream. Cannot undistort frame "
                                    << curFrame << ", shutting down." << std::endl;
          return 1;
        case Core::ControllerStatusCode::ErrorWithStatus:
          Logger::get(Logger::Error) << "Failed to undistort frame " << curFrame << std::endl;
          return 1;
        case VideoStitch::Core::ControllerStatusCode::Ok:
          break;
      }
      curFrame++;
    }
  }
}
