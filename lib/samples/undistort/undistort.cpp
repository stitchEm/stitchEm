// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "libgpudiscovery/backendLibHelper.hpp"
#include "libgpudiscovery/fileHelper.hpp"

#include <libvideostitch/logging.hpp>
#include <libvideostitch/ptv.hpp>
#include <libvideostitch/inputDef.hpp>
#include <libvideostitch/panoDef.hpp>
#include <libvideostitch/parse.hpp>
#include <libvideostitch/inputFactory.hpp>
#include <libvideostitch/undistortController.hpp>
#include <libvideostitch/audioPipeDef.hpp>
#include <libvideostitch/profile.hpp>
#include <libvideostitch/stitchOutput.hpp>
#include <libvideostitch/allocator.hpp>
#include <libvideostitch/gpu_device.hpp>
#include <libvideostitch/overrideDef.hpp>
#include "version.hpp"

#include "../common/cmdUtils.hpp"

#include <fstream>
#include <iostream>
#include <cstring>
#include <sstream>

using namespace VideoStitch;
using VideoStitch::Logger;

enum class option { reset_rotation, focal, projection, width, height, invalid_option };

namespace {

void printUsage(const char* execName, VideoStitch::ThreadSafeOstream& os) {
  os << "Usage: " << execName << " [options] -i <input.ptv>" << std::endl;
  os << "Options are:" << std::endl;
  os << "  -d <int>: Use the specified device id for undistortion." << std::endl;
  os << "  -f <int>: First frame of the project. Use the first stitchable frame if absent." << std::endl;
  os << "  -l <int>: Last frame of the project. Use the last stitchable frame if absent." << std::endl;
  os << "  -t <output_type>: Set the output type to <output_type> (default 'mp4')." << std::endl;
  os << "  --focal <double>: Replace the output focals by this value. If this parameter is not set the frames' half "
        "size will be used."
     << std::endl;
  os << "  --reset_rotation: Rotate the inputs to yaw=0, pitch=0 and roll=0." << std::endl;
  os << "  --projection <proj>: Use a different projection for the resulting files. Valid values: equirectangular, "
        "ff_fisheye, ff_fisheye_opt, rectilinear."
     << std::endl;
  os << "  --width <int>: Override width of created outputs in pixels. Implies also setting --height." << std::endl;
  os << "  --height <int>: Override height of created outputs in pixels. Implies also setting --width." << std::endl;
  os << "  -o <output.ptv>: Export .ptv" << std::endl;
  os << "  -p <directory>: scans directory and loads found plug-ins."
     << " This option can be used several times." << std::endl;
  os << "  -v <q|0|1|2|3|4> : Log level: quiet, error, warning, info (default), verbose, debug." << std::endl;
  os << "If output is not specified, the same name as input is used, with a 'ptv' extension." << std::endl;
}

option getOption(const std::string& s) {
  if (s == "--reset_rotation") {
    return option::reset_rotation;
  } else if (s == "--focal") {
    return option::focal;
  } else if (s == "--projection") {
    return option::projection;
  } else if (s == "--width") {
    return option::width;
  } else if (s == "--height") {
    return option::height;
  } else {
    return option::invalid_option;
  }
}

}  // namespace

int main(int argc, char** argv) {
  // Prints the SDK version
  if (argc == 2 && !strcmp(argv[1], "--version")) {
    std::cout << "undistort, copyright (c) 2018 stitchEm" << std::endl;
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
  char* ptvOutPath = nullptr;

  std::string outfn;
  std::vector<std::string> pluginDirs;

  int firstFrame = 0;
  int lastFrame = NO_LAST_FRAME;

  Core::OverrideOutputDefinition outputDef{};

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
        case 'o':
          if (!Cmd::parseOutputPath(argc, argv, i, &ptvOutPath)) {
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
        case 't':
          if (i >= argc - 1 || argv[i + 1][0] == '-') {
            Logger::get(Logger::Error) << "The -t option takes a parameter." << std::endl;
            printUsage(argv[0], Logger::get(Logger::Error));
            return 1;
          }
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
        case '-': {
          option flag = getOption(std::string(argv[i]));
          // Long options.
          switch (flag) {
            case option::reset_rotation:
              outputDef.resetRotation = true;
              break;
            case option::focal:
              if (i >= argc - 1 || argv[i + 1][0] == '-') {
                Logger::get(Logger::Error) << "The --focal option takes a parameter." << std::endl;
                printUsage(argv[0], Logger::get(Logger::Error));
                return 1;
              }
              ++i;
              outputDef.overrideFocal = std::atof(argv[i]);
              if (outputDef.overrideFocal > 0.0) {
                outputDef.manualFocal = true;
              } else {
                Logger::get(Logger::Error) << "Error: focal passed to \"--focal\" option is negative" << std::endl;
                printUsage(argv[0], errLog);
                return 1;
              }
              break;
            case option::projection:
              if (i >= argc - 1 || argv[i + 1][0] == '-') {
                Logger::get(Logger::Error) << "The --projection option takes a parameter." << std::endl;
                printUsage(argv[0], Logger::get(Logger::Error));
                return 1;
              }
              ++i;
              if (Core::InputDefinition::getFormatFromName(std::string(argv[i]), outputDef.newFormat)) {
                outputDef.changeOutputFormat = true;
              } else {
                Logger::get(Logger::Error) << "Error: format passed to \"--projection\" option is unknown" << std::endl;
                printUsage(argv[0], errLog);
                return 1;
              }
              break;
            case option::width:
              if (i >= argc - 1 || argv[i + 1][0] == '-') {
                Logger::get(Logger::Error) << "The --width option takes a parameter." << std::endl;
                printUsage(argv[0], Logger::get(Logger::Error));
                return 1;
              }
              ++i;
              outputDef.width = std::atoi(argv[i]);
              if (outputDef.width > 0) {
                outputDef.changeOutputSize = true;
              } else {
                Logger::get(Logger::Error) << "Error: focal passed to \"--width\" option is negative" << std::endl;
                printUsage(argv[0], errLog);
                return 1;
              }
              break;
            case option::height:
              if (i >= argc - 1 || argv[i + 1][0] == '-') {
                Logger::get(Logger::Error) << "The --height option takes a parameter." << std::endl;
                printUsage(argv[0], Logger::get(Logger::Error));
                return 1;
              }
              ++i;
              outputDef.height = std::atoi(argv[i]);
              if (outputDef.height > 0) {
                outputDef.changeOutputSize = true;
              } else {
                Logger::get(Logger::Error) << "Error: focal passed to \"--height\" option is negative" << std::endl;
                printUsage(argv[0], errLog);
                return 1;
              }
              break;
            default:
              errLog << "No such option: " << argv[i] << std::endl << std::endl;
              printUsage(argv[0], errLog);
              return 1;
          }
          break;
        }
        default:
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

  if (outputDef.changeOutputSize && outputDef.height == 0) {
    Logger::get(Logger::Error) << "Error: setting --width implies also providing --height" << std::endl;
    return 1;
  } else if (outputDef.changeOutputSize && outputDef.width == 0) {
    Logger::get(Logger::Error) << "Error: setting --height implies also providing --width" << std::endl;
    return 1;
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

  std::unique_ptr<Core::AudioPipeDefinition> audioPipeDef;
  if (parser->getRoot().has("audio_pipe")) {
    audioPipeDef = std::unique_ptr<Core::AudioPipeDefinition>(
        Core::AudioPipeDefinition::create(*parser->getRoot().has("audio_pipe")));
  } else {
    audioPipeDef = std::unique_ptr<Core::AudioPipeDefinition>(
        Core::AudioPipeDefinition::createAudioPipeFromPanoInputs(pano.get()));
  }

  Potential<Core::UndistortController> ctrl =
      Core::createUndistortController(*pano, *audioPipeDef, readerFactory.release(), outputDef);
  if (!ctrl.ok()) {
    errLog << "UndistortionController creation failed!" << std::endl;
    return 1;
  }

  if (!Cmd::normalizeFrameBoundaries(*ctrl.object(), firstFrame, lastFrame)) {
    return 1;
  }

  std::unique_ptr<Core::PanoDefinition> updatedPano;
  if (ptvOutPath) {
    Potential<Core::PanoDefinition> noDistortion = ctrl->createPanoDefWithoutDistortion();
    if (!noDistortion.ok()) {
      errLog << "Unable to create undistorted panorama configuration" << std::endl;
      return 1;
    }

    updatedPano.reset(noDistortion.release());
  }

  std::vector<std::unique_ptr<Core::ExtractOutput>> extractsOutputs;
  std::vector<std::shared_ptr<Core::SourceSurface>> surfs;

  for (int id = 0; id < pano->numInputs(); ++id) {
    const VideoStitch::Core::InputDefinition& inputDef = pano->getInput(id);
    if (inputDef.getIsVideoEnabled()) {
      VideoStitch::Input::VideoReader::Spec spec = ctrl->getReaderSpec(id);
      auto potSurf = Core::OffscreenAllocator::createSourceSurface(
          (unsigned)(outputDef.changeOutputSize ? outputDef.width : spec.width),
          (unsigned)(outputDef.changeOutputSize ? outputDef.height : spec.height), "Output surfaces");
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
      std::string& type = extractOutputConfig->get("type")->asString();

      if (outputDef.changeOutputSize) {
        extractOutputConfig->get("width")->asInt() = outputDef.width;
        extractOutputConfig->get("height")->asInt() = outputDef.height;
      }

      // TODO RES-639 filename pattern
      filename += "-undistorted-" + std::to_string(id);
      VideoStitch::Input::VideoReader::Spec spec = ctrl->getReaderSpec(id);

      Potential<Output::Output> potOutput = Output::create(
          *extractOutputConfig, filename, (unsigned)(outputDef.changeOutputSize ? outputDef.width : spec.width),
          (unsigned)(outputDef.changeOutputSize ? outputDef.height : spec.height), ctrl->getFrameRate());

      if (!potOutput.ok()) {
        Logger::get(Logger::Error) << "Output writer creation failed!" << std::endl;
        return 1;
      }

      // change file names to newly created undistorted images
      if (updatedPano) {
        if (type == "png" || type == "jpg" || type == "tiff") {
          // account for picture file numbering pattern
          filename += "-%05i";
        }
        filename += "." + type;
        VideoStitch::Core::InputDefinition& updatedInputDef = updatedPano->getInput(id);
        updatedInputDef.setReaderConfig(Ptv::Value::stringObject(filename));
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

  if (updatedPano) {
    // set offset_frame to zero
    for (int id = 0; id < pano->numInputs(); ++id) {
      updatedPano->getInput(id).setFrameOffset(0);
    }

    std::unique_ptr<Ptv::Value> updatedPtvRoot(ptvRoot->clone());
    {
      auto prev = updatedPtvRoot->push("pano", updatedPano->serialize());
      delete prev;
    }
    // update first and last frames
    updatedPtvRoot->get("first_frame")->asInt() = firstFrame;
    updatedPtvRoot->get("last_frame")->asInt() = lastFrame;
    std::ofstream ofs(ptvOutPath, std::ios_base::out);
    if (!ofs.is_open()) {
      Logger::get(Logger::Error) << "Error: cannot open '" << outfn << "' for writing." << std::endl;
      return 1;
    }
    updatedPtvRoot->printJson(ofs);
  }

  {
    Util::SimpleProfiler globalComputeProf("Global computation time", false, Logger::get(Logger::Info));
    frameid_t curFrame = firstFrame;
    std::vector<Core::ExtractOutput*> rawExtractOutputPtrs;
    auto getRawPtr = [](const std::unique_ptr<Core::ExtractOutput>& ptr) -> Core::ExtractOutput* { return ptr.get(); };
    std::transform(extractsOutputs.begin(), extractsOutputs.end(), std::back_inserter(rawExtractOutputPtrs), getRawPtr);
    while (curFrame <= lastFrame) {
      switch (ctrl->undistort(rawExtractOutputPtrs).getCode()) {
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
