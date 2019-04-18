// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "libvideostitch/inputFactory.hpp"
#include "libvideostitch/inputDef.hpp"

#include "util/plugin.hpp"

#include "exprReader.hpp"
#include "maskedReader.hpp"
#include "proceduralParser.hpp"
#include "audio/audiogen.hpp"

#include <iostream>
#include <sstream>

namespace VideoStitch {

using namespace Plugin;

namespace Input {

ProbeResult ReaderFactory::probe(const std::string& filename) const {
  std::unique_ptr<Ptv::Value> config(Ptv::Value::emptyObject());
  config->asString() = filename;
  return probe(*config);
}

DefaultReaderFactory::DefaultReaderFactory(frameid_t firstFrame, frameid_t lastFrame)
    : firstFrame(firstFrame < 0 ? 0 : firstFrame), lastFrame(lastFrame) {}

DefaultReaderFactory::~DefaultReaderFactory() {}

frameid_t DefaultReaderFactory::getFirstFrame() const { return firstFrame; }

frameid_t DefaultReaderFactory::getNumFrames() const {
  if (lastFrame < 0) {
    return -1;
  } else {
    return (lastFrame - firstFrame) + 1;
  }
}

Potential<Reader> DefaultReaderFactory::create(readerid_t id, const Core::ReaderInputDefinition& def) const {
  const Ptv::Value& config = def.getReaderConfig();
  if (def.getFrameOffset() < 0) {
    std::stringstream msg;
    msg << "The frame offset is negative (" << def.getFrameOffset() << ")";
    return {Origin::Input, ErrType::InvalidConfiguration, msg.str()};
  }

  Reader* reader = nullptr;
  // try to open with plugins
  {
    std::unique_lock<std::mutex> lock(pluginsMutex);
    for (VSReaderPlugin::InstanceVector::const_iterator l_it = VSReaderPlugin::Instances().begin(),
                                                        l_last = VSReaderPlugin::Instances().end();
         l_it != l_last; ++l_it) {
      if ((*l_it)->handles(&config)) {
        Potential<Reader>* potReader = (*l_it)->create(
            &config, VSReaderPlugin::Config(id, firstFrame, lastFrame, def.getWidth(), def.getHeight()));
        if (!potReader->ok()) {
          std::stringstream msg;
          const Status readerCreationStatus = potReader->status();
          delete potReader;
          msg << "Couldn't create the reader for plugin " + (*l_it)->getName();
          return {Origin::Input, ErrType::SetupFailure, msg.str(), readerCreationStatus};
        } else {
          reader = potReader->release();
          delete potReader;
          break;
        }
      }
    }
  }

  // try to open object-style procedurals
  if (!reader) {
    if (config.has("type")) {
      if (config.has("type")->asString() == "procedural") {
        reader = ProceduralReader::create(id, config, def.getWidth(), def.getHeight());
      } else if (config.has("type")->asString() == Audio::getAudioGeneratorId()) {
        reader = Audio::AudioGenFactory::create(id, config);
      } else {
        return {Origin::Input, ErrType::InvalidConfiguration,
                "Unknown reader configuration type '" + config.has("type")->asString() + "'"};
      }
    }
  }

  // try to open single-line string-style procedurals
  if (!reader) {
    if (config.getType() == Ptv::Value::STRING) {
      const std::string& filename = config.asString();
      // Try to see if we have a procedural reader.
      ProceduralInputSpec proceduralSpec(filename);
      if (proceduralSpec.isProcedural()) {
        // backwards compatibility.
        std::unique_ptr<Ptv::Value> proceduralConfig(proceduralSpec.getPtvConfig());
        reader = ProceduralReader::create(id, *proceduralConfig, def.getWidth(), def.getHeight());
      }
    }
  }

  if (!reader) {
    // failed to create any reader. return an error message for invalid configurations
    if (config.getType() == Ptv::Value::STRING) {
      const std::string& filename = config.asString();
      std::stringstream msg;
      msg << "Could not create a reader for configuration '" << filename << "'";
      if (VSReaderPlugin::Instances().empty()) {
        msg << ". \n"
            << " No input plugin has been loaded. Check your software installation.";
      }
      return {Origin::Input, ErrType::InvalidConfiguration, msg.str()};
    } else {
      return {Origin::Input, ErrType::InvalidConfiguration, "Invalid reader configuration type"};
    }
  }

  // If there is a mask, read the data.
  /*
  if (def.getMaskPixelDataIfValid()) {
    Reader* maskedReader = MaskedReader::create(reader, def.getMaskPixelDataIfValid());
    if (maskedReader) {
      reader = maskedReader;
    } else {
      Logger::get(Logger::Warning) << "Could not set mask for input." << std::endl;
    }
  }
  */

  // Seek to the correct frame:
  Input::VideoReader* videoReader = reader->getVideoReader();
  if (videoReader) {
    frameid_t seekToFrame = firstFrame + def.getFrameOffset();
    if (seekToFrame != 0) {
      if (firstFrame <= seekToFrame && seekToFrame <= videoReader->getLastFrame()) {
        videoReader->seekFrame(seekToFrame);
      } else {
        Logger::get(Logger::Error) << "No such frame " << seekToFrame << " for input " << id << " to seek to";
      }
    }
  }

  return Potential<Reader>(reader);
}

ProbeResult DefaultReaderFactory::probe(const Ptv::Value& config) const {
  if (config.has("type") && config.has("type")->asString() == "procedural") {
    // Procedural readers have no limits and accept any size.
    return ProbeResult({ProceduralReader::isKnown(config), false, 0, NO_LAST_FRAME, -1, -1, false, true});
  }

  if (config.has("type") && config.has("type")->asString() == Audio::getAudioGeneratorId()) {
    // Procedural readers have no limits and accept any size.
    return ProbeResult({ProceduralReader::isKnown(config), false, 0, NO_LAST_FRAME, -1, -1, true, true});
  }

  if (config.getType() == Ptv::Value::STRING) {
    // Filename-based readers.
    const std::string& filename = config.asString();
    for (const VSProbeReaderPlugin* lPlugin : VSProbeReaderPlugin::Instances()) {
      if (lPlugin->handles(&config)) {
        return lPlugin->probe(filename);
      }
    }
    // Try to see if we have a procedural reader.
    ProceduralInputSpec proceduralSpec(filename);
    if (proceduralSpec.isProcedural()) {
      // backwards compatibility.
      std::unique_ptr<Ptv::Value> proceduralConfig(proceduralSpec.getPtvConfig());
      return ProbeResult({ProceduralReader::isKnown(*proceduralConfig), false, 0, NO_LAST_FRAME, -1, -1, false,
                          ProceduralReader::isKnown(*proceduralConfig)});
    }
    return ProbeResult({false, false, -1, -1, -1, -1, false, false});
  }

  return ProbeResult({false, false, -1, -1, -1, -1, false, false});
}
}  // namespace Input
}  // namespace VideoStitch
