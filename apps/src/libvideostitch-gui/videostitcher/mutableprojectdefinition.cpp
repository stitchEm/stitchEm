// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "mutableprojectdefinition.hpp"

#include "libvideostitch-base/logmanager.hpp"

#include "libvideostitch/inputFactory.hpp"
#include "libvideostitch/logging.hpp"
#include "libvideostitch/parse.hpp"
#include "version.hpp"

MutableProjectDefinition::MutableProjectDefinition(VideoStitch::Core::PanoDefinition* pano,
                                                   VideoStitch::Core::AudioPipeDefinition* audioPipe,
                                                   VideoStitch::Core::StereoRigDefinition* rig, int bufferFrames,
                                                   VideoStitch::Core::ImageMergerFactory* mergerFactory,
                                                   VideoStitch::Core::ImageWarperFactory* warperFactory,
                                                   VideoStitch::Core::ImageFlowFactory* flowFactory, QString libVersion)
    : pano(pano),
      audioPipe(audioPipe),
      rig(rig),
      bufferFrames(bufferFrames),
      mergerFactory(mergerFactory),
      flowFactory(flowFactory),
      warperFactory(warperFactory),
      libVersion(libVersion) {}

MutableProjectDefinition::MutableProjectDefinition(const MutableProjectDefinition& that)
    : pano(that.pano->clone()),
      audioPipe(that.audioPipe ? that.audioPipe->clone() : nullptr),
      rig(that.rig ? that.rig->clone() : nullptr),
      bufferFrames(that.bufferFrames),
      mergerFactory(that.mergerFactory->clone()),
      flowFactory(that.flowFactory->clone()),
      warperFactory(that.warperFactory->clone()),
      libVersion(that.libVersion) {}

MutableProjectDefinition::~MutableProjectDefinition() {
  delete pano;
  delete audioPipe;
  delete rig;
  delete mergerFactory;
  delete warperFactory;
  delete flowFactory;
}

bool MutableProjectDefinition::hasFileFormatChanged() const {
  if (libVersion.isEmpty()) {
    return true;
  }

  QString newLibVersion = LIB_VIDEOSTITCH_VERSION;
  QString simplifiedNewVersion = newLibVersion.left(newLibVersion.indexOf("-"));
  QString simplifiedVersion = libVersion.left(libVersion.indexOf("-"));
  return QString::compare(simplifiedVersion, simplifiedNewVersion) < 0;
}

void MutableProjectDefinition::updateFileFormat() { libVersion = LIB_VIDEOSTITCH_VERSION; }

MutableProjectDefinition* MutableProjectDefinition::create(const VideoStitch::Ptv::Value& value) {
  // Make sure value is an object.
  if (!VideoStitch::Parse::checkType("ProjectDefinition", value, VideoStitch::Ptv::Value::OBJECT)) {
    return nullptr;
  }

  std::string versionStr;
  VideoStitch::Parse::populateString("ProjectDefinition", value, "lib_version", versionStr, false);

  int bufferFrames = 1;
  if (VideoStitch::Parse::populateInt("ProjectDefinition", value, "buffer_frames", bufferFrames, false) ==
      VideoStitch::Parse::PopulateResult_WrongType) {
    return nullptr;
  }

  const VideoStitch::Ptv::Value* tmp = value.has("merger");
  if (!tmp) {
    VideoStitch::Logger::get(VideoStitch::Logger::Warning)
        << "Missing field 'merger' for BaseProjectDefinition." << std::endl;
    return nullptr;
  }
  VideoStitch::Potential<VideoStitch::Core::ImageMergerFactory> mergerFactory =
      VideoStitch::Core::ImageMergerFactory::createMergerFactory(*tmp);
  // TODOLATERSTATUS: propagate and display failure
  if (!mergerFactory.ok()) {
    return nullptr;
  }

  tmp = value.has("warper");

  VideoStitch::Potential<VideoStitch::Core::ImageWarperFactory> warperFactory =
      VideoStitch::Core::ImageWarperFactory::createWarperFactory(tmp);
  if (!warperFactory.ok()) {
    return nullptr;
  }

  tmp = value.has("flow");
  VideoStitch::Potential<VideoStitch::Core::ImageFlowFactory> flowFactory =
      VideoStitch::Core::ImageFlowFactory::createFlowFactory(tmp);
  if (!flowFactory.ok()) {
    return nullptr;
  }

  tmp = value.has("pano");
  if (!tmp) {
    VideoStitch::Logger::get(VideoStitch::Logger::Warning)
        << "Missing field 'pano' for BaseProjectDefinition." << std::endl;
    return nullptr;
  }
  VideoStitch::Core::PanoDefinition* pano = VideoStitch::Core::PanoDefinition::create(*tmp);
  if (!pano) {
    VideoStitch::Logger::get(VideoStitch::Logger::Error) << "Error: failed to create pano definition" << std::endl;
    return nullptr;
  }
  // probe inputs
  std::unique_ptr<VideoStitch::Input::DefaultReaderFactory> factory(
      new VideoStitch::Input::DefaultReaderFactory(0, NO_LAST_FRAME));
  std::unique_ptr<VideoStitch::Ptv::Value> val(VideoStitch::Ptv::Value::emptyObject());

  for (int inputId = 0; inputId < pano->numInputs(); ++inputId) {
    VideoStitch::Core::InputDefinition& input = pano->getInput(inputId);
    val->asString() = pano->getInput(inputId).getReaderConfig().asString();
    VideoStitch::Input::ProbeResult result = factory->probe(*val);

    if (result.valid) {
      input.setIsAudioEnabled(result.hasAudio);
      input.setIsVideoEnabled(result.hasVideo);
    }
  }

  tmp = value.has("rig");
  VideoStitch::Core::StereoRigDefinition* rig = nullptr;
  if (!tmp) {
    VideoStitch::Logger::get(VideoStitch::Logger::Warning)
        << "Missing field 'rig' for BaseProjectDefinition." << std::endl;
  } else {
    rig = VideoStitch::Core::StereoRigDefinition::create(*tmp);
    if (!rig) {
      return nullptr;
    }
  }

  VideoStitch::Core::AudioPipeDefinition* audioPipe = nullptr;
  tmp = value.has("audio_pipe");
  if (tmp && pano->numAudioInputs() > 0) {
    audioPipe = VideoStitch::Core::AudioPipeDefinition::create(*tmp);
    if (audioPipe->numAudioInputs() != pano->numAudioInputs() || audioPipe->numAudioMixes() == 0 ||
        !audioPipe->readersAreConsistent(pano)) {
      delete audioPipe;
      audioPipe = VideoStitch::Core::AudioPipeDefinition::createAudioPipeFromPanoInputs(pano);
    }
  } else if (pano->numAudioInputs() > 0) {
    audioPipe = VideoStitch::Core::AudioPipeDefinition::createAudioPipeFromPanoInputs(pano);
  } else {
    audioPipe = VideoStitch::Core::AudioPipeDefinition::createDefault();
  }
  audioPipe->setSamplingRate(48000);

  return new MutableProjectDefinition(pano, audioPipe, rig, bufferFrames, mergerFactory.release(),
                                      warperFactory.release(), flowFactory.release(),
                                      QString::fromStdString(versionStr));
}

VideoStitch::Ptv::Value* MutableProjectDefinition::serialize() const {
  VideoStitch::Ptv::Value* root = VideoStitch::Ptv::Value::emptyObject();
  root->get("lib_version")->asString() = LIB_VIDEOSTITCH_VERSION;
  root->get("buffer_frames")->asInt() = bufferFrames;
  if (mergerFactory != nullptr) {
    delete root->push("merger", mergerFactory->serialize());
  }
  if (warperFactory != nullptr) {
    delete root->push("warper", warperFactory->serialize());
  }
  if (flowFactory != nullptr) {
    delete root->push("flow", flowFactory->serialize());
  }
  if (pano != nullptr) {
    delete root->push("pano", pano->serialize());
  }
  if (rig != nullptr) {
    delete root->push("rig", rig->serialize());
  }

  if (audioPipe != nullptr && audioPipe->numAudioInputs() > 0) {
    delete root->push("audio_pipe", audioPipe->serialize());
  }
  return root;
}

void MutableProjectDefinition::setAudioPipe(VideoStitch::Core::AudioPipeDefinition* audioPipeDefinition) {
  delete audioPipe;
  audioPipe = audioPipeDefinition;
}

void MutableProjectDefinition::setPano(VideoStitch::Core::PanoDefinition* panoDefinition) {
  delete pano;
  pano = panoDefinition;
}

void MutableProjectDefinition::setRig(VideoStitch::Core::StereoRigDefinition* rigDefinition) {
  delete rig;
  rig = rigDefinition;
}

void MutableProjectDefinition::setMergerFactory(VideoStitch::Core::ImageMergerFactory* merger) {
  delete mergerFactory;
  mergerFactory = merger;
}

void MutableProjectDefinition::setWarperFactory(VideoStitch::Core::ImageWarperFactory* warper) {
  delete warperFactory;
  warperFactory = warper;
}

void MutableProjectDefinition::setFlowFactory(VideoStitch::Core::ImageFlowFactory* flow) {
  delete flowFactory;
  flowFactory = flow;
}
