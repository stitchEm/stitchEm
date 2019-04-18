// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "gpu/testing.hpp"
#include "parallax/noFlow.hpp"
#include "parallax/noWarper.hpp"

#include "libvideostitch/controller.hpp"
#include "libvideostitch/inputFactory.hpp"
#include "libvideostitch/imageMergerFactory.hpp"
#include "libvideostitch/panoDef.hpp"
#include "libvideostitch/logging.hpp"
#include <parse/json.hpp>

#include <memory>

using namespace VideoStitch;

namespace VideoStitch {
namespace Testing {

Ptv::Value *createMinimalPTV() {
  // ************* ALL THE VALUES BETWEEN COMMENTS ARE REQUIRED (begin) *************
  // build minimal global PTV
  Ptv::Value *ptv = Ptv::Value::emptyObject();
  ptv->push("width", new Parse::JsonValue(1024));
  ptv->push("height", new Parse::JsonValue(512));
  ptv->push("hfov", new Parse::JsonValue(360));
  ptv->push("proj", new Parse::JsonValue("equirectangular"));

  // add an input (required by Controller)
  Ptv::Value *jsonInputs = new Parse::JsonValue((void *)NULL);
  Ptv::Value *input = Ptv::Value::emptyObject();
  input->push("width", new Parse::JsonValue(1024));
  input->push("height", new Parse::JsonValue(512));
  input->push("hfov", new Parse::JsonValue(360));
  input->push("yaw", new Parse::JsonValue(0.0));
  input->push("pitch", new Parse::JsonValue(0.0));
  input->push("roll", new Parse::JsonValue(0.0));
  input->push("proj", new Parse::JsonValue("equirectangular"));
  input->push("viewpoint_model", new Parse::JsonValue("hugin"));
  input->push("response", new Parse::JsonValue("emor"));

  // add a procedural input
  Ptv::Value *inputConfig = Ptv::Value::emptyObject();
  inputConfig->push("filename", new Parse::JsonValue("toto"));
  inputConfig->push("type", new Parse::JsonValue("procedural"));
  inputConfig->push("name", new Parse::JsonValue("frameNumber"));
  input->push("reader_config", inputConfig);
  jsonInputs->asList().push_back(input);
  ptv->push("inputs", jsonInputs);
  // ************* ALL THE VALUES BETWEEN COMMENTS ARE REQUIRED (end) *************

  return ptv;
}

}  // namespace Testing
}  // namespace VideoStitch

int main() {
  VideoStitch::Testing::initTest();
  VideoStitch::Testing::ENSURE(VideoStitch::GPU::setDefaultBackendDevice(0));

  // ************* ALL THE VALUES BETWEEN COMMENTS ARE REQUIRED (begin) *************
  const int firstFrame = 0, lastFrame = 0;
  // ************* ALL THE VALUES BETWEEN COMMENTS ARE REQUIRED (end) *************

  std::unique_ptr<Ptv::Value> ptv(Testing::createMinimalPTV());

  // PanoDefinition creation should fill with default values
  std::unique_ptr<Core::PanoDefinition> pano(Core::PanoDefinition::create(*ptv.get()));
  std::unique_ptr<Core::AudioPipeDefinition> audioPipe(Core::AudioPipeDefinition::createDefault());
  if (!pano.get()) {
    std::cout << "PanoDefinition creation failed. Needed value to build it may have changed." << std::endl;
    return 1;
  }

  // Create a controller
  Input::DefaultReaderFactory *readerFactory = new Input::DefaultReaderFactory(firstFrame, lastFrame);
  if (!readerFactory) {
    Logger::get(Logger::Error) << "Reader factory creation failed!" << std::endl;
    return 1;
  }
  Potential<Core::ImageMergerFactory> mergerFactory(Core::ImageMergerFactory::newImpotentMergerFactory());
  Core::PotentialController controller =
      Core::createController(*pano.get(), *mergerFactory.object(), Core::NoWarper::Factory(), Core::NoFlow::Factory(),
                             readerFactory, *audioPipe);
  if (!controller.ok()) {
    Logger::get(Logger::Error) << "Controller creation failed!" << std::endl;
    return 1;
  }

  return 0;
}
