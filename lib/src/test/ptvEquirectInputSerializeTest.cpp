// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "gpu/testing.hpp"

#include "libvideostitch/panoDef.hpp"
#include "libvideostitch/parse.hpp"
#include "libvideostitch/ptv.hpp"
#include "libvideostitch/logging.hpp"

#include <algorithm>
#include <memory>
#include <sstream>

using namespace VideoStitch;

namespace VideoStitch {
namespace Testing {

static const std::string jsonSerialized = R"(
  {
    "width" : 4096,
    "height" : 2048,
    "pad_top" : 0,
    "pad_bottom" : 0,
    "hfov" : 360,
    "proj" : "equirectangular",
    "inputs" : [
                {
                "reader_config" : "equirectwhite4K.jpg",
                "width" : 4096,
                "height" : 2048,
                "proj" : "equirectangular",
                "hfov" : 360,
                "response" : "emor"
                }
              ]
  }
)";

std::unique_ptr<Core::PanoDefinition> parseIntoPano(std::string jsonPtv) {
  Potential<Ptv::Parser> parser = Ptv::Parser::create();
  Logger::get(Logger::Info) << "parsing" << std::endl;
  if (!parser->parseData(jsonPtv)) {
    Logger::get(Logger::Error) << parser->getErrorMessage() << std::endl;
    ENSURE(false);
  }
  Logger::get(Logger::Info) << "creating" << std::endl;
  return std::unique_ptr<Core::PanoDefinition>(Core::PanoDefinition::create(parser->getRoot()));
}

std::unique_ptr<Ptv::Value> parseAndSerialize() {
  const auto pano = parseIntoPano(jsonSerialized);
  ENSURE(pano.get());
  Logger::get(Logger::Info) << "serializing" << std::endl;
  std::unique_ptr<Ptv::Value> value(pano->serialize());
  std::stringstream ss;
  Logger::get(Logger::Info) << "printing" << std::endl;
  value->printJson(ss);
  return std::unique_ptr<Ptv::Value>(value.release());
}

void testPanoDefinitionWithEquirectInputSerializationHorizontalFov() {
  const std::unique_ptr<Ptv::Value> generated = parseAndSerialize();
  // input PandoDefinition had an "hfov" element, serialized one should not
  ENSURE(generated->has("inputs"));
  // single input
  ENSURE(generated->get("inputs")->asList().size() == 1);
  Ptv::Value *input = generated->get("inputs")->asList()[0];
  // equirectangular
  ENSURE(input->get("proj")->asString() == "equirectangular");
  // no hfov
  ENSURE(!input->has("hfov"));
  // should have horizontalFocal in geometries
  ENSURE(input->has("geometries"));
  ENSURE(input->get("geometries")->has("horizontalFocal"));
  ENSURE_EQ(input->get("geometries")->get("horizontalFocal")->asDouble(), 651.8986206054688);
}

}  // namespace Testing
}  // namespace VideoStitch

int main() {
  VideoStitch::Testing::initTest();

  // This checks VSA-7234, testing that horizontalFocal is serialized from input hfov with enough precision for
  // equirectangular inputs.
  VideoStitch::Testing::testPanoDefinitionWithEquirectInputSerializationHorizontalFov();
  return 0;
}
