// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "gpu/testing.hpp"

#include "libvideostitch/panoDef.hpp"
#include "libvideostitch/parse.hpp"
#include "libvideostitch/ptv.hpp"
#include "libvideostitch/logging.hpp"

#include <algorithm>
#include <memory>
#include <fstream>
#include <sstream>

using namespace VideoStitch;

namespace VideoStitch {
namespace Testing {

static const std::string jsonSerializedFileName = "data/ptvTest.ptv";
static std::string jsonSerialized;

std::string loadJsonIntoString(const std::string jsonPtvFileName) {
  std::ifstream ifs(jsonPtvFileName);
  return std::string((std::istreambuf_iterator<char>(ifs)), (std::istreambuf_iterator<char>()));
}

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

std::string parseAndSerialize() {
  const auto pano = parseIntoPano(jsonSerialized);
  Logger::get(Logger::Info) << "serializing" << std::endl;
  const std::unique_ptr<Ptv::Value> value(pano->serialize());
  std::stringstream ss;
  Logger::get(Logger::Info) << "printing" << std::endl;
  value->printJson(ss);
  return ss.str();
}

void testPanoDefinitionSerializationStringEquals() {
  std::string generated = parseAndSerialize();
  generated.erase(
      std::remove_if(generated.begin(), generated.end(),
                     [](char c) { return !(isalnum(c) || c == '{' || c == '}' || c == '[' || c == ']' || c == ':'); }),
      generated.end());

  std::string updatedjson = jsonSerialized;
  updatedjson.erase(
      std::remove_if(updatedjson.begin(), updatedjson.end(),
                     [](char c) { return !(isalnum(c) || c == '{' || c == '}' || c == '[' || c == ']' || c == ':'); }),
      updatedjson.end());

  ENSURE_EQ(updatedjson, generated);
}

void testPanoDefinitionSerializationObjectEquals() {
  const auto panoFromJson = parseIntoPano(jsonSerialized);

  std::string generated = parseAndSerialize();
  const auto panoFromGenerated = parseIntoPano(generated);

  ENSURE(*panoFromJson == *panoFromGenerated,
         "Pano object `p` generated from json should equal pano object created from serialized, parsed `p`");
}

}  // namespace Testing
}  // namespace VideoStitch

int main() {
  VideoStitch::Testing::initTest();

  // Load PTV
  VideoStitch::Testing::jsonSerialized =
      VideoStitch::Testing::loadJsonIntoString(VideoStitch::Testing::jsonSerializedFileName);

  // This checks the consistency of the PanoDefinition. You'll need to change this if you made big changes to pano
  // serialization or if you change default values.
  VideoStitch::Testing::testPanoDefinitionSerializationStringEquals();
  VideoStitch::Testing::testPanoDefinitionSerializationObjectEquals();
  return 0;
}
