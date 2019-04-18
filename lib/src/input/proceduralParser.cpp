// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "proceduralParser.hpp"

#include "util/strutils.hpp"
#include "libvideostitch/logging.hpp"

#include <cstdlib>
#include <iostream>
#include <map>
#include <sstream>
#include <string>

namespace VideoStitch {
namespace Input {

ProceduralInputSpec::ProceduralInputSpec(const std::string& spec) {
  if (!Util::startsWith(spec.c_str(), "procedural:")) {
    return;
  }
  std::string stripped = spec.substr(11);
  if (spec[spec.size() - 1] != ')') {
    name = stripped;
    return;
  }
  size_t openParenthesis = stripped.find('(');
  if (openParenthesis == std::string::npos) {
    Logger::get(Logger::Warning) << "Weird procedural input specification '" << spec
                                 << "'. Trying to make the best out of it." << std::endl;
    name = stripped;
    return;
  }
  name = stripped.substr(0, openParenthesis);
  stripped = stripped.substr(openParenthesis + 1, stripped.size() - openParenthesis - 2);
  if (!stripped.empty()) {
    std::vector<std::string> splitOptions;
    Util::split(stripped.c_str(), ',', &splitOptions);
    for (std::vector<std::string>::const_iterator it = splitOptions.begin(); it != splitOptions.end(); ++it) {
      std::string first, second;
      if (Util::splitOnce(it->c_str(), '=', &first, &second)) {
        std::string& value = options[first];
        if (!value.empty()) {
          Logger::get(Logger::Warning) << "In procedural input specification '" << spec << "', option '" << first
                                       << "' is specified several times." << std::endl;
        }
        value = second;
      } else {
        Logger::get(Logger::Warning) << "In procedural input specification '" << spec << "', ignoring unnamed option '"
                                     << *it << "'." << std::endl;
      }
    }
  }
}

bool ProceduralInputSpec::isProcedural() const { return !name.empty(); }

const std::string& ProceduralInputSpec::getName() const { return name; }

const std::string* ProceduralInputSpec::getOption(const std::string& option) const {
  std::map<std::string, std::string>::const_iterator it = options.find(option);
  if (it == options.end()) {
    return NULL;
  } else {
    return &(it->second);
  }
}

bool ProceduralInputSpec::getIntOption(const std::string& option, int& v) const {
  const std::string* opt = getOption(option);
  if (!opt) {
    return false;
  }
  v = atoi(opt->c_str());
  return true;
}

bool ProceduralInputSpec::getDoubleOption(const std::string& option, double& v) const {
  const std::string* opt = getOption(option);
  if (!opt) {
    return false;
  }
  v = atof(opt->c_str());
  return true;
}

bool ProceduralInputSpec::getColorOption(const std::string& option, uint32_t& v) const {
  const std::string* opt = getOption(option);
  if (!opt) {
    return false;
  }
  return Util::parseHtmlColor(*opt, v);
}

Ptv::Value* ProceduralInputSpec::getPtvConfig() const {
  Ptv::Value* res = Ptv::Value::emptyObject();
  for (MapT::const_iterator it = options.begin(); it != options.end(); ++it) {
    Ptv::Value* entry = Ptv::Value::emptyObject();
    uint32_t dummyColor = 0;
    int tmpInt = 0;
    double tmpDouble = 0.0;
    if (getColorOption(it->first, dummyColor)) {
      entry->asString() = it->second;
    } else if (it->second.find(".") == std::string::npos && getIntOption(it->first, tmpInt)) {
      entry->asInt() = tmpInt;
    } else if (getDoubleOption(it->first, tmpDouble)) {
      entry->asDouble() = tmpDouble;
    } else {
      entry->asString() = it->second;
    }
    delete res->push(it->first, entry);
  }
  Ptv::Value* entry = Ptv::Value::emptyObject();
  entry->asString() = getName();
  delete res->push("name", entry);
  entry = Ptv::Value::emptyObject();
  entry->asString() = "procedural";
  delete res->push("type", entry);
  return res;
}

}  // namespace Input
}  // namespace VideoStitch
