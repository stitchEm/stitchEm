// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "util/strutils.hpp"

#include "libvideostitch/logging.hpp"
#include "libvideostitch/ptv.hpp"
#include "libvideostitch/parse.hpp"

namespace VideoStitch {
namespace Parse {
bool checkVar(const std::string& objName, const std::string& varName, const Ptv::Value* var, bool mandatory) {
  if (!var) {
    if (mandatory) {
      Logger::get(Logger::Error) << "Missing mandatory field '" << varName << "' for object " << objName << "."
                                 << std::endl;
    }
    return false;
  } else {
    return true;
  }
}

bool checkType(const std::string& varName, const Ptv::Value& value, Ptv::Value::Type expectedType) {
  if (!value.isConvertibleTo(expectedType)) {
    Logger::get(Logger::Error) << "Wrong type for " << varName << ", expected " << Ptv::Value::getTypeName(expectedType)
                               << ", got " << Ptv::Value::getTypeName(value.getType()) << "." << std::endl;
    return false;
  } else {
    return true;
  }
}

PopulateResult populateBool(const std::string& objName, const Ptv::Value& obj, const std::string& varName, bool& v,
                            bool mandatory) {
  const Ptv::Value* var = obj.has(varName);
  if (!checkVar(objName, varName, var, mandatory)) {
    return PopulateResult_DoesNotExist;
  }
  if (!checkType(varName, *var, Ptv::Value::BOOL)) {
    return PopulateResult_WrongType;
  }
  v = var->asBool();
  return PopulateResult_Ok;
}

template <typename IntT>
PopulateResult populateInt(const std::string& objName, const Ptv::Value& obj, const std::string& varName, IntT& v,
                           bool mandatory) {
  const Ptv::Value* var = obj.has(varName);
  if (!checkVar(objName, varName, var, mandatory)) {
    return PopulateResult_DoesNotExist;
  }
  if (!checkType(varName, *var, Ptv::Value::INT)) {
    return PopulateResult_WrongType;
  }
  v = (IntT)var->asInt();
  return PopulateResult_Ok;
}

template VS_EXPORT PopulateResult populateInt<int>(const std::string& objName, const Ptv::Value& obj,
                                                   const std::string& varName, int& v, bool mandatory);
template VS_EXPORT PopulateResult populateInt<int64_t>(const std::string& objName, const Ptv::Value& obj,
                                                       const std::string& varName, int64_t& v, bool mandatory);
template VS_EXPORT PopulateResult populateInt<unsigned int>(const std::string& objName, const Ptv::Value& obj,
                                                            const std::string& varName, unsigned int& v,
                                                            bool mandatory);
#if (__SIZEOF_SIZE_T__ != 4)
template VS_EXPORT PopulateResult populateInt<size_t>(const std::string& objName, const Ptv::Value& obj,
                                                      const std::string& varName, size_t& v, bool mandatory);
#endif

PopulateResult populateDouble(const std::string& objName, const Ptv::Value& obj, const std::string& varName, double& v,
                              bool mandatory) {
  const Ptv::Value* var = obj.has(varName);
  if (!checkVar(objName, varName, var, mandatory)) {
    return PopulateResult_DoesNotExist;
  }
  if (!checkType(varName, *var, Ptv::Value::DOUBLE)) {
    return PopulateResult_WrongType;
  }
  v = var->asDouble();
  return PopulateResult_Ok;
}

PopulateResult populateString(const std::string& objName, const Ptv::Value& obj, const std::string& varName,
                              std::string& v, bool mandatory) {
  const Ptv::Value* var = obj.has(varName);
  if (!checkVar(objName, varName, var, mandatory)) {
    return PopulateResult_DoesNotExist;
  }
  if (!checkType(varName, *var, Ptv::Value::STRING)) {
    return PopulateResult_WrongType;
  }
  v = var->asString();
  return PopulateResult_Ok;
}

PopulateResult populateColor(const std::string& objName, const Ptv::Value& obj, const std::string& varName, uint32_t& v,
                             bool mandatory) {
  const Ptv::Value* var = obj.has(varName);
  if (!checkVar(objName, varName, var, mandatory)) {
    return PopulateResult_DoesNotExist;
  }
  if (!checkType(varName, *var, Ptv::Value::STRING)) {
    return PopulateResult_WrongType;
  }
  if (Util::parseHtmlColor(var->asString(), v)) {
    return PopulateResult_Ok;
  }
  Logger::get(Logger::Error) << "Wrong type for " << varName << ", expected an RRGGBB of RRGGBBAA color, got '"
                             << var->asString() << "'." << std::endl;
  return PopulateResult_WrongType;
}

PopulateResult VS_EXPORT populateIntList(const std::string& objName, const Ptv::Value& obj, const std::string& varName,
                                         std::vector<int64_t>& v, bool mandatory) {
  const Ptv::Value* var = obj.has(varName);
  if (!checkVar(objName, varName, var, mandatory)) {
    return PopulateResult_DoesNotExist;
  }
  if (!checkType(varName, *var, Ptv::Value::LIST)) {
    return PopulateResult_WrongType;
  }

  const std::vector<Ptv::Value*>& listValues = var->asList();
  for (Ptv::Value* value : listValues) {
    if (!checkType(varName, *value, Ptv::Value::INT)) {
      return PopulateResult_WrongType;
    }
    v.push_back(value->asInt());
  }
  return PopulateResult_Ok;
}

}  // namespace Parse
}  // namespace VideoStitch
