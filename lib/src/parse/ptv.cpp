// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "json.hpp"

#include "libvideostitch/logging.hpp"
#include "libvideostitch/ptv.hpp"
#include "libvideostitch/object.hpp"

namespace VideoStitch {
namespace Ptv {

Value* Value::emptyObject() { return new Parse::JsonValue(); }

Value* Value::boolObject(const bool& value) { return new Parse::JsonValue(value); }

Value* Value::intObject(const int64_t& value) { return new Parse::JsonValue(value); }

Value* Value::doubleObject(const double& value) { return new Parse::JsonValue(value); }

Value* Value::stringObject(const std::string& value) { return new Parse::JsonValue(value); }

const char* Value::getTypeName(Type type) {
  switch (type) {
    case NIL:
      return "null";
    case BOOL:
      return "bool";
    case INT:
      return "int";
    case DOUBLE:
      return "double";
    case STRING:
      return "string";
    case OBJECT:
      return "object";
    case LIST:
      return "list";
  }
  return NULL;
}

bool Value::isConvertibleTo(Type t) const {
  Type type = getType();
  // Int is convertible to double and bool.
  if (type == INT && (t == BOOL || t == DOUBLE)) {
    return true;
  }
  // And a type is always convertible to itself.
  return type == t;
}

Object::~Object() {}

void Value::populateWithPrimitiveDefaults(const Value& defaults) {
  if (defaults.getType() != getType()) {
    Logger::get(Logger::Debug) << "Inconsistent type for default value: Default is " << getTypeName(defaults.getType())
                               << ", got " << getTypeName(getType()) << std::endl;
    return;
  }
  switch (getType()) {
    case NIL:
    case BOOL:
    case INT:
    case DOUBLE:
    case STRING:
      // We have a value, and we're given a default value. Do nothing.
      break;
    case OBJECT:
      // Iterate over defaults and look into *this, else we risk infinite loops.
      for (int i = 0; i < defaults.size(); ++i) {
        std::pair<const std::string*, const Value*> p = defaults.get(i);
        const Value* here = has(*p.first);
        if (here) {
          // We already have the member, recurse.
          get(*p.first)->populateWithPrimitiveDefaults(*p.second);
        } else {
          switch (p.second->getType()) {
            case NIL:
              get(*p.first);
              return;
            case BOOL:
              get(*p.first)->asBool() = p.second->asBool();
              return;
            case INT:
              get(*p.first)->asInt() = p.second->asInt();
              return;
            case DOUBLE:
              get(*p.first)->asDouble() = p.second->asDouble();
              return;
            case STRING:
              get(*p.first)->asString() = p.second->asString();
              return;
            case OBJECT:
              // Do nothing, we want to only insert primitive values.
              return;
            case LIST:
              // Insert an empty list.
              get(*p.first)->asList();
              return;
          }
        }
      }
      break;
    case LIST:
      const std::vector<Value*>& defaultElems = defaults.asList();
      if (defaultElems.empty()) {
        return;
      }
      const std::vector<Value*>& elems = asList();
      for (size_t i = 0; i < elems.size(); ++i) {
        // The last list element is repeated indefinitely.
        size_t j = i < defaultElems.size() ? i : defaultElems.size() - 1;
        elems[i]->populateWithPrimitiveDefaults(*defaultElems[j]);
      }
      break;
  }
}

bool Value::operator==(const Value& other) const {
  if (other.getType() != getType()) {
    return false;
  }

  switch (getType()) {
    case NIL:
      return true;
    case BOOL:
      return asBool() == other.asBool();
    case INT:
      return asInt() == other.asInt();
    case DOUBLE:
      return asDouble() == other.asDouble();
    case STRING:
      return asString() == other.asString();
    case OBJECT:
      if (size() != other.size()) {
        return false;
      }
      for (int i = 0; i < size(); ++i) {
        std::pair<const std::string*, const Value*> p = get(i);
        const Value* elem = has(*p.first);
        if (!elem) {
          return false;
        } else if (!(*elem == *p.second)) {
          return false;
        }
      }
      for (int i = 0; i < other.size(); ++i) {
        std::pair<const std::string*, const Value*> p = other.get(i);
        const Value* elem = has(*p.first);
        if (!elem) {
          return false;
        } else if (!(*elem == *p.second)) {
          return false;
        }
      }
      return true;
    case LIST:
      const std::vector<Value*>& otherElems = other.asList();
      const std::vector<Value*>& elems = asList();
      if (otherElems.size() != elems.size()) {
        return false;
      }
      for (size_t i = 0; i < elems.size(); ++i) {
        if (!(*elems[i] == *otherElems[i])) {
          return false;
        }
      }
      return true;
  }
  return true;
}

bool Value::operator!=(const Value& other) const { return !(*this == other); }

namespace {
Ptv::Value* objDiff(const Ptv::Value& /*left*/, const Ptv::Value& /*right*/) {
  // TODO
  return NULL;
}
}  // namespace

Ptv::Value* diff(const Ptv::Value& left, const Ptv::Value& right) {
  if (right.getType() != left.getType()) {
    return right.clone();
  }
  switch (right.getType()) {
    case Ptv::Value::NIL:
    case Ptv::Value::BOOL:
    case Ptv::Value::INT:
    case Ptv::Value::DOUBLE:
    case Ptv::Value::STRING:
      // Diffing primitive types is useless.
      return right.clone();
    case Ptv::Value::OBJECT:
      return objDiff(left, right);
    case Ptv::Value::LIST:
      return NULL;  // TODO
  }
  return NULL;
}
}  // namespace Ptv
}  // namespace VideoStitch
