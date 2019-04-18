// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "json.hpp"

#include "util/strutils.hpp"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <sstream>
#include <map>
#include <utility>

namespace VideoStitch {
namespace Parse {

namespace {
class Indent {
 public:
  explicit Indent(int i) : indent(i) {}
  void print(std::ostream& os) const {
    for (int i = 0; i < indent; ++i) {
      os << ' ' << ' ';
    }
  }

 private:
  int indent;
};

std::ostream& operator<<(std::ostream& os, const Indent& i) {
  i.print(os);
  return os;
}
}  // namespace

template <typename T>
OrderedMap<T>::OrderedMap() : numHoles(0) {}

template <typename T>
OrderedMap<T>::~OrderedMap() {
  clear();
}

template <typename T>
void OrderedMap<T>::reverse() {
  std::reverse(keys.begin(), keys.end());
  std::reverse(values.begin(), values.end());
}

template <typename T>
int OrderedMap<T>::size() const {
  return (int)values.size() - numHoles;
}

template <typename T>
std::pair<const std::string*, const T*> OrderedMap<T>::get(int i) const {
  int id = 0;
  for (size_t j = 0; j < keys.size(); ++j) {
    if (values[j] != NULL) {
      if (id == i) {
        return std::pair<const std::string*, const T*>(&keys[j], values[j]);
      }
      ++id;
    }
  }
  return std::make_pair(static_cast<const std::string*>(NULL), static_cast<const T*>(NULL));
}

template <typename T>
std::pair<const std::string*, T*> OrderedMap<T>::get(int i) {
  int id = 0;
  for (size_t j = 0; j < keys.size(); ++j) {
    if (values[j] != NULL) {
      if (id == i) {
        return std::pair<const std::string*, T*>(&keys[j], values[j]);
      }
      ++id;
    }
  }
  return std::make_pair(static_cast<const std::string*>(NULL), static_cast<T*>(NULL));
}

template <typename T>
bool OrderedMap<T>::put(const std::string& key, T* v) {
  assert(v != NULL);
  assert(keys.size() == values.size());
  if (get(key) != NULL) {
    return false;
  }
  keys.push_back(key);
  values.push_back(v);
  return true;
}

template <typename T>
const T* OrderedMap<T>::get(const std::string& key) const {
  assert(keys.size() == values.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    if (values[i] != NULL && keys[i] == key) {
      return values[i];
    }
  }
  return NULL;
}

template <typename T>
T* OrderedMap<T>::get(const std::string& key) {
  assert(keys.size() == values.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    if (values[i] != NULL && keys[i] == key) {
      return values[i];
    }
  }
  return NULL;
}

template <typename T>
T* OrderedMap<T>::remove(const std::string& key) {
  assert(keys.size() == values.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    if (values[i] != NULL && keys[i] == key) {
      T* res = values[i];
      values[i] = NULL;
      keys[i].clear();
      ++numHoles;
      if (numHoles > (int)values.size() / 2) {
        compact();
      }
      return res;
    }
  }
  return NULL;
}

template <typename T>
void OrderedMap<T>::clear() {
  assert(keys.size() == values.size());
  for (typename std::vector<T*>::iterator it = values.begin(); it != values.end(); ++it) {
    delete *it;
  }
  keys.clear();
  values.clear();
}

template <typename T>
void OrderedMap<T>::compact() {
  assert(keys.size() == values.size());
  // Compact the vector.
  size_t k;
  for (k = 0; k < keys.size(); ++k) {
    if (values[k] == NULL) {
      break;
    }
  }
  if (k == keys.size()) {
    return;  // No hole.
  }
  for (size_t i = 0; i < keys.size(); ++i) {
    if (k < i && values[k] == NULL && values[i] != NULL) {
      std::swap(values[k], values[i]);
      std::swap(keys[k], keys[i]);
      for (; k < i; ++k) {
        if (values[k] == NULL) {
          break;
        }
      }
    }
  }
  while (!values.empty() && values.back() == NULL) {
    --numHoles;
    values.pop_back();
    keys.pop_back();
  }
  assert(numHoles == 0);
}

JsonValue::JsonValue(void*
#ifdef NDEBUG
                     /*v*/)
#else
                         v)
#endif
    : type(Ptv::Value::NIL), boolValue(false), intValue(0), doubleValue(0.0) {
  assert(v == NULL);
}

std::vector<Ptv::Value*>* intListToPtvList(const std::vector<int64_t>& vec) {
  std::vector<Ptv::Value*>* objects = new std::vector<Ptv::Value*>();
  for (int64_t value : vec) {
    auto ptv = Ptv::Value::emptyObject();
    ptv->asInt() = value;
    objects->push_back(ptv);
  }
  return objects;
}

JsonValue::JsonValue(bool v) : type(Ptv::Value::BOOL), boolValue(v), intValue(0), doubleValue(0.0) {}
JsonValue::JsonValue(int64_t v) : type(Ptv::Value::INT), boolValue(v != 0), intValue(v), doubleValue((double)v) {}
JsonValue::JsonValue(int v) : type(Ptv::Value::INT), boolValue(v != 0), intValue(v), doubleValue(v) {}
JsonValue::JsonValue(const std::vector<int64_t>& vec) : JsonValue(intListToPtvList(vec)) {}
JsonValue::JsonValue(double v) : type(Ptv::Value::DOUBLE), boolValue(false), intValue(0), doubleValue(v) {}
JsonValue::JsonValue(const std::string& v)
    : type(Ptv::Value::STRING), boolValue(false), intValue(0), doubleValue(0.0), stringValue(v) {}
JsonValue::JsonValue(const char* v)
    : type(Ptv::Value::STRING), boolValue(false), intValue(0), doubleValue(0.0), stringValue(v) {}
JsonValue::JsonValue(std::vector<Value*>* v)
    : type(Ptv::Value::LIST), boolValue(false), intValue(0), doubleValue(0.0), listValue(*v) {
  delete v;
}
JsonValue::JsonValue() : type(Ptv::Value::OBJECT), boolValue(false), intValue(0), doubleValue(0.0) {}

JsonValue::~JsonValue() { reset(Ptv::Value::NIL); }

void JsonValue::reverse() { content.reverse(); }

Ptv::Value* JsonValue::clone() const {
  switch (type) {
    case Ptv::Value::NIL:
      return new JsonValue((void*)NULL);
    case Ptv::Value::BOOL:
      return new JsonValue(boolValue);
    case Ptv::Value::INT:
      return new JsonValue(intValue);
    case Ptv::Value::DOUBLE:
      return new JsonValue(doubleValue);
    case Ptv::Value::STRING:
      return new JsonValue(stringValue);
    case Ptv::Value::LIST: {
      JsonValue* res = new JsonValue((void*)NULL);
      res->asList();
      for (std::vector<Ptv::Value*>::const_iterator it = listValue.begin(); it != listValue.end(); ++it) {
        res->listValue.push_back((*it)->clone());
      }
      return res;
    }
    case Ptv::Value::OBJECT: {
      JsonValue* res = new JsonValue();
      for (int i = 0; i < content.size(); ++i) {
        std::pair<const std::string*, const Ptv::Value*> p = content.get(i);
        res->content.put(*p.first, p.second->clone());
      }
      return res;
    }
  }
  assert(false);
  return NULL;
}

Ptv::Value::Type JsonValue::getType() const { return type; }

bool JsonValue::asBool() const {
  // Int is convertible to bool.
  if (type == Ptv::Value::INT) {
    return intValue != 0;
  }
  return boolValue;
}

int64_t JsonValue::asInt() const { return intValue; }

double JsonValue::asDouble() const {
  // Int is convertible to double.
  if (type == Ptv::Value::INT) {
    return (double)intValue;
  }
  return doubleValue;
}

const std::string& JsonValue::asString() const { return stringValue; }
const std::vector<Ptv::Value*>& JsonValue::asList() const { return listValue; }

void JsonValue::reset(Ptv::Value::Type t) {
  type = t;
  boolValue = false;
  intValue = 0;
  doubleValue = 0.0;
  for (std::vector<Ptv::Value*>::iterator it = listValue.begin(); it != listValue.end(); ++it) {
    delete *it;
  }
  listValue.clear();
  content.clear();
}

void JsonValue::resetIfNotType(Ptv::Value::Type t) {
  if (type != t) {
    reset(t);
  }
}

void JsonValue::asNil() { resetIfNotType(Ptv::Value::NIL); }

bool& JsonValue::asBool() {
  // Int is convertible to bool.
  if (type == Ptv::Value::INT) {
    type = Ptv::Value::BOOL;
    boolValue = (intValue != 0);
  } else {
    resetIfNotType(Ptv::Value::BOOL);
  }
  return boolValue;
}

int64_t& JsonValue::asInt() {
  resetIfNotType(Ptv::Value::INT);
  return intValue;
}

double& JsonValue::asDouble() {
  // Int is convertible to double.
  if (type == Ptv::Value::INT) {
    type = Ptv::Value::DOUBLE;
    doubleValue = (double)intValue;
  } else {
    resetIfNotType(Ptv::Value::DOUBLE);
  }
  return doubleValue;
}

std::string& JsonValue::asString() {
  resetIfNotType(Ptv::Value::STRING);
  return stringValue;
}
std::vector<Ptv::Value*>& JsonValue::asList() {
  resetIfNotType(Ptv::Value::LIST);
  return listValue;
}
Ptv::Value& JsonValue::asObject() {
  resetIfNotType(Ptv::Value::OBJECT);
  return *this;
}

const Ptv::Value* JsonValue::has(const std::string& key) const {
  if (type != Ptv::Value::OBJECT) {
    return NULL;
  }
  return content.get(key);
}

Ptv::Value* JsonValue::get(const std::string& key) {
  if (type != Ptv::Value::OBJECT) {
    return NULL;
  }
  Value* v = content.get(key);
  if (!v) {
    v = new JsonValue((void*)NULL);
    content.put(key, v);
  }
  return v;
}

Ptv::Value* JsonValue::remove(const std::string& key) {
  if (type != Ptv::Value::OBJECT) {
    return NULL;
  }
  return content.remove(key);
}

int JsonValue::size() const { return content.size(); }

std::pair<const std::string*, const Ptv::Value*> JsonValue::get(int i) const {
  if (type != Ptv::Value::OBJECT) {
    return std::make_pair(static_cast<const std::string*>(NULL), static_cast<const Value*>(NULL));
  }
  return content.get(i);
}

std::pair<const std::string*, Ptv::Value*> JsonValue::get(int i) {
  if (type != Ptv::Value::OBJECT) {
    return std::make_pair(static_cast<const std::string*>(NULL), static_cast<Value*>(NULL));
  }
  return content.get(i);
}

Ptv::Value* JsonValue::push(const std::string& key, Ptv::Value* v) {
  if (type != Ptv::Value::OBJECT) {
    return NULL;
  }
  Value* prev = content.remove(key);
  content.put(key, v);
  return prev;
}

void JsonValue::printJson(std::ostream& os, int indent) const {
  Util::UsingCLocaleOnStream usingCLocale(os);
  // VSA-7234: increase the precision of serialized double values
  auto precision = os.precision();
  os.precision(std::numeric_limits<double>::max_digits10);
  printJsonCLocale(os, indent);
  os.precision(precision);
}

std::string JsonValue::getJsonStr() const {
  std::ostringstream outputStream;
  printJson(outputStream);
  return outputStream.str();
}

void JsonValue::printJsonCLocale(std::ostream& os, int indent) const {
  switch (type) {
    case Ptv::Value::NIL:
      os << "null";
      break;
    case Ptv::Value::BOOL:
      os << (boolValue ? "true" : "false");
      break;
    case Ptv::Value::INT:
      os << intValue;
      break;
    case Ptv::Value::DOUBLE:
      os << doubleValue;
      break;
    case Ptv::Value::STRING:
      os << "\"" << Util::escapeStr(stringValue) << "\"";
      break;
    case Ptv::Value::LIST:
      os << "[";
      if (!listValue.empty()) {
        os << '\n' << Indent(indent + 1);
        listValue.front()->printJson(os, indent + 1);
        for (size_t i = 1; i < listValue.size(); ++i) {
          os << ",\n" << Indent(indent + 1);
          listValue[i]->printJson(os, indent + 1);
        }
        os << '\n' << Indent(indent);
      }
      os << "]";
      break;
    case Ptv::Value::OBJECT:
      os << "{";
      if (content.size() > 0) {
        std::pair<const std::string*, const Ptv::Value*> p = content.get(0);
        assert(p.first && p.second);
        os << '\n' << Indent(indent + 1) << "\"" << Util::escapeStr(*p.first) << "\" : ";
        p.second->printJson(os, indent + 1);
        for (int i = 1; i < content.size(); ++i) {
          std::pair<const std::string*, const Ptv::Value*> p = content.get(i);
          os << ", \n" << Indent(indent + 1) << "\"" << Util::escapeStr(*p.first) << "\" : ";
          p.second->printJson(os, indent + 1);
        }
        os << '\n' << Indent(indent);
      }
      os << "}";
      break;
  }
}

// Explicit instanciations.
template class OrderedMap<int>;
template class OrderedMap<Ptv::Value>;
}  // namespace Parse
}  // namespace VideoStitch
