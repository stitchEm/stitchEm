// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "json.hpp"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <iostream>

namespace VideoStitch {
namespace Parse {

DataInputStream::DataInputStream(const std::string& data) : data(data), pos(0) {}

int DataInputStream::get() {
  if (pos < data.size()) {
    return data[pos++];
  } else {
    return EOF;
  }
}

int DataInputStream::peek() {
  if (pos < data.size()) {
    return data[pos];
  } else {
    return EOF;
  }
}

DataInputStream& DataInputStream::read(char* s, size_t n) {
  if (pos + n <= data.size()) {
    std::memcpy(s, data.data() + pos, n);
    pos += n;
  } else {
    std::memcpy(s, data.data() + pos, n);
    pos = data.size();
  }
  return *this;
}

bool DataInputStream::fail() const { return pos >= data.size(); }

namespace {
#if __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error TODO
#endif

/**
 * Prints the given int value on the smallest number of bytes, with type marker.
 * @param os output sink
 * @param value value
 */
void printUBJsonInt(std::ostream& os, const int64_t value) {
  if (-128ll <= value && value <= 127ll) {
    os << 'i' << (char)value;
  } else if (-32768ll <= value && value <= 32767ll) {
    os << 'I' << (char)(((uint64_t)value >> 8) & 0xffull) << (char)((value)&0xffull);
  } else if (-2147483648ll <= value && value <= 2147483647ll) {
    os << 'l' << (char)(((uint64_t)value >> 24) & 0xffull) << (char)(((uint64_t)value >> 16) & 0xffull)
       << (char)(((uint64_t)value >> 8) & 0xffull) << (char)(value & 0xffull);
  } else {
    os << 'L' << (char)(((uint64_t)value >> 56) & 0xffull) << (char)(((uint64_t)value >> 48) & 0xffull)
       << (char)(((uint64_t)value >> 40) & 0xffull) << (char)(((uint64_t)value >> 32) & 0xffull)
       << (char)(((uint64_t)value >> 24) & 0xffull) << (char)(((uint64_t)value >> 16) & 0xffull)
       << (char)(((uint64_t)value >> 8) & 0xffull) << (char)(value & 0xffull);
  }
}

/**
 * Writes a double value as UBJson, without leading string marker 'D'.
 * We assume a ieee754 compliant compiler.
 * @param os output sink
 * @param value value
 */
void printUBJsonDouble(std::ostream& os, double value) {
  union Bits {
    double dValue;
    uint64_t iValue;
  };
  Bits bits;
  bits.dValue = value;
  os << (char)((bits.iValue >> 56) & 0xffull) << (char)((bits.iValue >> 48) & 0xffull)
     << (char)((bits.iValue >> 40) & 0xffull) << (char)((bits.iValue >> 32) & 0xffull)
     << (char)((bits.iValue >> 24) & 0xffull) << (char)((bits.iValue >> 16) & 0xffull)
     << (char)((bits.iValue >> 8) & 0xffull) << (char)((bits.iValue) & 0xffull);
}

/**
 * Prints the given string value, without leading string marker 'S'.
 * @param os output sink
 * @param value value
 */
void printUBJsonString(std::ostream& os, const std::string& value) {
  printUBJsonInt(os, (int64_t)value.size());
  os << value;
}
}  // namespace

void JsonValue::printUBJson(std::ostream& os) const {
  switch (type) {
    case Ptv::Value::NIL:
      os << 'Z';
      break;
    case Ptv::Value::BOOL:
      os << (boolValue ? 'T' : 'F');
      break;
    case Ptv::Value::INT:
      printUBJsonInt(os, intValue);
      break;
    case Ptv::Value::DOUBLE:
      os << 'D';
      printUBJsonDouble(os, doubleValue);
      break;
    case Ptv::Value::STRING:
      os << 'S';
      printUBJsonString(os, stringValue);
      break;
    case Ptv::Value::LIST:
      os << '[';
      for (size_t i = 0; i < listValue.size(); ++i) {
        listValue[i]->printUBJson(os);
      }
      os << ']';
      break;
    case Ptv::Value::OBJECT:
      os << '{';
      for (int i = 0; i < content.size(); ++i) {
        std::pair<const std::string*, const Ptv::Value*> p = content.get(i);
        printUBJsonString(os, *p.first);
        p.second->printUBJson(os);
      }
      os << '}';
      break;
  }
}

namespace {

/**
 * Reads an int64_t from an UBJson stream. On error, the stream state is undefined.
 * @param token the int type token.
 * @param input Input stream.
 * @param value Result.
 * @return false on error.
 */
template <class StreamT>
bool readUBJsonInt(const int token, StreamT& input, int64_t& value) {
  unsigned char buffer[] = {0, 0, 0, 0, 0, 0, 0, 0};
  switch (token) {
    case 'i':
      input.read((char*)buffer, 1);
      if (input.fail()) {
        return false;
      }
      value = (char)buffer[0];
      return true;
    case 'U':
      input.read((char*)buffer, 1);
      if (input.fail()) {
        return false;
      }
      value = buffer[0];
      return true;
    case 'I':
      input.read((char*)buffer, 2);
      if (input.fail()) {
        return false;
      }
      value = (int16_t)(((uint16_t)buffer[0] << 8) | ((uint16_t)buffer[1]));
      return true;
    case 'l':
      input.read((char*)buffer, 4);
      if (input.fail()) {
        return false;
      }
      value = (int32_t)(((uint32_t)buffer[0] << 24) | ((uint32_t)buffer[1] << 16) | ((uint32_t)buffer[2] << 8) |
                        ((uint32_t)buffer[3]));
      return true;
    case 'L':
      input.read((char*)buffer, 8);
      if (input.fail()) {
        return false;
      }
      value = (int64_t)(((uint64_t)buffer[0] << 56) | ((uint64_t)buffer[1] << 48) | ((uint64_t)buffer[2] << 40) |
                        ((uint64_t)buffer[3] << 32) | ((uint64_t)buffer[4] << 24) | ((uint64_t)buffer[5] << 16) |
                        ((uint64_t)buffer[6] << 8) | ((uint64_t)buffer[7]));
      return true;
  }
  value = 0;
  return false;
}

/**
 * Reads a string from an UBJson stream (starting at str len). On error, the stream state is undefined.
 * @param input Input stream.
 * @param value Result.
 * @return false on error.
 */
template <class StreamT>
bool readUBJsonString(StreamT& input, std::string& str) {
  str.clear();
  const int token = input.get();
  int64_t sLen = 0;
  if (!readUBJsonInt(token, input, sLen) || sLen < 0) {
    return false;
  }
  for (int i = 0; i < sLen; ++i) {
    str.push_back((char)input.get());
  }
  if (input.fail()) {
    return false;
  }
  return true;
}
}  // namespace

template <class StreamT>
PotentialJsonValue JsonValue::parseUBJson(StreamT& input) {
  if (!(input.get() == '{' && (input.peek() == 'i' || input.peek() == 'U' || input.peek() == 'I' ||
                               input.peek() == 'l' || input.peek() == 'L'))) {
    return ParseStatus::fromCode<ParseStatus::StatusCode::NotUBJsonFormat>();
  }
  PotentialJsonValue result(new JsonValue());
  // The objects being populated. If we are expecting a value, the object will be a NIL.
  // If we are expecting a list item, it will be a LIST, If we are expecting a field name, it will be an OBJECT.
  // Anything else is invalid. Elements are not owned.
  std::vector<JsonValue*> valueStack;
  valueStack.push_back(result.object());
  for (;;) {
    if (input.peek() == EOF) {
      if (!valueStack.empty()) {
        return ParseStatus::fromCode<ParseStatus::StatusCode::InvalidUBJson>();
      }
      return result;
    }
    if (valueStack.empty()) {
      return ParseStatus::fromCode<ParseStatus::StatusCode::InvalidUBJson>();
    }
    if (valueStack.back()->getType() == OBJECT) {
      if (input.peek() == '}') {
        input.get();
        // End the object.
        valueStack.pop_back();
        continue;
      }
      // Expect a field name.
      std::string fieldName;
      if (!readUBJsonString(input, fieldName)) {
        return ParseStatus::fromCode<ParseStatus::StatusCode::InvalidUBJson>();
      }
      JsonValue* newValue = new JsonValue((void*)NULL);
      delete valueStack.back()->push(fieldName, newValue);
      valueStack.push_back(newValue);
    } else {
      const int token = input.get();
      JsonValue* value = NULL;
      if (valueStack.back()->getType() == LIST) {
        if (token == ']') {
          // End the list.
          valueStack.pop_back();
          continue;
        } else {
          // Add an element.
          value = new JsonValue((void*)NULL);
          valueStack.back()->asList().push_back(value);
        }
      } else {
        value = valueStack.back();
        assert(value->getType() == NIL);
        valueStack.pop_back();
      }

      switch (token) {
        case 'Z':
          value->asNil();
          break;
        case 'T':
          value->asBool() = true;
          break;
        case 'F':
          value->asBool() = false;
          break;
        case 'i':
        case 'U':
        case 'I':
        case 'l':
        case 'L':
          if (!readUBJsonInt(token, input, value->asInt())) {
            return ParseStatus::fromCode<ParseStatus::StatusCode::InvalidUBJson>();
          }
          break;
        case 'D': {
          union Bits {
            double dValue;
            int64_t iValue;
          };
          Bits bits;
          if (!readUBJsonInt('L', input, bits.iValue)) {
            return ParseStatus::fromCode<ParseStatus::StatusCode::InvalidUBJson>();
          }
          value->asDouble() = bits.dValue;
          break;
        }
        case 'S':
          if (!readUBJsonString(input, value->asString())) {
            return ParseStatus::fromCode<ParseStatus::StatusCode::InvalidUBJson>();
          }
          break;
        case '[':
          value->asList();
          valueStack.push_back(value);
          break;
        case '{':
          value->asObject();
          valueStack.push_back(value);
          break;
        default:
          return ParseStatus::fromCode<ParseStatus::StatusCode::InvalidUBJson>();
      }
    }
  }
}

// Explicit instantiations.
template PotentialJsonValue JsonValue::parseUBJson(std::istream& input);
template PotentialJsonValue JsonValue::parseUBJson(DataInputStream& input);

}  // namespace Parse
}  // namespace VideoStitch
