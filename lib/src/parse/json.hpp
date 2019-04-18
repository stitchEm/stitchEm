// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "libvideostitch/ptv.hpp"
#include "libvideostitch/status.hpp"

#include <algorithm>
#include <iostream>
#include <map>
#include <string>
#include <utility>

namespace VideoStitch {
namespace Parse {

class JsonValue;

enum class ParseStatusCode {
  // generic states
  Ok,
  ErrorWithStatus,
  // custom parse states
  InvalidUBJson,
  NotUBJsonFormat,
};

typedef Result<ParseStatusCode> ParseStatus;
typedef Potential<JsonValue, DefaultDeleter<Ptv::Value>, ParseStatus> PotentialJsonValue;

/**
 * A map pointer class that preserves insertion order.
 * Right now it's implemented as a vector of pairs, so lookup is via linear search. It's OK because we expect json
 * objects to be small. It takes ownership of the objects being inserted.
 * TODO: add an index of top if this no longer holds changes.
 * Insertion just pushes at the back and removal makes a hole. The vectors are compacted every now and then.
 */
template <typename T>
class OrderedMap {
 public:
  OrderedMap();
  ~OrderedMap();

  void reverse();

  int size() const;

  std::pair<const std::string*, const T*> get(int i) const;
  std::pair<const std::string*, T*> get(int i);

  /**
   * Inserts given key/value pair. Fails if the key already exists.
   * @param key The key.
   * @param v The value. On successful insertion, ownership is transferred to the OrderedMap. Cannot be NULL.
   * @return false if the object already exists, true if the insertion succeeds.
   */
  bool put(const std::string& key, T* v);

  /**
   * Fetches a value (const).
   * @param key The key.
   * @return NULL if the key does not exist.
   */
  const T* get(const std::string& key) const;

  /**
   * Fetches a value. Ownership is retained.
   * @param key The key.
   * @return NULL if the key does not exist.
   */
  T* get(const std::string& key);

  /**
   * Removes an entry. The returned object must be deleted.
   * @param key The key.
   * @return NULL if the key does not exist.
   */
  T* remove(const std::string& key);

  /**
   * Clears the container.
   */
  void clear();

  /**
   * Compact/reindex.
   */
  void compact();

 private:
  std::vector<std::string> keys;
  std::vector<T*> values;
  int numHoles;
};

/**
 * A stream that reads from a string.
 */
class DataInputStream {
 public:
  /**
   * Input shall remain alive four our lifetime.
   * @param input Input data.
   */
  explicit DataInputStream(const std::string& data);

  /**
   * This mimicks the std::istream interface.
   * @{
   */
  int get();
  int peek();
  DataInputStream& read(char* s, size_t n);
  bool fail() const;
  /**
   *@}
   */

 private:
  const std::string& data;
  size_t pos;
};

/**
 * @brief A Json Value.
 */
class JsonValue : public Ptv::Value {
 public:
  /**
   * Creates a NULL value.
   */
  explicit JsonValue(void* v);

  /**
   * Creates a BOOL value.
   */
  explicit JsonValue(bool v);

  /**
   * Creates an INT value.
   */
  explicit JsonValue(int64_t v);
  explicit JsonValue(int v);

  /**
   * Creates a LIST of INT values
   */
  explicit JsonValue(const std::vector<int64_t>& v);

  /**
   * Creates an DOUBLE value.
   */
  explicit JsonValue(double v);

  /**
   * Creates a STRING value.
   */
  explicit JsonValue(const std::string& v);
  explicit JsonValue(const char* v);

  /**
   * Creates a LIST value.
   * @param v List of values. Takes ownership of v.
   */
  explicit JsonValue(std::vector<Ptv::Value*>* v);

  /**
   * Parse from an UBJson stream.
   * @param input Input stream.
   * StreamT can be either std::istream or DataStream.
   */
  template <class StreamT>
  static PotentialJsonValue parseUBJson(StreamT& input);

  /**
   * Creates an empty OBJECT value.
   */
  JsonValue();

  ~JsonValue();

  /**
   * Reverses the order of values within an object.
   */
  void reverse();

  Ptv::Value* clone() const override;
  Type getType() const override;
  bool asBool() const override;
  int64_t asInt() const override;
  double asDouble() const override;
  const std::string& asString() const override;
  const std::vector<Ptv::Value*>& asList() const override;

  void reset(Ptv::Value::Type t);
  void resetIfNotType(Ptv::Value::Type t);

  void asNil() override;
  bool& asBool() override;
  int64_t& asInt() override;
  double& asDouble() override;
  std::string& asString() override;
  std::vector<Ptv::Value*>& asList() override;
  Ptv::Value& asObject() override;

  void clear();

  const Ptv::Value* has(const std::string& key) const override;

  Ptv::Value* get(const std::string& key) override;

  Ptv::Value* remove(const std::string& key) override;

  int size() const override;

  std::pair<const std::string*, const Ptv::Value*> get(int i) const override;
  std::pair<const std::string*, Ptv::Value*> get(int i) override;

  Ptv::Value* push(const std::string& key, Ptv::Value* v) override;

  void printJson(std::ostream& os, int indent = 0) const override;
  std::string getJsonStr() const override;
  void printUBJson(std::ostream& os) const override;

 private:
  void printJsonCLocale(std::ostream& os, int indent) const;

  Ptv::Value::Type type;
  bool boolValue;
  int64_t intValue;
  double doubleValue;
  std::string stringValue;
  std::vector<Ptv::Value*> listValue;
  OrderedMap<Value> content;
};
}  // namespace Parse
}  // namespace VideoStitch
