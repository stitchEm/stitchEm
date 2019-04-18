// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef PTV_HPP_
#define PTV_HPP_

#include "config.hpp"

#include <iosfwd>
#include <utility>
#include <vector>

namespace VideoStitch {

/**
 * The PTV format is built on several layers:
 *  1 - The textual layer that uses Json to represent a VideoStitch project as a file.
 *  2 - Ptv::Value provides a very simple layer of abtraction that provides no semantics except typing.
 *      It's just there to easily manipulate objects of the textual layer.
 *      Ptv::Values can be parsed from Json text using Ptv::Parser and written to text using Ptv::Value::printJson().
 *  3 - Ptv::Objects provide a semantic layer on top of Ptv::Value.
 *      Ptv::Objects can be built from Ptv::Values using static factory methods for each object.
 *      Ptv::Objects can be serialized to Ptv::Values using Ptv::Object::serialize();
 *  4 - Applications may build their own semantic layer by writing a Ptv::Object.
 */
namespace Ptv {
/**
 * @brief A parsed PTV value.
 */
class VS_EXPORT Value {
 public:
  /**
   * Const Key/Value pair.
   */
  typedef std::pair<const std::string*, const Value*> ConstKV;
  /**
   * Key/Value pair.
   */
  typedef std::pair<const std::string*, Value*> KV;

  /**
   * \enum Type
   * \brief Defines the possible Variable types.
   */
  enum Type { NIL, BOOL, INT, DOUBLE, STRING, OBJECT, LIST };

  virtual ~Value() {}

  /**
   * Creates an empty OBJECT value.
   */
  static Value* emptyObject();

  /**
   * Creates an OBJECT from bool value.
   */
  static Value* boolObject(const bool& value);

  /**
   * Creates an OBJECT from int value.
   */
  static Value* intObject(const int64_t& value);

  /**
   * Creates an OBJECT from double value.
   */
  static Value* doubleObject(const double& value);

  /**
   * Creates an OBJECT from string value.
   */
  static Value* stringObject(const std::string& value);

  /**
   * Clones a value.
   */
  virtual Value* clone() const = 0;

  /**
   * Returns the type of the value.
   */
  virtual Type getType() const = 0;

  /**
   * Returns true this value is implicitly convertible to the given type.
   */
  bool isConvertibleTo(Type t) const;

  /**
   * Return the boolean value of this value, or false.
   */
  virtual bool asBool() const = 0;

  /**
   * Return the integer value of this value, or 0.
   */
  virtual int64_t asInt() const = 0;

  /**
   * Return the floating point value of this value, or 0.0.
   */
  virtual double asDouble() const = 0;

  /**
   * Return the string value of this value, or an empty string.
   */
  virtual const std::string& asString() const = 0;

  /**
   * Return the list value of this value, or an empty vector.
   */
  virtual const std::vector<Value*>& asList() const = 0;

  /**
   * Make this value into a NIL.
   * If the value is already a NIL.
   */
  virtual void asNil() = 0;

  /**
   * Make this value into a bool and return a mutable reference to its value.
   * If the value is already a bool, this is a noop in itself.
   */
  virtual bool& asBool() = 0;

  /**
   * Make this value into an int and return a mutable reference to its value.
   * If the value is already an int, this is a noop in itself.
   */
  virtual int64_t& asInt() = 0;

  /**
   * Make this value into a double and return a mutable reference to its value.
   * If the value is already a double, this is a noop in itself.
   */
  virtual double& asDouble() = 0;

  /**
   * Make this value into a string and return a mutable reference to its value.
   * If the value is already a string, this is a noop in itself.
   */
  virtual std::string& asString() = 0;

  /**
   * Make this value into a list and return itself.
   * If the value is already a list, this is a noop in itself.
   */
  virtual std::vector<Value*>& asList() = 0;

  /**
   * Make this value into an object and return itself.
   * If the value is already an object, this is a noop in itself.
   */
  virtual Value& asObject() = 0;

  /**
   * Returns the member with the given name.
   * @param name The name of the value to retrieve.
   * @return The value with this name, or NULL if this value does not exist.
   * @note Only OBJECT typed values support this.
   */
  virtual const Value* has(const std::string& name) const = 0;

  /**
   * Returns the member with the given name.
   * If no member with this name exist, a new member is created, with a default type of NIL.
   * @param name The name of the value to add.
   * @return The value with this name, or NULL if this value is not an OBJECT. Remains the property of the value. Call
   * remove() to get ownership.
   * @note Only OBJECT typed values support this.
   */
  virtual Value* get(const std::string& name) = 0;

  /**
   * Adds a member to an OBJECT value. Does nothing if the value is not an Object.
   * @param key Name of the member to add.
   * @param v The value to add. Ownership is transferred to the value.
   * @return The previous value with the given key, NULL if there was none, or @a v if not an object. Must be deleted.
   */
  virtual Value* push(const std::string& key, Value* v) = 0;

  /**
   * Remove and returns the member with the given name.
   * If no member with this name exist, return NULL.
   * @param name The name of the value to remove.
   * @note Only OBJECT typed values support this.
   */
  virtual Value* remove(const std::string& name) = 0;

  /**
   * Returns the number of values in an OBJECT value.
   * @return 0 if this value is not an OBJECT.
   */
  virtual int size() const = 0;

  /**
   * Returns the i-th value in an OBJECT value.
   * @param i index of the object to get.
   * @return The (key, value) pair. Ownership is retained.
   */
  virtual ConstKV get(int i) const = 0;

  /**
   * Returns the i-th value in an OBJECT value.
   * @param i index of the object to get.
   * @return The (key, value) pair. Ownership is retained.
   */
  virtual KV get(int i) = 0;

  /**
   * Print the variable as Json.
   * @param os Output stream. This must be a character stream
   * (opened with the default mode).
   * @param indent How much to indent. Usually left as default.
   */
  virtual void printJson(std::ostream& os, int indent = 0) const = 0;

  /**
   * @brief getJsonStr
   * @return Object serialized as a Json std::string.
   */
  virtual std::string getJsonStr() const = 0;

  /**
   * Print the variable as UBJson.
   * @param os Output stream. This must be a binary stream
   * (opened with the 'binary' mode).
   */
  virtual void printUBJson(std::ostream& os) const = 0;

  /**
   * Tests equality.
   *  - Types must be the same, no conversion,
   *  - Object members can be in any order,
   *  - List elements must be in the same order.
   * @param other Other value to test against.
   */
  bool operator==(const Value& other) const;

  /**
   * Tests inequality.
   * @param other Other value to test against.
   */
  bool operator!=(const Value& other) const;

  /**
   * Traverse a Value recursively and set the defaults from another Value.
   * We will only set values that are primitive types and lists (i.e. we will never copy a whole object that is in
   * defaults but not in *this). Lists of defaults are applied element-wise. If there are less elements in the default
   * than in *this, the last element is repeated. This makes it possible to apply the same default to all elements of a
   * list by specifying only one element. Empty lists are ignored (don't use them). *this before call defaults *this
   * after call { "a": { "b": 2 } }            { "a": { "b": 1 } }                 { "a": { "b": 2 } } { "a": {        }
   * }            { "a": { "b": 1 } }                 { "a": { "b": 1 } } {                 }            { "a": { "b": 1
   * } }                 {                 } {                 }            { "l": [{ "c": 3 }] }               { "l":
   * []         } { "l": [{},{}] }               { "l": [{ "c": 3 }] }               { "l": [ { "c": 3 }, { "c": 3 } ] }
   *  { "l": [{},{}] }               { "l": [{ "c": 3 }, { "d": 2 }] }   { "l": [ { "c": 3 }, { "d": 2 } ] }
   *  { "l": [{},{}] }               { "l": [] }                         { "l": [{},{}] }
   */
  void populateWithPrimitiveDefaults(const Value& defaults);

  /**
   * Traverse a Value recursively and set the defaults from another Value.
   * If object types are not aligned, we skip the recursion subtree and output a debug message.
   */
  // void populateWithDefaults(const Value& defaults);

  /**
   * Returns the name of a given type. Used for error messages and debug.
   * Don't rely on these values for anything else than display, and don't expect them to be immutable.
   * @param type The type of which to get the name.
   */
  static const char* getTypeName(Type type);
};
}  // namespace Ptv
}  // namespace VideoStitch

#endif
