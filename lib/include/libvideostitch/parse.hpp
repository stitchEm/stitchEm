// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef PARSE_HPP_
#define PARSE_HPP_

#include "ptv.hpp"
#include "status.hpp"

#include <string>

namespace VideoStitch {
namespace Ptv {

/**
 * @brief A parser class for PTV files.
 *
 * NOT thread safe.
 */
class VS_EXPORT Parser {
 public:
  /**
   * Creates a new parser.
   * @return NULL on error.
   */
  static Potential<Parser> create();
  virtual ~Parser() {}

  /**
   * Run the parser.
   * @param fileName the name of the file to parse.
   * @return false on error.
   */
  virtual bool parse(const std::string& fileName) = 0;

  /**
   * Run the parser.
   * @param data the data parse.
   * @return false on error.
   */
  virtual bool parseData(const std::string& data) = 0;

  /**
   * Returns the parsing error messages.
   */
  virtual std::string getErrorMessage() const = 0;

  /**
   * Returns the root parsed value.
   */
  virtual const Ptv::Value& getRoot() const = 0;
};
}  // namespace Ptv

namespace Parse {

/**
 * Compute the one-way diff between left and right. Note that this is non-commutative.
 * Lists that have the same size are diffed element-wise. Else the whole right-side list is rewritten.
 * There is no way to remove stuff on the right side, you can only add or replace.
 * For example:
 * left = {
 *  "a": "0",
 *  "b": 1,
 *  "c": { "d": 1 , "e": 2},
 *  "l1": [1, 2, 3],
 *  "l2": [4, 5, 6],
 *  "l3": [{"x": 1, "y": 2}, {"x": 4, "y": 5}]
 * }
 *
 * right = {
 *  "b": 1,
 *  "c": { "d": 3},
 *  "l1": [1, 2, 3],
 *  "l2": [8],
 *  "l3": [{"x": 1, "y": 4}, {"x": 4, "y": 5}],
 * }
 *
 * Will yield:
 * {
 *  "c": { "d": 3 },
 *  "l2": [8],
 *  "l3": [{"y": 4}, {}],
 * }
 *
 * @param left base value. Must be an object.
 * @param right diffed value; Must be an object.
 * @return The diff. Ownership is passed to the caller.
 */
// Ptv::Value* diff(const Ptv::Value& left, const Ptv::Value& right);

/**
 * Checks that a Ptv::Value can represent a typed value. Else, prints an error and returns false.
 * @param varName The name used to refer to the variable in the error message.
 * @param value The value whose type to check.
 * @param expectedType The expected type.
 */
bool VS_EXPORT checkType(const std::string& varName, const Ptv::Value& value, Ptv::Value::Type expectedType);

/**
 * Checks that a Ptv::Value can has a variable named varName. Else, prints an error and returns false.
 * @param objName The name used to refer to the value in the error message.
 * @param varName The variable to look for.
 * @param var The object to search into.
 * @param mandatory Whether we should print an error message.
 */
bool VS_EXPORT checkVar(const std::string& objName, const std::string& varName, const Ptv::Value* var, bool mandatory);

/**
 * Return states for the functions below. They indicate respectively success, missing variable, or incorrect type.
 */
enum class PopulateResult { OK, DoesNotExist, WrongType };
#define PopulateResult_Ok PopulateResult::OK
#define PopulateResult_DoesNotExist PopulateResult::DoesNotExist
#define PopulateResult_WrongType PopulateResult::WrongType

/**
 * Populates a boolean from a named Ptv::Value member. If the member type is not convertible to a boolean, prints an
 * error message and returns WrongType. If the member does not exist, return DoesNotExist.
 * @param objName The name used to refer to the object in the error message.
 * @param obj The object. Must be of type Ptv::Value::OBJECT.
 * @param varName The name of the member to fetch.
 * @param v The variable to populate.
 * @param mandatory If true, will display an error message if the member does not exist.
 */
PopulateResult VS_EXPORT populateBool(const std::string& objName, const Ptv::Value& obj, const std::string& varName,
                                      bool& v, bool mandatory);

/**
 * Populates an int from a named Ptv::Value member. If the member type is not convertible to an int, prints an error
 * message and returns WrongType. If the member does not exist, return DoesNotExist.
 * @param objName The name used to refer to the object in the error message.
 * @param obj The object. Must be of type Ptv::Value::OBJECT.
 * @param varName The name of the member to fetch.
 * @param v The variable to populate.
 * @param mandatory If true, will display an error message if the member does not exist.
 */
template <typename IntT> /*int, int64_t*/
PopulateResult VS_EXPORT populateInt(const std::string& objName, const Ptv::Value& obj, const std::string& varName,
                                     IntT& v, bool mandatory);

/**
 * Populates a double from a named Ptv::Value member. If the member type is not convertible to a double, prints an error
 * message and returns WrongType. If the member does not exist, return DoesNotExist.
 * @param objName The name used to refer to the object in the error message.
 * @param obj The object. Must be of type Ptv::Value::OBJECT.
 * @param varName The name of the member to fetch.
 * @param v The variable to populate.
 * @param mandatory If true, will display an error message if the member does not exist.
 */
PopulateResult VS_EXPORT populateDouble(const std::string& objName, const Ptv::Value& obj, const std::string& varName,
                                        double& v, bool mandatory);

/**
 * Populates a string from a named Ptv::Value member. If the member type is not convertible to a string, prints an error
 * message and returns WrongType. If the member does not exist, return DoesNotExist.
 * @param objName The name used to refer to the object in the error message.
 * @param obj The object. Must be of type Ptv::Value::OBJECT.
 * @param varName The name of the member to fetch.
 * @param v The variable to populate.
 * @param mandatory If true, will display an error message if the member does not exist.
 */
PopulateResult VS_EXPORT populateString(const std::string& objName, const Ptv::Value& obj, const std::string& varName,
                                        std::string& v, bool mandatory);

/**
 * Populates a color from a named Ptv::Value member. If the member type is not convertible to a color, prints an error
 * message and returns WrongType. If the member does not exist, return DoesNotExist.
 * @param objName The name used to refer to the object in the error message.
 * @param obj The object. Must be of type Ptv::Value::OBJECT.
 * @param varName The name of the member to fetch.
 * @param v The variable to populate.
 * @param mandatory If true, will display an error message if the member does not exist.
 * @note a color is a string of the form "00ff00" (rrggbb) or "00ff00ff" (rrggbbaa). The result is of form ABGR (8 bits
 * per component).
 */
PopulateResult VS_EXPORT populateColor(const std::string& objName, const Ptv::Value& obj, const std::string& varName,
                                       uint32_t& v, bool mandatory);

/**
 * Populates a list of integers from a named Ptv::Value member. If the member type is not convertible to a list of ints,
 * prints an error message and returns WrongType. If the member does not exist, return DoesNotExist.
 * @param objName The name used to refer to the object in the error message.
 * @param obj The object. Must be of type Ptv::Value::OBJECT.
 * @param varName The name of the member to fetch.
 * @param v The variable to populate.
 * @param mandatory If true, will display an error message if the member does not exist.
 */
PopulateResult VS_EXPORT populateIntList(const std::string& objName, const Ptv::Value& obj, const std::string& varName,
                                         std::vector<int64_t>& v, bool mandatory);

}  // namespace Parse

}  // namespace VideoStitch

#endif
