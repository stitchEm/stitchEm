// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef OBJECT_HPP_
#define OBJECT_HPP_

#include "config.hpp"

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
 *      A simple such implementation is provided for applications that require a immutable project in
 * Core::ImmutableProject.
 */
namespace Ptv {

class Value;

/**
 * @brief An interface for objects that can be serialized to PTV.
 */
class VS_EXPORT Object {
 public:
  virtual ~Object();
  /**
   * Serialize to a Ptv::Value.
   * @return A Ptv::Value that represents the object. Must be deleted after use.
   */
  virtual Value* serialize() const = 0;
};
}  // namespace Ptv
}  // namespace VideoStitch

#endif
