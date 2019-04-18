// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "libvideostitch/ptv.hpp"

#include <map>
#include <string>
#include <stdint.h>

namespace VideoStitch {
namespace Input {
/**
 * A helper for parsing procedural input specs.
 * A spec looks like "procedural:aname(some=3,options=3.2)".
 * We provide no semantics or typing, just key/values. Keys and values may not contain '=' or ','.
 */
class ProceduralInputSpec {
 public:
  /**
   * Parse an spec. The spec can be anything, use isProcedural() to determine if the spec was a procedural reader.
   */
  explicit ProceduralInputSpec(const std::string& spec);

  /**
   * Returns true if the spec is procedural.
   */
  bool isProcedural() const;

  /**
   * Return the name of the procedural input.
   */
  const std::string& getName() const;

  /**
   * Returns the value of the given option. If no option with this name exists, returns NULL.
   * @param option Option name.
   */
  const std::string* getOption(const std::string& option) const;

  /**
   * Tries to parse the given option as an int. Returns false on failure.
   * @param option Option name.
   * @param v Will be filled with the int value on success.
   */
  bool getIntOption(const std::string& option, int& v) const;

  /**
   * Tries to parse the given option as a double. Returns false on failure.
   * @param option Option name.
   * @param v Will be filled with the double value on success.
   */
  bool getDoubleOption(const std::string& option, double& v) const;

  /**
   * Tries to parse the given option as a color (RRGGBB, hex). Returns false on failure.
   * @param option Option name.
   * @param v Will be filled with the packed RGBA8888 color on success.
   */
  bool getColorOption(const std::string& option, uint32_t& v) const;

  /**
   * Returns the options as a Ptv config. Tries to type the options as best as it can.
   * @returns the config. Must be freed by the caller.
   */
  Ptv::Value* getPtvConfig() const;

 private:
  typedef std::map<std::string, std::string> MapT;
  std::string name;
  MapT options;
};
}  // namespace Input
}  // namespace VideoStitch
