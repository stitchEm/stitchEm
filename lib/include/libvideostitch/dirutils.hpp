// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "logging.hpp"
#include "status.hpp"

#include <string>

namespace VideoStitch {
namespace Util {
/**
 * Checks if an absolute path is a valid existing path
 * @param absolutePath full path to check
 * @returns true if the absolute path exists, otherwise returns false.
 */
VS_EXPORT bool directoryExists(const std::string& absolutePath);

/**
 * Directory location where user-specific non-essential (cached) data, shared across applications, should be written.
 * @param companyName name of the subfolder
 * Returned value can be an empty string
 * @returns a string pointing to the generic cache location.
 */
VS_EXPORT PotentialValue<std::string> getGenericCacheLocation(const std::string& companyName);

/**
 * Directory location where user-specific non-essential (cached) data shared across VideoStitch applications can be
 * stored. Returned value can be an empty string
 * @returns a string pointing to the generic cache location.
 */
VS_EXPORT PotentialValue<std::string> getVSCacheLocation();

/**
 * Directory location where persistent data shared across applications can be stored.
 * @param companyName name of the subfolder
 * Returned value can be an empty string
 * @returns a string pointing to the generic cache location.
 */
VS_EXPORT PotentialValue<std::string> getGenericDataLocation(const std::string& companyName);

/**
 * Directory location where persistent data shared across VideoStitch applications can be stored.
 * Returned value can be an empty string
 * @returns a string pointing to the generic cache location.
 */
VS_EXPORT PotentialValue<std::string> getVSDataLocation();
}  // namespace Util
}  // namespace VideoStitch
