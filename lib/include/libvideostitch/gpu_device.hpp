// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "config.hpp"
#include "status.hpp"
#include <libgpudiscovery/genericDeviceInfo.hpp>
#include <sstream>
#include <vector>

namespace VideoStitch {
namespace Core {

/**
 * A structure that stores the identifier of the GPU device that is used for stitching
 */
struct PanoDeviceDefinition {
  int device;  ///< Identifier of device that is used for stitching

  /**
   * Default constructor for PanoDeviceDefinition
   * @param device - device id. Default value = 0
   */
  explicit PanoDeviceDefinition(const int device = 0) { this->device = device; }

  /**
   * @param rhs rhs.
   */
  bool operator<(const PanoDeviceDefinition& rhs) const { return this->device > rhs.device; }
};

/**
 * A structure that stores the identifier of the GPU devices that are used for stereo stitching
 */
struct StereoDeviceDefinition {
  int leftDevice;   ///< Identifier of device that is used for stitching the left eye
  int rightDevice;  ///< Identifier of device that is used for stitching the right eye
  int& device = leftDevice;
  /**
   * @param rhs rhs.
   */
  bool operator<(const StereoDeviceDefinition& rhs) const { return this->leftDevice > rhs.leftDevice; }
};
}  // namespace Core

namespace GPU {
VS_EXPORT Status checkDefaultBackendDeviceInitialization();

VS_EXPORT Status useDefaultBackendDevice();

VS_EXPORT Status setDefaultBackendDevice(int device);
VS_EXPORT Status setDefaultBackendDeviceVS(int device);

VS_EXPORT Status getDefaultBackendDevice(int* device);

VS_EXPORT Status getDefaultBackendDeviceContext(void* context);

VS_EXPORT Discovery::Framework getFramework();

VS_EXPORT PotentialValue<size_t> getMemoryUsage();

VS_EXPORT PotentialValue<std::vector<size_t> > getMemoryUsageByDevices();

}  // namespace GPU
}  // namespace VideoStitch
