// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "ximeaDiscovery.hpp"

#include <iostream>

#include "libvideostitch/logging.hpp"

#include <chrono>
#include <thread>

using namespace VideoStitch;
using namespace Plugin;

XimeaDiscovery::XimeaDiscovery(std::vector<std::shared_ptr<Device>> devices)
    : m_devices(devices), width(0), height(0), fps(0.0) {
  // Check paramater from cam 0 only as it take a lot of time to op en it
  HANDLE xiHdl;
  XI_RETURN stat = xiOpenDevice(0, &xiHdl);
  if (stat != XI_OK) {
    Logger::get(Logger::Info) << "[Ximea] Error opening device : " << stat << std::endl;
  }

  XI_PRM_TYPE xiType = xiTypeString;
  char buff[255];
  DWORD val = 0;
  DWORD size = 255 * sizeof(char);

  stat = xiGetParam(xiHdl, XI_PRM_API_VERSION, buff, &size, &xiType);
  if (stat != XI_OK) {
    Logger::get(Logger::Error) << "[Ximea] Error getting version " << stat << std::endl;
  }
  Logger::get(Logger::Info) << "[Ximea] Api Version : " << buff << std::endl;

  xiType = xiTypeInteger;
  size = sizeof(DWORD);
  stat = xiGetParam(xiHdl, XI_PRM_WIDTH, &val, &size, &xiType);
  if (stat != XI_OK) {
    Logger::get(Logger::Error) << "[Ximea] Error getting width " << stat << std::endl;
  }
  width = (uint16_t)val;
  stat = xiGetParam(xiHdl, XI_PRM_HEIGHT, &val, &size, &xiType);
  if (stat != XI_OK) {
    Logger::get(Logger::Error) << "[Ximea] Error getting height " << stat << std::endl;
  }
  height = (uint16_t)val;
  xiType = xiTypeFloat;
  float fval = 0.0;
  size = sizeof(float);
  stat = xiGetParam(xiHdl, XI_PRM_FRAMERATE, &fval, &size, &xiType);
  if (stat != XI_OK) {
    Logger::get(Logger::Error) << "[Ximea] Error getting fps " << stat << std::endl;
  }
  fps = fval;

  Logger::get(Logger::Info) << "[Ximea] image width x height @fps : " << width << "x" << height << "@" << fps
                            << std::endl;

  stat = xiCloseDevice(xiHdl);
  if (stat != XI_OK) {
    Logger::get(Logger::Info) << "[Ximea] Error closing device : " << stat << std::endl;
  }
}

XimeaDiscovery::~XimeaDiscovery() {
  // TODO Auto-generated destructor stub
}

XimeaDiscovery* XimeaDiscovery::create() {
  DWORD numDev = 0;
  XI_RETURN stat = xiGetNumberDevices(&numDev);

  if ((stat != XI_OK) || (numDev == 0)) {
    Logger::get(Logger::Info) << "[Ximea] no device found or when getting the list : " << stat
                              << " num device : " << numDev << std::endl;
    return nullptr;
  }

  Logger::get(Logger::Info) << "[Ximea] Number of device found : " << numDev << std::endl;

  char buff[255];
  DWORD size = 255;
  XI_PRM_TYPE xiType = xiTypeString;

  std::vector<std::shared_ptr<Device>> devices;
  // Ximea cam are labeled from 1 to numDev
  for (uint8_t i = 0; i < numDev; ++i) {
    Logger::get(Logger::Info) << "[Ximea] Device " << int(i) + 1;

    std::shared_ptr<InputDevice> inputDevice = std::make_shared<InputDevice>();

    stat = xiGetDeviceInfo(i, XI_PRM_DEVICE_NAME, buff, &size, &xiType);
    if (stat != XI_OK) {
      Logger::get(Logger::Error) << "[Ximea] Error getting device name : " << stat << std::endl;
    } else {
      Logger::get(Logger::Info) << " name " << buff << std::endl;
    }

    inputDevice->pluginDevice.displayName = std::string(buff) + " Input " + std::to_string(i + 1);
    inputDevice->pluginDevice.name = std::string(buff) + std::to_string(i);
    inputDevice->pluginDevice.type = Plugin::DiscoveryDevice::CAPTURE;
    inputDevice->pluginDevice.mediaType = Plugin::DiscoveryDevice::MediaType::VIDEO;
    inputDevice->camIdx = i;
    devices.push_back(inputDevice);
  }

  XimeaDiscovery* XD = new XimeaDiscovery(devices);
  return XD;
}

std::vector<Plugin::DiscoveryDevice> XimeaDiscovery::inputDevices() {
  std::vector<Plugin::DiscoveryDevice> pluginDevices;

  for (auto it = m_devices.begin(); it != m_devices.end(); ++it) {
    pluginDevices.push_back((*it)->pluginDevice);
  }

  return pluginDevices;
}

void XimeaDiscovery::registerAutoDetectionCallback(AutoDetection&) { return; }

std::vector<DisplayMode> XimeaDiscovery::supportedDisplayModes(const Plugin::DiscoveryDevice&) {
  // Ximea Cam support downsampling 1x1,2,3,4,5,6,7,8,9,10,16x16
  std::vector<DisplayMode> dpVec;

  DisplayMode dp(width, height, false, 30, false);
  dpVec.push_back(dp);

  return dpVec;
}
std::vector<PixelFormat> XimeaDiscovery::supportedPixelFormat(const Plugin::DiscoveryDevice&) {
  std::vector<PixelFormat> pfVec;
  pfVec.push_back(Grayscale);

  return pfVec;
}

/*
bool XimeaDiscovery::supportVideoMode(const Plugin::DiscoveryDevice&, const DisplayMode&, const PixelFormat&) {

        return false;
}
*/

DisplayMode XimeaDiscovery::currentDisplayMode(const Plugin::DiscoveryDevice& device) {
  return DisplayMode(width, height, false, (int)fps, false);
}
