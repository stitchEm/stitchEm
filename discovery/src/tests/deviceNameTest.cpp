// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "testing_common.hpp"
#include "deviceHelper.hpp"

#include <cstring>

namespace VideoStitch {
namespace Testing {

void testImproveDeviceName(const char *deviceName, Discovery::Vendor vendor, const char *expectedName) {
  Discovery::DeviceProperties deviceProp{
      "", "v1", 64, 1500000, Discovery::DeviceType::GPU, vendor, false, Discovery::Framework::CUDA};
  memcpy(deviceProp.name, deviceName, strlen(deviceName) + 1);
  improveDeviceName(deviceProp);
  const std::string name(deviceProp.name);
  ENSURE(strcmp(deviceProp.name, expectedName) == 0);
}

void testDeviceNamesIntel() {
  testImproveDeviceName("Intel(R) Iris(TM) Pro Graphics 580", Discovery::Vendor::INTEL, "Intel Iris Pro Graphics 580");

  testImproveDeviceName("Intel Iris Pro Graphics 580", Discovery::Vendor::INTEL, "Intel Iris Pro Graphics 580");

  testImproveDeviceName("Intel(R) HD Graphics 4600", Discovery::Vendor::INTEL, "Intel HD Graphics 4600");

  testImproveDeviceName("Intel(R) Graphics Chiuahua(R) 4500", Discovery::Vendor::INTEL, "Intel Graphics Chiuahua 4500");

  testImproveDeviceName("Iris Pro", Discovery::Vendor::INTEL, "Intel Iris Pro");
}

void testDeviceNamesAMD() {
  testImproveDeviceName("Fiji", Discovery::Vendor::AMD, "AMD Fiji");

  testImproveDeviceName("AMD Radeon HD - FirePro D300", Discovery::Vendor::AMD, "AMD Radeon HD - FirePro D300");

  testImproveDeviceName("Radeon R9", Discovery::Vendor::AMD, "AMD Radeon R9");
}

void testDeviceNamesNvidia() {
  testImproveDeviceName("GeForce GTX 980", Discovery::Vendor::NVIDIA, "Nvidia GeForce GTX 980");
}

void testUnknownVendor() {
  testImproveDeviceName("Unknown Vendor(R) Unknown Device(TM)", Discovery::Vendor::UNKNOWN,
                        "Unknown Vendor Unknown Device");
}

#define NAME_256     \
  "xxxxxxxxxxxxxxxx" \
  "xxxxxxxxxxxxxxxx" \
  "xxxxxxxxxxxxxxxx" \
  "xxxxxxxxxxxxxxxx" \
  "xxxxxxxxxxxxxxxx" \
  "xxxxxxxxxxxxxxxx" \
  "xxxxxxxxxxxxxxxx" \
  "xxxxxxxxxxxxxxxx" \
  "xxxxxxxxxxxxxxxx" \
  "xxxxxxxxxxxxxxxx" \
  "xxxxxxxxxxxxxxxx" \
  "xxxxxxxxxxxxxxxx" \
  "xxxxxxxxxxxxxxxx" \
  "xxxxxxxxxxxxxxxx" \
  "xxxxxxxxxxxxxxxx" \
  "xxxxxxxxxxxxxxx"

void testLongDeviceName() { testImproveDeviceName(NAME_256, Discovery::Vendor::INTEL, NAME_256); }

}  // namespace Testing
}  // namespace VideoStitch

int main(int, char **) {
  VideoStitch::Testing::initTest();

  VideoStitch::Testing::testDeviceNamesIntel();
  VideoStitch::Testing::testDeviceNamesAMD();
  VideoStitch::Testing::testDeviceNamesNvidia();

  VideoStitch::Testing::testUnknownVendor();

  VideoStitch::Testing::testLongDeviceName();

  return 0;
}
