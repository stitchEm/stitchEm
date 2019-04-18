// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "gpu/testing.hpp"
#include "gpu/util.hpp"
#include "common/ptv.hpp"

#include <gpu/stream.hpp>
#include "gpu/allocator.hpp"

#include <core/photoTransform.hpp>
#include <processors/photoCorrProcessor.hpp>

#include "libvideostitch/allocator.hpp"
#include "libvideostitch/gpu_device.hpp"
#include "libvideostitch/inputDef.hpp"
#include "libvideostitch/logging.hpp"

#include <memory>
#include <vector>

namespace VideoStitch {
namespace Testing {
void testCommon(const Core::PreProcessor& processor, const int width, const int height, const uint32_t* inputData,
                const uint32_t* expectedOutputData) {
  GPU::Buffer<uint32_t> buffer = GPU::Buffer<uint32_t>::allocate(width * height, "staging").value();
  GPU::memcpyBlocking(buffer, inputData, width * height * 4);
  auto surf = Core::OffscreenAllocator::createSourceSurface(width, height, "ProcessorsTest");
  ENSURE(surf.ok());
  GPU::memcpyBlocking(*surf->pimpl->surface, buffer.as_const());

  GPU::Stream s = GPU::Stream::getDefault();
  processor.process(0, *surf->pimpl->surface, width, height, 0, s);
  s.synchronize();

  std::vector<uint32_t> output;
  output.resize(width * height);
  GPU::memcpyBlocking(output.data(), *surf->pimpl->surface);
  ENSURE_RGBA8888_ARRAY_EQ(expectedOutputData, output.data(), width, height);

  buffer.release();
}

void testPhotoCorrProcessorCommon(const std::string photoResponse, const std::string& additionalConfig,
                                  const uint32_t* expectedOutputData, const float3 colorMult) {
  std::string mergedConfig =
      "{"
      " \"width\": 4,"
      " \"height\": 3,"
      " \"viewpoint_model\": \"ptgui\","
      " \"response\": \"" +
      photoResponse +
      "\","
      " \"reader_config\": {},"      // Dummy
      " \"hfov\": 123.4,"            // Dummy
      " \"yaw\": 0.0,"               // Dummy
      " \"pitch\": 0.0,"             // Dummy
      " \"roll\": 0.0,"              // Dummy
      " \"proj\": \"rectilinear\"";  // Dummy

  if (additionalConfig.size()) {
    mergedConfig += ",";
    mergedConfig += additionalConfig;
  }

  mergedConfig += "}";

  const std::unique_ptr<Ptv::Value> inputDefPtv(makePtvValue(mergedConfig));
  const std::unique_ptr<Core::InputDefinition> inputDef(Core::InputDefinition::create(*inputDefPtv));
  ENSURE((bool)inputDef, "cannot create inputDef");
  const std::unique_ptr<Core::DevicePhotoTransform> photoTransform(Core::DevicePhotoTransform::create(*inputDef));
  Core::PhotoCorrPreProcessor processor(*inputDef, colorMult, *photoTransform);
  const uint32_t inputData[] = {
      Image::RGBA::pack(128, 0, 0, 255), Image::RGBA::pack(0, 128, 0, 255), Image::RGBA::pack(0, 0, 128, 255),
      Image::RGBA::pack(0, 128, 0, 255), Image::RGBA::pack(128, 0, 0, 255), Image::RGBA::pack(0, 128, 0, 255),
      Image::RGBA::pack(0, 0, 128, 255), Image::RGBA::pack(0, 128, 0, 255), Image::RGBA::pack(128, 0, 0, 255),
      Image::RGBA::pack(0, 128, 0, 255), Image::RGBA::pack(0, 0, 128, 255), Image::RGBA::pack(0, 128, 0, 255),
  };
  testCommon(processor, (int)inputDef->getWidth(), (int)inputDef->getHeight(), inputData, expectedOutputData);
}

void testPhotoCorrProcessorIdentity() {
  const uint32_t expectedOutputData[] = {
      Image::RGBA::pack(128, 0, 0, 255), Image::RGBA::pack(0, 128, 0, 255), Image::RGBA::pack(0, 0, 128, 255),
      Image::RGBA::pack(0, 128, 0, 255), Image::RGBA::pack(128, 0, 0, 255), Image::RGBA::pack(0, 128, 0, 255),
      Image::RGBA::pack(0, 0, 128, 255), Image::RGBA::pack(0, 128, 0, 255), Image::RGBA::pack(128, 0, 0, 255),
      Image::RGBA::pack(0, 128, 0, 255), Image::RGBA::pack(0, 0, 128, 255), Image::RGBA::pack(0, 128, 0, 255),
  };
  const float3 colorMult = {1.0f, 1.0f, 1.0f};
  testPhotoCorrProcessorCommon("linear", "", expectedOutputData, colorMult);
}

void testPhotoCorrProcessorColorMult() {
  const uint32_t expectedOutputData[] = {
      Image::RGBA::pack(154, 0, 0, 255), Image::RGBA::pack(0, 179, 0, 255), Image::RGBA::pack(0, 0, 102, 255),
      Image::RGBA::pack(0, 179, 0, 255), Image::RGBA::pack(154, 0, 0, 255), Image::RGBA::pack(0, 179, 0, 255),
      Image::RGBA::pack(0, 0, 102, 255), Image::RGBA::pack(0, 179, 0, 255), Image::RGBA::pack(154, 0, 0, 255),
      Image::RGBA::pack(0, 179, 0, 255), Image::RGBA::pack(0, 0, 102, 255), Image::RGBA::pack(0, 179, 0, 255),
  };
  const float3 colorMult = {1.2f, 1.4f, 0.8f};
  testPhotoCorrProcessorCommon("linear", "", expectedOutputData, colorMult);
}

void testPhotoCorrProcessorGamma() {
  const uint32_t expectedOutputData[] = {
      Image::RGBA::pack(128, 0, 0, 255), Image::RGBA::pack(0, 168, 0, 255), Image::RGBA::pack(0, 0, 203, 255),
      Image::RGBA::pack(0, 168, 0, 255), Image::RGBA::pack(128, 0, 0, 255), Image::RGBA::pack(0, 168, 0, 255),
      Image::RGBA::pack(0, 0, 203, 255), Image::RGBA::pack(0, 168, 0, 255), Image::RGBA::pack(128, 0, 0, 255),
      Image::RGBA::pack(0, 168, 0, 255), Image::RGBA::pack(0, 0, 203, 255), Image::RGBA::pack(0, 168, 0, 255),
  };

  const float3 colorMult = {1.0, 1.5, 2.0};
  testPhotoCorrProcessorCommon("gamma", " \"gamma\": 1.5", expectedOutputData, colorMult);
}

void testPhotoCorrProcessorEmor() {
  const uint32_t expectedOutputData[] = {
      Image::RGBA::pack(128, 8, 21, 255), Image::RGBA::pack(0, 180, 21, 255), Image::RGBA::pack(0, 8, 243, 255),
      Image::RGBA::pack(0, 180, 21, 255), Image::RGBA::pack(128, 8, 21, 255), Image::RGBA::pack(0, 180, 21, 255),
      Image::RGBA::pack(0, 8, 243, 255),  Image::RGBA::pack(0, 180, 21, 255), Image::RGBA::pack(128, 8, 21, 255),
      Image::RGBA::pack(0, 180, 21, 255), Image::RGBA::pack(0, 8, 243, 255),  Image::RGBA::pack(0, 180, 21, 255),
  };

  const float3 colorMult = {1.0, 1.5, 2.0};
  testPhotoCorrProcessorCommon("emor",
                               " \"emor_a\" : 6.75603, \"emor_b\" : 0.27799, \"emor_c\" : 0.749483, \"emor_d\" : "
                               "-0.438113, \"emor_e\" : 0.0659597",
                               expectedOutputData, colorMult);
}

void testPhotoCorrProcessorVignetting() {
  /*
  +----+----+----+----+
  | a  | b  | b  | a  |
  +----+----+----+----+
  | c  | d  | d  | c  |
  +----+----+----+----+
  | a  | b  | b  | a  |
  +----+----+----+----+
  R^2 = 4^2 + 3^2 / 2^2 = 25 / 4
  a has absolute coords (3/2, 1) -> 1 + (r / R)^2 = 1 + 13 / 25  -> output ~ 84
  b has absolute coords (1/2, 1) -> 1 + (r / R)^2 = 1 + 1 / 5  -> output ~ 107
  c has absolute coords (3/2, 0) -> 1 + (r / R)^2 = 1 + 9 / 25  -> output ~ 94
  c has absolute coords (1/2, 0) -> 1 + (r / R)^2 = 1 + 1 / 25  -> output ~ 123

  */

  const uint32_t expectedOutputData[] = {
      Image::RGBA::pack(84, 0, 0, 255),  Image::RGBA::pack(0, 107, 0, 255), Image::RGBA::pack(0, 0, 107, 255),
      Image::RGBA::pack(0, 84, 0, 255),  Image::RGBA::pack(94, 0, 0, 255),  Image::RGBA::pack(0, 123, 0, 255),
      Image::RGBA::pack(0, 0, 123, 255), Image::RGBA::pack(0, 94, 0, 255),  Image::RGBA::pack(84, 0, 0, 255),
      Image::RGBA::pack(0, 107, 0, 255), Image::RGBA::pack(0, 0, 107, 255), Image::RGBA::pack(0, 84, 0, 255),
  };
  const float3 colorMult = {1.0f, 1.0f, 1.0f};
  testPhotoCorrProcessorCommon("linear",
                               " \"vign_a\": 1.0,"
                               " \"vign_b\": 1.0",
                               expectedOutputData, colorMult);
}

/////////////////////////////////

void testForDebug() {
  std::string mergedConfig =
      "{"
      " \"width\": 640,"
      " \"height\": 480,"
      " \"viewpoint_model\": \"ptgui\","
      " \"response\": \"emor\","
      " \"reader_config\": {},"  // Dummy
      " \"hfov\": 123.4,"        // Dummy
      " \"yaw\": 0.0,"           // Dummy
      " \"pitch\": 0.0,"         // Dummy
      " \"roll\": 0.0,"          // Dummy
      " \"vign_a\": 1.0,"
      " \"vign_b\": 1.0,"
      " \"proj\": \"rectilinear\"}";  // Dummy

  const std::unique_ptr<Ptv::Value> inputDefPtv(makePtvValue(mergedConfig));
  const std::unique_ptr<Core::InputDefinition> inputDef(Core::InputDefinition::create(*inputDefPtv));
  ENSURE((bool)inputDef, "cannot create inputDef");
  const std::unique_ptr<Core::DevicePhotoTransform> photoTransform(Core::DevicePhotoTransform::create(*inputDef));
  const float3 colorMult = {1.0f, 1.0f, 1.0f};
  Core::PhotoCorrPreProcessor processor(*inputDef, colorMult, *photoTransform);

  std::vector<uint32_t> inputData((int)inputDef->getWidth() * (int)inputDef->getHeight(),
                                  Image::RGBA::pack(128, 128, 128, 255));
  GPU::Buffer<uint32_t> buffer =
      GPU::Buffer<uint32_t>::allocate(inputDef->getWidth() * inputDef->getHeight(), "staging").value();
  GPU::memcpyBlocking(buffer, inputData.data());
  auto surf =
      Core::OffscreenAllocator::createSourceSurface(inputDef->getWidth(), inputDef->getHeight(), "ProcessorsTest");
  ENSURE(surf.ok());
  GPU::memcpyBlocking(*surf->pimpl->surface, buffer.as_const());

  GPU::Stream s = GPU::Stream::getDefault();
  processor.process(0, *surf->pimpl->surface, inputDef->getWidth(), inputDef->getHeight(), 0, s);
  s.synchronize();

  std::vector<uint32_t> output;
  output.resize(inputDef->getWidth() * inputDef->getHeight());
  GPU::memcpyBlocking(output.data(), *surf->pimpl->surface);
  Util::PngReader reader;
  ENSURE(reader.writeRGBAToFile("vignettingtest.png", inputDef->getWidth(), inputDef->getHeight(), output.data()));
}

}  // namespace Testing
}  // namespace VideoStitch

int main(int /*argc*/, char** /*argv*/) {
  VideoStitch::Testing::initTest();
  VideoStitch::Testing::ENSURE(VideoStitch::GPU::setDefaultBackendDevice(0));

  VideoStitch::Logger::setLevel(VideoStitch::Logger::Debug);
  VideoStitch::Testing::testPhotoCorrProcessorIdentity();
  VideoStitch::Testing::testPhotoCorrProcessorColorMult();
  VideoStitch::Testing::testPhotoCorrProcessorGamma();
  VideoStitch::Testing::testPhotoCorrProcessorEmor();
  VideoStitch::Testing::testPhotoCorrProcessorVignetting();
  // VideoStitch::Testing::testForDebug();
  return 0;
}
