// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "gpu/testing.hpp"

#include "gpu/core1/transform.hpp"

#include "common/ptv.hpp"

#include "libvideostitch/allocator.hpp"

#include "libvideostitch/gpu_device.hpp"
#include "core/geoTransform.hpp"

#include "gpu/allocator.hpp"

#include <future>

// #define DUMP_DEBUG

// #define DEBUG_RUN_SERIAL

#ifdef DUMP_DEBUG
#include "util/debugUtils.hpp"
#endif

namespace VideoStitch {
namespace Testing {

void testMapBuffer(const Core::PanoDefinition& panoDef) {
  // test the equivalence of pano-input mapping between GPU and C++ backends:
  //
  // encode the input coordinates in the RGBA values of the input image
  // map the input to the pano on the GPU (mapBuffer), pano->input projection
  // --> the pano contains pixels that reference their original input coordinates
  // copy the resulting pano to RAM
  // map some examples back to the input on the CPU (mapPanoramaToInput)
  // the computed coordinates should be the same as the ones originally encoded in the image RGBA

  const auto& firstInput = panoDef.getVideoInput(0);

  std::unique_ptr<Core::Transform> transform(Core::Transform::create(firstInput));
  ENSURE(transform != nullptr);

  Logger::get(Logger::Info) << "Testing pano size: " << panoDef.getWidth() << " x " << panoDef.getHeight() << std::endl;
  Logger::get(Logger::Info) << "Input size: " << firstInput.getWidth() << " x " << firstInput.getHeight() << std::endl;

  GPU::Stream stream = GPU::Stream::getDefault();
  const frameid_t frame = 0;

  Potential<Core::SourceSurface> sourceSurf =
      Core::OffscreenAllocator::createSourceSurface(firstInput.getWidth(), firstInput.getHeight(), "testMapBuffer");
  ENSURE(sourceSurf.ok());

  const unsigned char* mask = nullptr;
  Core::Rect boundingBox =
      Core::Rect::fromInclusiveTopLeftBottomRight(0, 0, panoDef.getHeight() - 1, panoDef.getWidth() - 1);

  ENSURE_EQ(boundingBox.getWidth(), panoDef.getWidth());
  ENSURE_EQ(boundingBox.getHeight(), panoDef.getHeight());

  GPU::memsetToZeroAsync(*sourceSurf->pimpl->surface, stream);

  // populate input image with sparse encoded coordinate values
  unsigned char* hostBuf = new unsigned char[firstInput.getHeight() * firstInput.getWidth() * 4];
  for (int y = 0; y < firstInput.getHeight(); y++) {
    for (int x = 0; x < firstInput.getWidth(); x++) {
      const auto idx = (y * firstInput.getWidth() + x) * 4;
      if (x % 2 == 0 && y % 2 == 0) {
        hostBuf[idx + 0] = x % 255;  // R
        hostBuf[idx + 1] = y % 255;  // G
        hostBuf[idx + 2] = 0;        // B
        hostBuf[idx + 3] = 255;      // A
      } else {
        hostBuf[idx + 0] = 0;  // R
        hostBuf[idx + 1] = 0;  // G
        hostBuf[idx + 2] = 0;  // B
        hostBuf[idx + 3] = 0;  // A
      }
    }
  }

  ENSURE(GPU::memcpyAsync(*sourceSurf->pimpl->surface, (uint32_t*)hostBuf, stream));

  Potential<Core::SourceSurface> emptySurf = Core::OffscreenAllocator::createSourceSurface(256, 256, "testMapBuffer");
  GPU::memsetToZeroAsync(*emptySurf->pimpl->surface, stream);

  auto panoOut = GPU::uniqueBuffer<uint32_t>(panoDef.getWidth() * panoDef.getHeight(), "panoOut");

  // map input to pano on the GPU
  ENSURE(transform
             ->mapBuffer(frame, panoOut.borrow(), *emptySurf->pimpl->surface, mask, boundingBox, panoDef, firstInput,
                         *sourceSurf->pimpl->surface, stream)
             .ok());

  uint8_t* result = new uint8_t[panoDef.getWidth() * panoDef.getHeight() * 4];

  GPU::memcpyAsync((uint32_t*)result, panoOut.borrow_const(), stream);
  ENSURE(stream.synchronize());

#ifdef DUMP_DEBUG
  static int example = 0;
  std::string fileName{"/tmp/panoOut-" + std::to_string(example) + ".png"};
  Debug::dumpRGBADeviceBuffer(fileName.c_str(), panoOut.borrow_const(), panoDef.getWidth(), panoDef.getHeight());
  example++;
#endif

  std::unique_ptr<Core::TransformStack::GeoTransform> geoTransform(
      Core::TransformStack::GeoTransform::create(panoDef, firstInput));
  ENSURE(geoTransform != nullptr);

  const int time = 0;
  uint64_t tested = 0;

  for (int y = 0; y < panoDef.getHeight(); y++) {
    for (int x = 0; x < panoDef.getWidth(); x++) {
      const unsigned char alpha = result[(y * panoDef.getWidth() + x) * 4 + 3];

      if (alpha >= 250) {
        const int inputX255 = result[(y * panoDef.getWidth() + x) * 4 + 0];
        const int inputY255 = result[(y * panoDef.getWidth() + x) * 4 + 1];
        const int inputEmpty = result[(y * panoDef.getWidth() + x) * 4 + 2];

        // sanity check
        ENSURE_EQ(inputEmpty, 0);

        Core::TopLeftCoords2 panoCoords{(float)x, (float)y};
        Core::TopLeftCoords2 panoCenter{(panoDef.getWidth() - 1) / 2.f, (panoDef.getHeight() - 1) / 2.f};

        Core::CenterCoords2 panoCoordsCentered{panoCoords, panoCenter};

        // map pano to input in C++
        Core::CenterCoords2 coords = geoTransform->mapPanoramaToInput(firstInput, panoCoordsCentered, time);

        Core::TopLeftCoords2 inputCenter{(firstInput.getWidth() - 1) / 2.f, (firstInput.getHeight() - 1) / 2.f};
        Core::TopLeftCoords2 inputTopLeft{coords, inputCenter};

        float epsilon = 0.6f;

        // if alpha is not full, there was interpolation with the surrounding (0) coord values
        // results will be way off, let's have some inexact comparisons for regression testing
        epsilon += (255 - alpha);

        ENSURE_APPROX_EQ(fmodf(inputTopLeft.x - inputX255 + epsilon, 255) - epsilon, 0.0f, epsilon);
        ENSURE_APPROX_EQ(fmodf(inputTopLeft.y - inputY255 + epsilon, 255) - epsilon, 0.0f, epsilon);

        tested++;
      }
    }
  }

  Logger::get(Logger::Info) << "Tested coordinates: " << tested << std::endl;

  delete[] result;
  delete[] hostBuf;
}

void testInverseMapBuffer(const Core::PanoDefinition& panoDef) {
  // map first input from input to pano on GPU (mapCoordInput), input->pano projection
  // stores the computed pano coordinates in GPU buffer
  // copy to host, choose a few random samples
  // map them to pano (input->pano projection) on host, check against stored coordinates

  const int scaleFactor = 1;
  const int time = 0;

  GPU::Stream stream = GPU::Stream::getDefault();

  const auto& firstInput = panoDef.getVideoInput(0);

  std::unique_ptr<Core::Transform> transform(Core::Transform::create(firstInput));
  ENSURE(transform != nullptr);

  std::unique_ptr<Core::TransformStack::GeoTransform> geoTransform(
      Core::TransformStack::GeoTransform::create(panoDef, firstInput));
  ENSURE(geoTransform != nullptr);

  auto coordGPUBuffer = GPU::uniqueBuffer<float2>(firstInput.getWidth() * firstInput.getHeight(), "InverseMapping");
  float2* hostBuf = new float2[firstInput.getWidth() * firstInput.getHeight()];

  const float2 INIT_VAL = {-12345.f, -12345.f};

  std::fill(hostBuf, hostBuf + firstInput.getWidth() * firstInput.getHeight(), INIT_VAL);
  GPU::memcpyAsync(coordGPUBuffer.borrow(), hostBuf, stream);

  ENSURE(transform->mapCoordInput(time, scaleFactor, coordGPUBuffer.borrow(), panoDef, firstInput, stream));

  GPU::memcpyAsync(hostBuf, coordGPUBuffer.borrow_const(), stream);
  ENSURE(stream.synchronize());

  const Core::TopLeftCoords2 panoCenter{(panoDef.getWidth() - 1) / 2.f, (panoDef.getHeight() - 1) / 2.f};
  const Core::TopLeftCoords2 inputCenter{(firstInput.getWidth() - 1) / 2.f, (firstInput.getHeight() - 1) / 2.f};

  double avgError = 0.;

  const int numTestCountPerTransform = 1000;

  const auto numCoordinates = firstInput.getWidth() * firstInput.getHeight();
  const auto coordTestSteps = numCoordinates / numTestCountPerTransform;

  for (int testCase = 0; testCase < numTestCountPerTransform; testCase++) {
    const auto testCoord = testCase * coordTestSteps;
    const auto x = testCoord % firstInput.getWidth();
    const auto y = testCoord / firstInput.getWidth();
    const float2 mappedCoords = hostBuf[y * firstInput.getWidth() + x];

    // every input pixel should have been written
    ENSURE(mappedCoords.x != INIT_VAL.x && mappedCoords.y != INIT_VAL.y);

    const Core::TopLeftCoords2 inputCoords{(float)x, (float)y};
    const Core::CenterCoords2 inputCoordsCentered{inputCoords, inputCenter};

    if (geoTransform->isWithinInputBounds(firstInput, inputCoords)) {
      const Core::CenterCoords2 hostMapped = geoTransform->mapInputToPanorama(firstInput, inputCoordsCentered, 0);
      const Core::TopLeftCoords2 hostMappedTopLeft{hostMapped, panoCenter};

      // consider float errors adding up to wrapping
      double xDiff =
          std::min(std::abs((double)hostMappedTopLeft.x - (double)mappedCoords.x),
                   std::abs(std::abs((double)hostMappedTopLeft.x - (double)mappedCoords.x) - panoDef.getWidth()));

      double yDiff =
          std::min(std::abs((double)hostMappedTopLeft.y - (double)mappedCoords.y),
                   std::abs(std::abs((double)hostMappedTopLeft.y - (double)mappedCoords.y) - panoDef.getHeight()));

      ENSURE_APPROX_EQ(xDiff, 0., 0.5);
      ENSURE_APPROX_EQ(yDiff, 0., 0.5);
    } else {
      ENSURE_EQ(mappedCoords.x, INVALID_FLOW_VALUE);
      ENSURE_EQ(mappedCoords.y, INVALID_FLOW_VALUE);
    }
  }

  ENSURE(avgError / 2. / firstInput.getHeight() / firstInput.getWidth() < 0.001);

  delete[] hostBuf;
}

void testCases() {
  int backendDeviceUsedForTest = 0;
  int vsDeviceIndex = -1;
  VideoStitch::Testing::ENSURE(VideoStitch::GPU::setDefaultBackendDevice(backendDeviceUsedForTest));
  ENSURE(VideoStitch::Discovery::getVSDeviceIndex(backendDeviceUsedForTest, vsDeviceIndex,
                                                  VideoStitch::GPU::getFramework()));

#if defined(OCLGRIND)
  std::vector<size_t> testCases{0, 7};
#else
  std::vector<size_t> testCases{0, 1, 2, 3, 4, 5, 6, 7};
#endif

  std::vector<std::unique_ptr<Core::PanoDefinition>> panoDefs;
  for (size_t test : testCases) {
    Potential<Ptv::Parser> parser = Ptv::Parser::create();
    ENSURE(parser.ok());
    // parser does not seem to be thread-safe
    ENSURE(parser->parse("data/ptv/test" + std::to_string(test) + ".ptv"));
    ENSURE(parser->getRoot().has("pano"));
    panoDefs.emplace_back(Core::PanoDefinition::create(*parser->getRoot().has("pano")));
  }

  {
    std::vector<std::future<void>> futures;
    int testID = 0;
    for (const auto& panoDef : panoDefs) {
      auto fut = std::async([=, &panoDef]() {
        testMapBuffer(*panoDef);

        // VSA-7361: test5.ptv produces completely different values on CPU and GPU
        // VSA-7361: test6.ptv produces up to 3.6 pixels error between CPU and GPU
        if (testID == 5 || testID == 6) {
          return;
        }

        Discovery::DeviceProperties deviceProps;
        ENSURE(Discovery::getDeviceProperties(vsDeviceIndex, deviceProps));
        if (deviceProps.vendor == Discovery::Vendor::INTEL) {
#ifdef __APPLE__
          // VSA-7363 Mac: Intel GPUs crash or produce invalid results with undistortion
          return;
#else
          // VSA-7377 Intel NUC produces different results between GPU and CPU for this example
          if (testID == 1) {
            return;
          }
#endif
        }

        testInverseMapBuffer(*panoDef);
      });
      testID++;
#if defined(DEBUG_RUN_SERIAL) || defined(OCLGRIND)
      fut.wait();
#else
      futures.push_back(std::move(fut));
#endif
    }
    for (auto& f : futures) {
      // supposed to be blocking in destructor, MSVC 2013 bug: doesn't block
      // workaround by explicitly .wait()ing
      f.wait();
    }
  }
}

}  // namespace Testing
}  // namespace VideoStitch

int main() {
  VideoStitch::Testing::initTest();
  VideoStitch::Testing::testCases();
  return 0;
}
