// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "gpu/testing.hpp"

#include "common/fakeReader.hpp"

#include <gpu/memcpy.hpp>
#include <core1/imageMapping.hpp>
#include <core1/inputsMap.hpp>
#include <core1/bounds.hpp>
#include <core1/textureTarget.hpp>
#include <util/pngutil.hpp>
#include <util/opticalFlowUtils.hpp>
#include <parallax/mergerPair.hpp>
#include <parallax/flowConstant.hpp>

#include "libvideostitch/parse.hpp"
#include "libvideostitch/panoDef.hpp"
#include "libvideostitch/inputFactory.hpp"
#include "libvideostitch/config.hpp"
#include "libvideostitch/output.hpp"
#include "libvideostitch/imageProcessingUtils.hpp"
#include "libvideostitch/gpu_device.hpp"

#include <string>
#include <sstream>
#include <algorithm>

/*
 * This test could be used to debug the space transform class in flow-based blending
 */

//#define DUMP_TEST_RESULT

#if defined(DUMP_TEST_RESULT)
//#undef NDEBUG
#ifdef NDEBUG
#error "This is not supposed to be included in non-debug mode."
#endif
#include <util/debugUtils.hpp>
#endif

namespace VideoStitch {
namespace Testing {

Status prepareMappings(const Core::PanoDefinition& panoDef,
                       std::map<readerid_t, VideoStitch::Core::ImageMapping*>& imageMappings,
                       GPU::Buffer<uint32_t> inputsMap) {
  std::atomic<int> readFrameCalls(0), readFrameExits(0);
  std::unique_ptr<FakeReaderFactory> fakeReaderFactory =
      std::unique_ptr<FakeReaderFactory>(new FakeReaderFactory(0, &readFrameCalls, &readFrameExits));
  std::map<readerid_t, Input::VideoReader*> readers;
  for (readerid_t in = 0; in < panoDef.numInputs(); ++in) {
    Potential<Input::Reader> reader = fakeReaderFactory->create(in, panoDef.getInput(in));
    FAIL_RETURN(reader.status());
    Input::VideoReader* videoReader = reader.release()->getVideoReader();
    if (videoReader) {
      readers[in] = videoReader;
    }
  }

  // Create mappings
  for (readerid_t inputId = 0; inputId < panoDef.numInputs(); ++inputId) {
    VideoStitch::Core::ImageMapping* imMap = new Core::ImageMapping(inputId);
    imageMappings[inputId] = imMap;
  }
  // Prepare image mapping
  VideoStitch::Potential<Core::InputsMap> potInputsMap = Core::InputsMap::create(panoDef);
  FAIL_RETURN(potInputsMap.status());

  potInputsMap.object()->compute(readers, panoDef, true);
  FAIL_RETURN(GPU::memcpyBlocking(inputsMap, potInputsMap.object()->getMask()));

  const int maxSize = (int)(panoDef.getWidth() > panoDef.getHeight() ? panoDef.getWidth() : panoDef.getHeight());
  auto tmpDevBuffer = VideoStitch::GPU::Buffer<uint32_t>::allocate(maxSize, "Input Bounding boxes");
  FAIL_RETURN(tmpDevBuffer.status());

  auto tmpHostBuffer = VideoStitch::GPU::HostBuffer<uint32_t>::allocate(maxSize, "Input Bounding boxes");
  FAIL_RETURN(tmpHostBuffer.status());

  FAIL_RETURN(Core::computeHBounds(Core::EQUIRECTANGULAR, panoDef.getWidth(), panoDef.getHeight(), imageMappings,
                                   nullptr, Eye::LeftEye, potInputsMap.object()->getMask(), tmpHostBuffer.value(),
                                   tmpDevBuffer.value(), VideoStitch::GPU::Stream::getDefault(), true))

  FAIL_RETURN(Core::computeVBounds(Core::EQUIRECTANGULAR, panoDef.getWidth(), panoDef.getHeight(), imageMappings,
                                   potInputsMap.object()->getMask(), tmpHostBuffer.value(), tmpDevBuffer.value(),
                                   VideoStitch::GPU::Stream::getDefault()));

  tmpDevBuffer.value().release();
  tmpHostBuffer.value().release();
  for (readerid_t in = 0; in < panoDef.numInputs(); ++in) {
    if (readers.find(in) != readers.end()) {
      delete readers[in];
    }
  }
  return Status::OK();
}

Status setupIdMask(const videoreaderid_t id, const Core::Rect rect, const GPU::Buffer<const float2> buffer,
                   const int2 size, std::vector<uint32_t>& outputVector) {
  std::vector<float2> data(rect.getArea());
  FAIL_RETURN(GPU::memcpyBlocking(&data[0], buffer, rect.getArea() * sizeof(float2)));

  for (int i = 0; i < rect.getWidth(); i++) {
    const int x = (rect.left() + i) % size.x;
    for (int j = 0; j < rect.getHeight(); j++) {
      const int y = (rect.top() + j) % size.y;
      const float2 value = data[j * rect.getWidth() + i];
      if (value.x != INVALID_FLOW_VALUE) {
        outputVector[y * size.x + x] |= (1 << id);
      }
    }
  }
  return Status::OK();
}

void testSpaceTransform() {
#ifdef DUMP_TEST_RESULT
  std::string workingPath =
      "C:/Users/Chuong.VideoStitch-09/Documents/GitHub/VideoStitch/VideoStitch-master/lib/src/test/";
#else
  std::string workingPath = "";
#endif

  std::vector<std::string> spaceTransformTests;

  for (int i = 0; i <= 0; i++) {
    spaceTransformTests.push_back(workingPath + "data/spacetransform/test" + std::to_string(i) + ".ptv");
  }

  for (int i = 0; i >= 0; i--) {
    std::string ptvFile = spaceTransformTests[i];
    Potential<Ptv::Parser> parser = Ptv::Parser::create();
    ENSURE(parser.ok());
    // Load the project and parse it.
    ENSURE(parser->parse(ptvFile));
    ENSURE(parser->getRoot().has("pano"));
    // Create a runtime panorama from the parsed project.
    Core::PanoDefinition* panoDef = Core::PanoDefinition::create(*parser->getRoot().has("pano"));
    ENSURE(panoDef);
    std::unique_ptr<Core::PanoDefinition> pano(panoDef);
    std::map<readerid_t, VideoStitch::Core::ImageMapping*> imageMappings;
    int2 size = make_int2((int)panoDef->getWidth(), (int)panoDef->getHeight());
    auto inputsMap = GPU::uniqueBuffer<uint32_t>(size.x * size.y, "Space Transform Test");
    ENSURE(prepareMappings(*pano.get(), imageMappings, inputsMap.borrow()));
    std::vector<uint32_t> inputsMapVector(size.x * size.y);
    ENSURE(GPU::memcpyBlocking(&inputsMapVector[0], inputsMap.borrow().as_const(), size.x * size.y * sizeof(uint32_t)));

    GPU::Stream stream = GPU::Stream::getDefault();
    bool first = true;
    std::vector<videoreaderid_t> id0s;
    std::pair<readerid_t, Core::ImageMapping*> prevMapping;

    for (auto mapping : imageMappings) {
      if (first) {
        first = false;
      } else {
        id0s.push_back((int)(prevMapping.first));
        Potential<Core::MergerPair> curPair =
            Core::MergerPair::create(*panoDef, nullptr, 1024, 175, id0s, mapping.second->getImId(),
                                     prevMapping.second->getOutputRect(Core::EQUIRECTANGULAR),
                                     mapping.second->getOutputRect(Core::EQUIRECTANGULAR), stream);
        ENSURE(curPair.status());
        if (curPair->doesOverlap()) {
          curPair->getInterToLookupSpaceCoordMappingBufferLevel(0, 0);

          std::vector<uint32_t> outputMaskVector(size.x * size.y, 0);

          const Core::LaplacianPyramid<float2>::LevelSpec<float2>& level0 =
              curPair->getInterToInputSpaceCoordMappingLaplacianPyramid(0)->getLevel(0);
          Core::Rect rect0 = curPair->getBoundingInterRect(0, 0);
          ENSURE(rect0.getWidth() == level0.width() && rect0.getHeight() == level0.height(), "Sizes do not matched");

          const Core::LaplacianPyramid<float2>::LevelSpec<float2>& level1 =
              curPair->getInterToInputSpaceCoordMappingLaplacianPyramid(1)->getLevel(0);
          Core::Rect rect1 = curPair->getBoundingInterRect(1, 0);
          ENSURE(rect1.getWidth() == level1.width() && rect1.getHeight() == level1.height(), "Sizes do not matched");

          setupIdMask(prevMapping.second->getImId(), rect0, level0.data(), size, outputMaskVector);
          setupIdMask(mapping.second->getImId(), rect1, level1.data(), size, outputMaskVector);

          {
            // Dump the masks of both images in the original pano space
            uint32_t mask = 0;
            std::stringstream ss;
            ss.str("");
            ss << workingPath + "data/spacetransform/test-" + std::to_string(i) << "-pair-pano";
            for (size_t j = 0; j < id0s.size(); j++) {
              ss << id0s[j];
              if (j != id0s.size() - 1) {
                ss << "+";
              }
              mask += (1 << id0s[j]);
            }
            std::vector<uint32_t> outputMaps;
            for (size_t j = 0; j < inputsMapVector.size(); j++) {
              uint32_t value = 0;
              if (inputsMapVector[j] & (1 << mapping.second->getImId())) {
                value += (1 << mapping.second->getImId());
              }
              if (inputsMapVector[j] & (mask)) {
                value += (1 << prevMapping.second->getImId());
              }
              outputMaps.push_back(value);
            }
            ss << "_" << mapping.second->getImId() << ".png";
#ifdef DUMP_TEST_RESULT
            Debug::dumpRGBAIndexDeviceBuffer<uint32_t>(ss.str().c_str(), outputMaps, size.x, size.y);
#else
            std::vector<unsigned char> outputMapsColor;
            Util::ImageProcessing::convertIndexToRGBA(outputMaps, outputMapsColor);
            ENSURE_PNG_FILE_EQ(ss.str(), outputMapsColor);
#endif  // DUMP_TEST_RESULT
          }
          {
            // Dump the masks of both images in the intermediate space
            std::stringstream ss;
            ss.str("");
            ss << workingPath + "data/spacetransform/test-" + std::to_string(i) << "-pair-intermediate";
            for (size_t j = 0; j < id0s.size(); j++) {
              ss << id0s[j];
              if (j != id0s.size() - 1) {
                ss << "+";
              }
            }
            ss << "_" << mapping.second->getImId() << ".png";

#ifdef DUMP_TEST_RESULT
            Debug::dumpRGBAIndexDeviceBuffer<uint32_t>(ss.str().c_str(), outputMaskVector, size.x, size.y);
#else
            std::vector<unsigned char> outputMapsColor;
            Util::ImageProcessing::convertIndexToRGBA(outputMaskVector, outputMapsColor);
            ENSURE_PNG_FILE_EQ(ss.str(), outputMapsColor);
#endif  // DUMP_TEST_RESULT
          }
        }
      }
      prevMapping = mapping;
    }

    for (auto imageMapping : imageMappings) {
      delete imageMapping.second;
    }

    std::cout << "*** Test " << i << " passed." << std::endl;
  }
}

}  // namespace Testing
}  // namespace VideoStitch

int main(int argc, char** argv) {
  VideoStitch::Testing::initTest();
  VideoStitch::Testing::ENSURE(VideoStitch::GPU::setDefaultBackendDevice(0));

  VideoStitch::Testing::testSpaceTransform();

  return 0;
}
