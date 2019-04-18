// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "gpu/testing.hpp"

#include "backend/common/core/types.hpp"

#include "common/container.hpp"
#include "common/fakeReader.hpp"
#include "common/ptv.hpp"

#include "core/geoTransform.hpp"
#include "core1/bounds.hpp"
#include "core1/imageMapping.hpp"
#include "core1/inputsMap.hpp"

#include "gpu/core1/transform.hpp"
#include "gpu/memcpy.hpp"

#include "parallax/flowConstant.hpp"
#include "parallax/mergerPair.hpp"
#include "util/opticalFlowUtils.hpp"
#include "util/pngutil.hpp"

#include "libvideostitch/config.hpp"
#include "libvideostitch/imageProcessingUtils.hpp"
#include "libvideostitch/inputFactory.hpp"
#include "libvideostitch/output.hpp"
#include "libvideostitch/panoDef.hpp"
#include "libvideostitch/parse.hpp"
#include "libvideostitch/gpu_device.hpp"

#include <algorithm>
#include <sstream>
#include <string>

//#define DUMP_TEST_RESULT

#if defined(DUMP_TEST_RESULT)
//#undef NDEBUG
#ifdef NDEBUG
#error "This is not supposed to be included in non-debug mode."
#endif
#include <util/debugUtils.hpp>
#include <util/opticalFlowUtils.hpp>
#endif

namespace VideoStitch {
namespace Testing {

void testMaskInterpolation() {
  std::string workingPath = "";

  std::vector<std::string> maskInterpolationTests;
  for (int i = 0; i <= 0; i++) {
    maskInterpolationTests.push_back(workingPath + "data/maskinterpolation/test" + std::to_string(i) + ".ptv");
  }
  for (int test = 0; test >= 0; test--) {
    std::string ptvFile = maskInterpolationTests[test];
    Potential<Ptv::Parser> parser = Ptv::Parser::create();
    ENSURE(parser.ok());
    // Load the project and parse it
    ENSURE(parser->parse(ptvFile));
    ENSURE(parser->getRoot().has("pano"));
    // Create a runtime panorama from the parsed project.
    std::unique_ptr<Core::PanoDefinition> panoDef(Core::PanoDefinition::create(*parser->getRoot().has("pano")));

    // Prepare image mapping
    VideoStitch::Potential<Core::InputsMap> potInputsMap = Core::InputsMap::create(*panoDef.get());
    ENSURE(potInputsMap.status());

    std::map<readerid_t, Input::VideoReader *> readers;
    ENSURE(prepareFakeReader(*panoDef.get(), readers));

    auto potInputMaskInterpolation = MaskInterpolation::InputMaskInterpolation::create(*panoDef.get(), readers);
    ENSURE(potInputMaskInterpolation.status());
    auto inputsMap = potInputsMap.object();
    std::unique_ptr<MaskInterpolation::InputMaskInterpolation> inputMaskInterpolation(
        potInputMaskInterpolation.release());
    bool loaded = false;
    std::vector<int> timeFrames = {0, 100, 200};
    for (auto frame : timeFrames) {
      std::cout << "*** Test " << test << " - frame " << frame << "" << std::endl;
      inputsMap->loadPrecomputedMap(frame, *panoDef.get(), readers, inputMaskInterpolation, loaded);
      std::stringstream ss;
      ss.str("");
      ss << workingPath + "data/maskinterpolation/test-" << test << "-frame-" << frame << "-lookup.png";
#ifdef DUMP_TEST_RESULT
      Debug::dumpRGBAIndexDeviceBuffer<uint32_t>(ss.str().c_str(), inputsMap->getMask(), panoDef->getWidth(),
                                                 panoDef->getHeight());
#else
      std::vector<uint32_t> hostRGBA(inputsMap->getMask().numElements());
      ENSURE(GPU::memcpyBlocking<uint32_t>(&hostRGBA[0], inputsMap->getMask(), inputsMap->getMask().byteSize()));
      std::vector<unsigned char> data;
      Util::ImageProcessing::convertIndexToRGBA(hostRGBA, data);
      ENSURE_PNG_FILE_AND_RGBA_BUFFER_SIMILARITY(ss.str(), data, 0.01f);
      std::cout << "*** Test " << test << " - frame " << frame << " passed." << std::endl;
#endif
    }
    deleteAllValues(readers);
    std::cout << "*** Test " << test << " passed." << std::endl;
  }
}

}  // namespace Testing
}  // namespace VideoStitch

int main(int argc, char **argv) {
  VideoStitch::Testing::initTest();
  VideoStitch::Testing::ENSURE(VideoStitch::GPU::setDefaultBackendDevice(0));

  VideoStitch::Testing::testMaskInterpolation();

  return 0;
}
