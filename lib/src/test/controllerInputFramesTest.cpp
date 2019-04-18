// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "gpu/testing.hpp"

#include "libvideostitch/panoDef.hpp"
#include "libvideostitch/parse.hpp"
#include "libvideostitch/ptv.hpp"

#include <core/controllerInputFrames.cpp>

namespace VideoStitch {
namespace Testing {

// implementation-defined in the checkerboard reader we're using here
// no access to the actual value through ControllerInputFrames
FrameRate checkerBoardFrameRate = {60, 1};

Core::PanoDefinition* getTestPanoDef(std::string proceduralName) {
  Potential<Ptv::Parser> parser(Ptv::Parser::create());

  std::stringstream ss;
  ss << "{"
        " \"width\": 1234, "
        " \"height\": 456, "
        " \"hfov\": 300.0, "
        " \"proj\": \"equirectangular\", "
        " \"inputs\": [ "
        "  { "
        "   \"reader_config\" : \"procedural:"
     << proceduralName
     << "(size=32,color1=000000,color2=ffffff,color3=ffffff)\","
        "   \"width\": 256, "
        "   \"height\": 128, "
        "   \"hfov\": 90.0, "
        "   \"yaw\": 0.0, "
        "   \"pitch\": 0.0, "
        "   \"roll\": 0.0, "
        "   \"proj\": \"rectilinear\", "
        "   \"viewpoint_model\": \"ptgui\", "
        "   \"response\": \"linear\", "
        "   \"filename\": \"\" "
        "  }, "
        "  { "
        "   \"reader_config\" : \"procedural:"
     << proceduralName
     << "(size=32,color1=000000,color2=ffffff,color3=ffffff)\","
        "   \"width\": 256, "
        "   \"height\": 128, "
        "   \"hfov\": 90.0, "
        "   \"yaw\": 0.0, "
        "   \"pitch\": 0.0, "
        "   \"roll\": 0.0, "
        "   \"proj\": \"rectilinear\", "
        "   \"viewpoint_model\": \"ptgui\", "
        "   \"response\": \"linear\", "
        "   \"filename\": \"\" "
        "  } "
        " ]"
        "}";

  if (!parser->parseData(ss.str())) {
    std::cerr << parser->getErrorMessage() << std::endl;
    ENSURE(false, "could not parse");
    return NULL;
  }
  std::unique_ptr<Core::PanoDefinition> panoDef(Core::PanoDefinition::create(parser->getRoot()));
  ENSURE((bool)panoDef);
  return panoDef.release();
}

template <typename readbackType>
void checkFrameStatus(std::map<readerid_t, PotentialValue<GPU::HostBuffer<readbackType>>> frames) {
  for (auto potFrame : frames) {
    ENSURE(potFrame.second.status());
  }
}

template <PixelFormat destinationColor, typename readbackType>
void testLoadAndSeek() {
  // test with movingChecker procedural reader, which reports correct time stamp after seeking
  std::unique_ptr<Core::PanoDefinition> panoDef(getTestPanoDef("movingChecker"));
  auto controller = Core::ControllerInputFrames<destinationColor, readbackType>::create(panoDef.get());
  ENSURE(controller.status());

  std::map<readerid_t, PotentialValue<GPU::HostBuffer<readbackType>>> frames;

  double frameTime = 1000 * 1000 * (double)checkerBoardFrameRate.den / checkerBoardFrameRate.num;

  // load / getCurrentFrame
  mtime_t loadDate = 0;
  controller->load(frames, &loadDate);
  checkFrameStatus(frames);
  ENSURE_EQ((mtime_t)round(0 * frameTime), loadDate, "Should have loaded first frame");
  frames.clear();

  controller->load(frames, &loadDate);
  checkFrameStatus(frames);
  ENSURE_EQ((mtime_t)round(1 * frameTime), loadDate, "Should have loaded second frame");
  frames.clear();

  controller->load(frames, &loadDate);
  checkFrameStatus(frames);
  ENSURE_EQ((mtime_t)round(2 * frameTime), loadDate, "Should have loaded third frame");
  frames.clear();

  ENSURE(controller->seek(12345));

  controller->load(frames, &loadDate);
  checkFrameStatus(frames);
  ENSURE_EQ((mtime_t)round(12345 * frameTime), loadDate, "Should have loaded frame which was seeked to");
  frames.clear();

  controller->load(frames, &loadDate);
  checkFrameStatus(frames);
  ENSURE_EQ((mtime_t)round(12346 * frameTime), loadDate, "Should have loaded frame following seek");
  frames.clear();
}

template <PixelFormat destinationColor, typename readbackType>
void testUnknownReader() {
  // test with movingChecker procedural reader, which reports correct time stamp after seeking
  std::unique_ptr<Core::PanoDefinition> panoDef(getTestPanoDef("readerNameThatDoesntExist"));
  auto controller = Core::ControllerInputFrames<destinationColor, readbackType>::create(panoDef.get());
  ENSURE(!controller.ok(), "ControllerInputFrames should not have set up correctly");

  // make sure there are no leaks and crashed when destroying incomplete ControllerInputFrames
}

// TODO ensure color space conversion is done correctly in ControllerInputFrames
// can't test with current set of procedural readers, which report frameDataSize = 1
// which breaks ControllerInputFrames

// void testColorSpaceConversion() {
// }

}  // namespace Testing
}  // namespace VideoStitch

int main() {
  VideoStitch::Testing::ENSURE(VideoStitch::GPU::setDefaultBackendDevice(0));

  VideoStitch::Testing::testLoadAndSeek<VideoStitch::PixelFormat::Grayscale, unsigned char>();
  VideoStitch::Testing::testLoadAndSeek<VideoStitch::PixelFormat::RGBA, uint32_t>();

  VideoStitch::Testing::testUnknownReader<VideoStitch::PixelFormat::Grayscale, unsigned char>();
  VideoStitch::Testing::testUnknownReader<VideoStitch::PixelFormat::RGBA, uint32_t>();

  return 0;
}
