// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "gpu/testing.hpp"

#include "libvideostitch/cameraDef.hpp"
#include "libvideostitch/rigDef.hpp"
#include "libvideostitch/parse.hpp"
#include "libvideostitch/ptv.hpp"

#include <iostream>
#include <memory>

using namespace VideoStitch;

namespace VideoStitch {
namespace Testing {

const std::string jsonCamera1 = R"(
{
  "name" : "testcam1",
  "format" : "ff_fisheye",
  "width" : 1920,
  "height" : 1080,
  "fu_mean" : 500.0,
  "fu_variance" : 1.0,
  "fv_mean" : 600.0,
  "fv_variance" : 2.0,
  "cu_mean" : 320.0,
  "cu_variance" : 3.0,
  "cv_mean" : 240.0,
  "cv_variance" : 4.0,
  "distorta_mean" : 0.1,
  "distorta_variance" : 0.01,
  "distortb_mean" : 0.2,
  "distortb_variance" : 0.02,
  "distortc_mean" : 0.3,
  "distortc_variance" : 0.03
}
)";

const std::string jsonCamera2 = R"(
{
  "name" : "testcam2",
  "format" : "ff_fisheye",
  "width" : 1920,
  "height" : 1080,
  "fu_mean" : 5000.0,
  "fu_variance" : 10.0,
  "fv_mean" : 6000.0,
  "fv_variance" : 20.0,
  "cu_mean" : 3200.0,
  "cu_variance" : 30.0,
  "cv_mean" : 2400.0,
  "cv_variance" : 40.0,
  "distorta_mean" : 0.01,
  "distorta_variance" : 0.001,
  "distortb_mean" : 0.02,
  "distortb_variance" : 0.002,
  "distortc_mean" : 0.03,
  "distortc_variance" : 0.003
}
)";

const std::string jsonRig = R"(
{
  "name" : "testrig1",
  "rigcameras" : [
    {
      "camera" : "testcam1",
      "yaw_mean" : 0.1,
      "roll_mean" : 0.2,
      "pitch_mean" : 0.3,
      "yaw_variance" : 0.01,
      "roll_variance" : 0.02,
      "pitch_variance" : 0.03
    },
    {
      "camera" : "testcam2",
      "yaw_mean" : -0.1,
      "roll_mean" : -0.2,
      "pitch_mean" : -0.3,
      "yaw_variance" : 0.1,
      "roll_variance" : 0.2,
      "pitch_variance" : 0.3
    }
  ]
}
)";

void checkCameraDefinition() {
  Potential<Ptv::Parser> parser = Ptv::Parser::create();
  if (!parser->parseData(jsonCamera1)) {
    std::cerr << parser->getErrorMessage() << std::endl;
    ENSURE(false);
  }

  VideoStitch::Core::CameraDefinition* cdef = VideoStitch::Core::CameraDefinition::create(parser->getRoot());
  if (!cdef) {
    std::cerr << "Invalid camera json" << std::endl;
    ENSURE(false);
  }

  ENSURE_EQ(cdef->getName(), std::string("testcam1"));
  ENSURE_EQ((int)cdef->getType(), (int)VideoStitch::Core::InputDefinition::Format::FullFrameFisheye);
  ENSURE_EQ(cdef->getFu().mean, 500.0);
  ENSURE_EQ(cdef->getFu().variance, 1.0);
  ENSURE_EQ(cdef->getFv().mean, 600.0);
  ENSURE_EQ(cdef->getFv().variance, 2.0);
  ENSURE_EQ(cdef->getCu().mean, 320.0);
  ENSURE_EQ(cdef->getCu().variance, 3.0);
  ENSURE_EQ(cdef->getCv().mean, 240.0);
  ENSURE_EQ(cdef->getCv().variance, 4.0);
  ENSURE_EQ(cdef->getDistortionA().mean, 0.1);
  ENSURE_EQ(cdef->getDistortionA().variance, 0.01);
  ENSURE_EQ(cdef->getDistortionB().mean, 0.2);
  ENSURE_EQ(cdef->getDistortionB().variance, 0.02);
  ENSURE_EQ(cdef->getDistortionC().mean, 0.3);
  ENSURE_EQ(cdef->getDistortionC().variance, 0.03);

  delete cdef;
}

void checkRigDefinition() {
  std::map<std::string, std::shared_ptr<VideoStitch::Core::CameraDefinition>> mapcam;

  {
    Potential<Ptv::Parser> parser = Ptv::Parser::create();
    if (!parser->parseData(jsonCamera1)) {
      std::cerr << parser->getErrorMessage() << std::endl;
      ENSURE(false);
    }

    std::shared_ptr<VideoStitch::Core::CameraDefinition> cdef(
        VideoStitch::Core::CameraDefinition::create(parser->getRoot()));
    if (!cdef.get()) {
      std::cerr << "Invalid camera json" << std::endl;
      ENSURE(false);
    }

    mapcam[cdef->getName()] = cdef;
  }

  {
    Potential<Ptv::Parser> parser = Ptv::Parser::create();
    if (!parser->parseData(jsonCamera2)) {
      std::cerr << parser->getErrorMessage() << std::endl;
      ENSURE(false);
    }

    std::shared_ptr<VideoStitch::Core::CameraDefinition> cdef(
        VideoStitch::Core::CameraDefinition::create(parser->getRoot()));
    if (!cdef.get()) {
      std::cerr << "Invalid camera json" << std::endl;
      ENSURE(false);
    }

    mapcam[cdef->getName()] = cdef;
  }

  {
    Potential<Ptv::Parser> parser = Ptv::Parser::create();
    if (!parser->parseData(jsonRig)) {
      std::cerr << parser->getErrorMessage() << std::endl;
      ENSURE(false);
    }

    std::unique_ptr<VideoStitch::Core::RigDefinition> rigdef(
        VideoStitch::Core::RigDefinition::create(mapcam, parser->getRoot()));
    if (!rigdef.get()) {
      std::cerr << "Invalid rig json" << std::endl;
      ENSURE(false);
    }
  }
}

}  // namespace Testing
}  // namespace VideoStitch

int main() {
  VideoStitch::Testing::initTest();

  VideoStitch::Testing::checkCameraDefinition();
  VideoStitch::Testing::checkRigDefinition();

  return 0;
}
