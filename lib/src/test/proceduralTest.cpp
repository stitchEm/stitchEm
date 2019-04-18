// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "gpu/testing.hpp"

#include <input/proceduralParser.hpp>

namespace VideoStitch {
namespace Testing {

void testNotProcedural() {
  Input::ProceduralInputSpec spec("nothing");
  ENSURE(!spec.isProcedural());
}

void testProceduralOneOption() {
  Input::ProceduralInputSpec spec("procedural:toto(titi=3)");
  ENSURE(spec.isProcedural());
  ENSURE_EQ(std::string("toto"), spec.getName());
  ENSURE(spec.getOption("titi"));
  ENSURE_EQ(std::string("3"), *spec.getOption("titi"));
}

void testProcedural() {
  Input::ProceduralInputSpec spec("procedural:toto(titi=3,tata=3.14,tutu=truc)");
  ENSURE(spec.isProcedural());
  ENSURE_EQ(std::string("toto"), spec.getName());
  ENSURE(spec.getOption("titi"));
  ENSURE_EQ(std::string("3"), *spec.getOption("titi"));
  ENSURE(spec.getOption("tata"));
  ENSURE_EQ(std::string("3.14"), *spec.getOption("tata"));
  ENSURE(spec.getOption("tutu"));
  ENSURE_EQ(std::string("truc"), *spec.getOption("tutu"));
}

void testProceduralNoOptions() {
  Input::ProceduralInputSpec spec("procedural:toto");
  ENSURE(spec.isProcedural());
  ENSURE_EQ(std::string("toto"), spec.getName());
}
}  // namespace Testing
}  // namespace VideoStitch

int main() {
  VideoStitch::Testing::initTest();

  VideoStitch::Testing::testNotProcedural();
  VideoStitch::Testing::testProceduralOneOption();
  VideoStitch::Testing::testProcedural();
  VideoStitch::Testing::testProceduralNoOptions();
  return 0;
}
