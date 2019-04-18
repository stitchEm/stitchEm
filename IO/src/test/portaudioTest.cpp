// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "testing_common.hpp"
#include <iostream>
#include <thread>
#include <chrono>
#include "libvideostitch/input.hpp"
#include "libvideostitch/inputDef.hpp"
#include "libvideostitch/inputFactory.hpp"
#include "libvideostitch/plugin.hpp"
#include "libvideostitch/ptv.hpp"
#include "libvideostitch/status.hpp"

namespace VideoStitch {
namespace Testing {

bool paTestReader() {
  initTest();

  Ptv::Value* config = Ptv::Value::emptyObject();
  config->get("type")->asString() = "portaudio";
  Core::InputDefinition* def = Core::InputDefinition::create(*config);
  std::unique_ptr<Input::DefaultReaderFactory> fact(new Input::DefaultReaderFactory(0, NO_LAST_FRAME));
  Potential<Input::Reader> reader = fact->create(0, *def);
  ENSURE(reader.object() != nullptr, "Could not create reader");
  Input::AudioReader* audioReader = dynamic_cast<Input::AudioReader*>(reader.object());

  Audio::Samples samples;
  Status ret = audioReader->readSamples(64, samples);  // First time will start stream
  ENSURE(ret.code() == Code::Ok);
  ENSURE(samples.getNbOfSamples() == 0, "Expected 0 samples on first read");

  std::this_thread::sleep_for((std::chrono::milliseconds)100);

  ret = audioReader->readSamples(64, samples);
  // On this read, we will have samples if a device is available, or a ReaderStarved code if
  // there is no audio available (no device).
  if (ret.code() == Code::Ok) {
    ENSURE(samples.getNbOfSamples() > 0, "Expected to get samples");
  } else {
    ENSURE(ret.code() == Code::ReaderStarved, "Read error");
  }

  return true;
}

}  // namespace Testing
}  // namespace VideoStitch

#if defined(_WIN32)
#define PORTAUDIO_PLUGIN_DIR VS_PLUGIN_DIR_NAME
#else
#define PORTAUDIO_PLUGIN_DIR "../" VS_PLUGIN_DIR_NAME
#endif

int main(int /*argc*/, char** /*argv*/) {
  VideoStitch::Plugin::loadPlugins(PORTAUDIO_PLUGIN_DIR);
  VideoStitch::Testing::ENSURE(VideoStitch::Testing::paTestReader());

  return 0;
}
