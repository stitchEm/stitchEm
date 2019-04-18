// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "gpu/testing.hpp"

#include "libvideostitch/parse.hpp"
#include "libvideostitch/panoDef.hpp"

#include "exposure/metadataProcessor.hpp"

namespace VideoStitch {
namespace Testing {

static const int NUM_INPUTS{4};

std::unique_ptr<Core::PanoDefinition> getTestPanoDef() {
  Potential<Ptv::Parser> parser(Ptv::Parser::create());
  if (!parser->parseData("{"
                         " \"width\": 513, "
                         " \"height\": 315, "
                         " \"hfov\": 90.0, "
                         " \"proj\": \"rectilinear\", "
                         " \"inputs\": [ "
                         "  { "
                         "   \"width\": 17, "
                         "   \"height\": 13, "
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
                         "   \"width\": 17, "
                         "   \"height\": 13, "
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
                         "   \"width\": 17, "
                         "   \"height\": 13, "
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
                         "   \"width\": 17, "
                         "   \"height\": 13, "
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
                         "}")) {
    std::cerr << parser->getErrorMessage() << std::endl;
    ENSURE(false, "could not parse");
    return NULL;
  }
  std::unique_ptr<Core::PanoDefinition> panoDef(Core::PanoDefinition::create(parser->getRoot()));
  ENSURE((bool)panoDef);
  return panoDef;
}

const std::array<uint16_t, 256>& getLinearToneCurve() {
  static std::array<uint16_t, 256> linear = []() -> std::array<uint16_t, 256> {
    std::array<uint16_t, 256> values;
    for (uint16_t i = 0; i < 256; i++) {
      values[i] = i;
    }
    return values;
  }();
  return linear;
}

// a linear metadata uint16_t tone curve applied on top the Orah 4i camera response
const std::array<uint16_t, 256>& getOrahLinearCurve() {
  static std::array<uint16_t, 256> values = {
      {0,   4,   7,   11,  15,  18,  21,  25,  28,  31,  35,  38,  41,  44,  47,  50,  53,  56,  58,  61,  64,  66,
       69,  72,  74,  77,  79,  82,  84,  86,  89,  91,  93,  95,  98,  100, 102, 104, 106, 108, 110, 112, 114, 116,
       118, 120, 121, 123, 125, 127, 128, 130, 132, 133, 135, 137, 138, 140, 141, 143, 144, 146, 147, 149, 150, 152,
       153, 154, 156, 157, 158, 160, 161, 162, 163, 164, 166, 167, 168, 169, 170, 171, 173, 174, 175, 176, 177, 178,
       179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 188, 189, 190, 191, 192, 193, 194, 194, 195, 196, 197, 198,
       198, 199, 200, 201, 202, 202, 203, 204, 204, 205, 206, 206, 207, 208, 208, 209, 210, 210, 211, 212, 212, 213,
       213, 214, 215, 215, 216, 216, 217, 218, 218, 219, 219, 220, 220, 221, 221, 222, 222, 223, 223, 224, 224, 225,
       225, 226, 226, 227, 227, 227, 228, 228, 229, 229, 230, 230, 230, 231, 231, 232, 232, 232, 233, 233, 234, 234,
       234, 235, 235, 236, 236, 236, 237, 237, 237, 238, 238, 238, 239, 239, 239, 240, 240, 240, 241, 241, 241, 241,
       242, 242, 242, 243, 243, 243, 244, 244, 244, 244, 245, 245, 245, 245, 246, 246, 246, 246, 247, 247, 247, 247,
       248, 248, 248, 248, 249, 249, 249, 249, 250, 250, 250, 250, 250, 251, 251, 251, 251, 252, 252, 252, 252, 252,
       253, 253, 253, 253, 253, 254, 254, 254, 254, 254, 254, 255, 255, 255}};
  return values;
}

std::array<uint16_t, 256> getGammaToneCurve(double gamma) {
  std::array<uint16_t, 256> values;
  for (uint16_t i = 0; i < 256; i++) {
    values[i] = static_cast<uint16_t>(round(pow(i / 255.f, gamma) * 255.f));
  }
  return values;
}

// a gamma metadata uint16_t tone curve applied on top the Orah 4i camera response
template <int gamma10>
std::array<uint16_t, 256>& getOrahGammaToneCurve();

template <>
std::array<uint16_t, 256>& getOrahGammaToneCurve<12>() {
  // libvideostitch has Orah 4i curve as float values
  // curve = np.array([ ... values from Gamma.gma.mpg ...]) / 1023 * 255
  //
  // The gamma 1.2 metadata tone curve is passed in uint16_t values --> round
  // gamma_1_2_u16 = np.rint((np.arange(256) / 255.) ** 1.2 * 255)
  // Result should be interpolated properly, then rounded to uint16_t again
  // values = np.round(np.interp(curve, np.arange(256), gamma_1_2_u16))
  static std::array<uint16_t, 256> values = {
      {0,   2,   3,   6,   9,   11,  13,  16,  18,  20,  24,  26,  28,  31,  34,  36,  39,  41,  43,  46,  49,  50,
       53,  56,  58,  61,  62,  65,  67,  69,  72,  74,  76,  78,  81,  83,  85,  87,  89,  91,  93,  95,  97,  99,
       101, 103, 104, 106, 108, 110, 112, 114, 116, 117, 119, 121, 122, 124, 125, 127, 129, 131, 132, 134, 135, 137,
       138, 139, 141, 142, 144, 146, 147, 148, 149, 150, 152, 153, 155, 156, 157, 158, 160, 161, 162, 163, 165, 166,
       167, 168, 169, 170, 171, 172, 173, 174, 176, 177, 177, 178, 179, 180, 181, 183, 184, 184, 185, 186, 187, 188,
       188, 190, 191, 192, 193, 193, 194, 195, 195, 196, 197, 198, 199, 200, 200, 201, 202, 202, 203, 204, 204, 205,
       206, 207, 208, 208, 209, 209, 210, 211, 211, 212, 212, 213, 214, 215, 215, 216, 216, 217, 217, 218, 218, 219,
       219, 220, 221, 222, 222, 222, 223, 223, 224, 224, 225, 225, 225, 226, 227, 227, 228, 228, 229, 229, 230, 230,
       230, 231, 231, 232, 232, 232, 233, 234, 234, 235, 235, 235, 236, 236, 236, 237, 237, 237, 238, 238, 238, 238,
       239, 239, 240, 240, 241, 241, 242, 242, 242, 242, 243, 243, 243, 243, 244, 244, 244, 244, 245, 245, 245, 246,
       246, 247, 247, 247, 248, 248, 248, 248, 249, 249, 249, 249, 249, 250, 250, 250, 250, 251, 251, 251, 251, 252,
       252, 253, 253, 253, 253, 254, 254, 254, 254, 254, 254, 255, 255, 255}};
  return values;
}

template <>
std::array<uint16_t, 256>& getOrahGammaToneCurve<15>() {
  // as getOrahGammaToneCurve<12>(), but with ** 1.5
  static std::array<uint16_t, 256> values = {
      {0,   1,   1,   2,   4,   5,   6,   8,   9,   11,  13,  15,  16,  18,  20,  22,  24,  26,  28,  30,  32,  34,
       36,  38,  40,  42,  44,  46,  48,  50,  53,  54,  56,  58,  61,  63,  65,  66,  68,  70,  72,  74,  76,  78,
       80,  82,  83,  85,  88,  90,  91,  93,  95,  96,  98,  100, 102, 104, 105, 107, 108, 110, 112, 114, 115, 117,
       119, 120, 122, 123, 124, 127, 128, 129, 130, 132, 134, 135, 136, 138, 139, 140, 142, 143, 145, 146, 147, 149,
       150, 151, 152, 154, 155, 156, 158, 159, 160, 161, 162, 163, 164, 165, 167, 168, 169, 170, 171, 172, 173, 174,
       175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 188, 189, 191, 191, 192, 193, 194, 195,
       195, 196, 197, 198, 199, 199, 200, 201, 202, 203, 203, 204, 204, 205, 206, 207, 208, 209, 209, 210, 210, 211,
       211, 212, 213, 214, 214, 215, 216, 216, 217, 217, 218, 218, 219, 220, 220, 221, 221, 222, 223, 223, 224, 224,
       225, 226, 226, 227, 227, 227, 228, 228, 229, 229, 230, 230, 231, 231, 232, 232, 233, 233, 234, 234, 234, 235,
       236, 236, 236, 237, 237, 237, 238, 239, 239, 239, 240, 240, 240, 241, 241, 242, 242, 242, 243, 243, 243, 244,
       244, 245, 245, 245, 246, 246, 246, 247, 247, 248, 248, 248, 248, 249, 249, 249, 250, 250, 250, 251, 251, 251,
       252, 252, 252, 252, 253, 253, 253, 254, 254, 254, 254, 255, 255, 255}};
  return values;
}

template <>
std::array<uint16_t, 256>& getOrahGammaToneCurve<9>() {
  // as getOrahGammaToneCurve<12>(), but with ** 0.9
  static std::array<uint16_t, 256> values = {
      {0,   6,   10,  15,  20,  23,  27,  32,  35,  38,  43,  46,  49,  52,  56,  59,  62,  65,  67,  70,  73,  76,
       79,  82,  84,  87,  89,  92,  94,  96,  99,  101, 103, 105, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126,
       127, 129, 130, 132, 134, 136, 137, 139, 141, 142, 144, 146, 147, 149, 150, 152, 152, 154, 155, 157, 158, 160,
       161, 162, 164, 165, 166, 168, 169, 170, 170, 171, 173, 174, 175, 176, 177, 178, 180, 181, 182, 183, 184, 185,
       185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 194, 195, 196, 197, 198, 198, 199, 199, 200, 201, 202, 203,
       203, 204, 205, 206, 207, 207, 208, 209, 209, 210, 210, 210, 211, 212, 212, 213, 214, 214, 215, 216, 216, 217,
       217, 218, 219, 219, 220, 220, 221, 221, 221, 222, 222, 223, 223, 224, 224, 225, 225, 226, 226, 227, 227, 228,
       228, 229, 229, 230, 230, 230, 231, 231, 231, 231, 232, 232, 232, 233, 233, 234, 234, 234, 235, 235, 236, 236,
       236, 237, 237, 238, 238, 238, 239, 239, 239, 240, 240, 240, 241, 241, 241, 241, 241, 241, 242, 242, 242, 242,
       243, 243, 243, 244, 244, 244, 245, 245, 245, 245, 246, 246, 246, 246, 247, 247, 247, 247, 248, 248, 248, 248,
       249, 249, 249, 249, 250, 250, 250, 250, 250, 250, 250, 250, 250, 251, 251, 251, 251, 252, 252, 252, 252, 252,
       253, 253, 253, 253, 253, 254, 254, 254, 254, 254, 254, 255, 255, 255}};
  return values;
}

void testAppendToneCurve() {
  std::unique_ptr<Core::PanoDefinition> panoDef = getTestPanoDef();
  ENSURE(panoDef->numVideoInputs() == NUM_INPUTS);

  FrameRate frameRate{30, 1};

  std::vector<std::pair<int, GPU::Buffer<const uint32_t>>> frames{{0, {}}};

  // ensure inputs are currently linear response
  for (const auto& videoInput : panoDef->getVideoInputs()) {
    ENSURE(videoInput.get().getPhotoResponse() == Core::InputDefinition::PhotoResponse::LinearResponse);
  }

  Exposure::MetadataProcessor mp;

  auto applyToneCurveToPanoAtFrame = [&mp, &frameRate, &panoDef](frameid_t currentStitchingFrame,
                                                                 const Input::MetadataChunk& metadata) {
    std::unique_ptr<Core::PanoDefinition> updated =
        mp.createUpdatedPano(metadata, *panoDef, frameRate, currentStitchingFrame);
    if (updated) {
      panoDef.reset(updated.release());
    }
  };

  auto updatePanoNoNewData = [&mp, &frameRate, &panoDef](frameid_t currentStitchingFrame) {
    std::unique_ptr<Core::PanoDefinition> updated =
        mp.createUpdatedPano({}, *panoDef, frameRate, currentStitchingFrame);
    if (updated) {
      panoDef.reset(updated.release());
    }
  };

  // ensure inputs are still linear response (tone curve arrives at frame 1000)
  for (const auto& videoInput : panoDef->getVideoInputs()) {
    ENSURE(videoInput.get().getPhotoResponse() == Core::InputDefinition::PhotoResponse::LinearResponse);
  }

  {
    frameid_t toneCurveDataFrame{1000};
    mtime_t toneCurveDataTime = frameRate.frameToTimestamp(toneCurveDataFrame);

    Metadata::ToneCurve toneCurve{toneCurveDataTime, getLinearToneCurve()};

    Input::MetadataChunk metadata;
    metadata.toneCurve.push_back({{0, toneCurve}});

    applyToneCurveToPanoAtFrame(999, metadata);
  }

  // ensure inputs are still linear response (tone curve arrives at frame 1000)
  for (const auto& videoInput : panoDef->getVideoInputs()) {
    ENSURE(videoInput.get().getPhotoResponse() == Core::InputDefinition::PhotoResponse::LinearResponse);
  }

  updatePanoNoNewData(999);

  // ensure inputs are still linear response (tone curve arrives at frame 1000)
  for (const auto& videoInput : panoDef->getVideoInputs()) {
    ENSURE(videoInput.get().getPhotoResponse() == Core::InputDefinition::PhotoResponse::LinearResponse);
  }

  updatePanoNoNewData(1000);
  // ensure input are now in curve response
  {
    videoreaderid_t videoInputID = 0;
    for (const auto& videoInput : panoDef->getVideoInputs()) {
      auto expectedResponse = videoInputID == 0 ? Core::InputDefinition::PhotoResponse::CurveResponse
                                                : Core::InputDefinition::PhotoResponse::LinearResponse;
      ENSURE(videoInput.get().getPhotoResponse() == expectedResponse);
      videoInputID++;
    }
  }
  ENSURE(panoDef->getVideoInput(0).getValueBasedResponseCurve() != nullptr, "Should have value based tone curve");
  ENSURE_EQ(*panoDef->getVideoInput(0).getValueBasedResponseCurve(), getOrahLinearCurve(),
            "Tone curve was set with linear values");
}

void testAppendToneCurveMoreSensors() {
  std::unique_ptr<Core::PanoDefinition> panoDef = getTestPanoDef();

  FrameRate frameRate{50, 1};

  std::vector<std::pair<int, GPU::Buffer<const uint32_t>>> frames{{0, {}}};

  Exposure::MetadataProcessor mp;

  auto applyToneCurveToPanoAtFrame = [&mp, &frameRate, &panoDef](frameid_t currentStitchingFrame,
                                                                 const Input::MetadataChunk& metadata) {
    std::unique_ptr<Core::PanoDefinition> updated =
        mp.createUpdatedPano(metadata, *panoDef, frameRate, currentStitchingFrame);
    if (updated) {
      panoDef.reset(updated.release());
    }
  };

  auto updatePanoNoNewData = [&mp, &frameRate, &panoDef](frameid_t currentStitchingFrame) {
    std::unique_ptr<Core::PanoDefinition> updated =
        mp.createUpdatedPano({}, *panoDef, frameRate, currentStitchingFrame);
    if (updated) {
      panoDef.reset(updated.release());
    }
  };

  // add tone curves for several sensors at once
  {
    Input::MetadataChunk metadata;
    metadata.toneCurve.push_back({{0, {frameRate.frameToTimestamp(1000), getGammaToneCurve(1.2)}},
                                  {1, {frameRate.frameToTimestamp(1001), getGammaToneCurve(1.5)}},
                                  {2, {frameRate.frameToTimestamp(1500), getGammaToneCurve(0.9)}},
                                  {3, {frameRate.frameToTimestamp(2000), getLinearToneCurve()}}});

    applyToneCurveToPanoAtFrame(0, metadata);
  }

  ENSURE(panoDef->getVideoInput(0).getValueBasedResponseCurve() == nullptr, "Frame of tone curve not reached yet");
  ENSURE(panoDef->getVideoInput(1).getValueBasedResponseCurve() == nullptr, "Frame of tone curve not reached yet");
  ENSURE(panoDef->getVideoInput(2).getValueBasedResponseCurve() == nullptr, "Frame of tone curve not reached yet");
  ENSURE(panoDef->getVideoInput(3).getValueBasedResponseCurve() == nullptr, "Frame of tone curve not reached yet");

  updatePanoNoNewData(1000);

  ENSURE_EQ(*panoDef->getVideoInput(0).getValueBasedResponseCurve(), getOrahGammaToneCurve<12>(), "Tone curve data");
  ENSURE(panoDef->getVideoInput(1).getValueBasedResponseCurve() == nullptr, "Frame of tone curve not reached yet");
  ENSURE(panoDef->getVideoInput(2).getValueBasedResponseCurve() == nullptr, "Frame of tone curve not reached yet");
  ENSURE(panoDef->getVideoInput(3).getValueBasedResponseCurve() == nullptr, "Frame of tone curve not reached yet");

  updatePanoNoNewData(1001);

  ENSURE_EQ(*panoDef->getVideoInput(0).getValueBasedResponseCurve(), getOrahGammaToneCurve<12>(), "Tone curve data");
  ENSURE_EQ(*panoDef->getVideoInput(1).getValueBasedResponseCurve(), getOrahGammaToneCurve<15>(), "Tone curve data");
  ENSURE(panoDef->getVideoInput(2).getValueBasedResponseCurve() == nullptr, "Frame of tone curve not reached yet");
  ENSURE(panoDef->getVideoInput(3).getValueBasedResponseCurve() == nullptr, "Frame of tone curve not reached yet");

  updatePanoNoNewData(1500);

  ENSURE_EQ(*panoDef->getVideoInput(0).getValueBasedResponseCurve(), getOrahGammaToneCurve<12>(), "Tone curve data");
  ENSURE_EQ(*panoDef->getVideoInput(1).getValueBasedResponseCurve(), getOrahGammaToneCurve<15>(), "Tone curve data");
  ENSURE_EQ(*panoDef->getVideoInput(2).getValueBasedResponseCurve(), getOrahGammaToneCurve<9>(), "Tone curve data");
  ENSURE(panoDef->getVideoInput(3).getValueBasedResponseCurve() == nullptr, "Frame of tone curve not reached yet");

  updatePanoNoNewData(2000);

  ENSURE_EQ(*panoDef->getVideoInput(0).getValueBasedResponseCurve(), getOrahGammaToneCurve<12>(), "Tone curve data");
  ENSURE_EQ(*panoDef->getVideoInput(1).getValueBasedResponseCurve(), getOrahGammaToneCurve<15>(), "Tone curve data");
  ENSURE_EQ(*panoDef->getVideoInput(2).getValueBasedResponseCurve(), getOrahGammaToneCurve<9>(), "Tone curve data");
  ENSURE_EQ(*panoDef->getVideoInput(3).getValueBasedResponseCurve(), getOrahLinearCurve(), "Tone curve data");

  updatePanoNoNewData(9999);

  ENSURE_EQ(*panoDef->getVideoInput(0).getValueBasedResponseCurve(), getOrahGammaToneCurve<12>(), "Tone curve data");
  ENSURE_EQ(*panoDef->getVideoInput(1).getValueBasedResponseCurve(), getOrahGammaToneCurve<15>(), "Tone curve data");
  ENSURE_EQ(*panoDef->getVideoInput(2).getValueBasedResponseCurve(), getOrahGammaToneCurve<9>(), "Tone curve data");
  ENSURE_EQ(*panoDef->getVideoInput(3).getValueBasedResponseCurve(), getOrahLinearCurve(), "Tone curve data");
}

void testPruning() {
  std::unique_ptr<Core::PanoDefinition> panoDef = getTestPanoDef();

  FrameRate frameRate{1001, 1000};

  std::vector<std::pair<int, GPU::Buffer<const uint32_t>>> frames{{0, {}}};

  Exposure::MetadataProcessor mp;

  auto applyToneCurveToPanoAtFrame = [&mp, &frameRate, &panoDef](frameid_t currentStitchingFrame,
                                                                 const Input::MetadataChunk& metadata) {
    std::unique_ptr<Core::PanoDefinition> updated =
        mp.createUpdatedPano(metadata, *panoDef, frameRate, currentStitchingFrame);
    if (updated) {
      panoDef.reset(updated.release());
    }
  };

  auto updatePanoNoNewData = [&mp, &frameRate, &panoDef](frameid_t currentStitchingFrame) {
    std::unique_ptr<Core::PanoDefinition> updated =
        mp.createUpdatedPano({}, *panoDef, frameRate, currentStitchingFrame);
    if (updated) {
      panoDef.reset(updated.release());
    }
  };

  // add tone curves for several sensors at once
  {
    Input::MetadataChunk metadata;
    metadata.toneCurve.push_back({{0, {frameRate.frameToTimestamp(1000), getGammaToneCurve(1.2)}},
                                  {1, {frameRate.frameToTimestamp(1000), getGammaToneCurve(1.2)}},
                                  {2, {frameRate.frameToTimestamp(1000), getGammaToneCurve(1.2)}},
                                  {3, {frameRate.frameToTimestamp(1000), getGammaToneCurve(1.2)}}});

    applyToneCurveToPanoAtFrame(0, metadata);
  }

  ENSURE(panoDef->getVideoInput(0).getValueBasedResponseCurve() == nullptr, "Frame of tone curve not reached yet");
  ENSURE(panoDef->getVideoInput(1).getValueBasedResponseCurve() == nullptr, "Frame of tone curve not reached yet");
  ENSURE(panoDef->getVideoInput(2).getValueBasedResponseCurve() == nullptr, "Frame of tone curve not reached yet");
  ENSURE(panoDef->getVideoInput(3).getValueBasedResponseCurve() == nullptr, "Frame of tone curve not reached yet");

  updatePanoNoNewData(1000);

  ENSURE_EQ(*panoDef->getVideoInput(0).getValueBasedResponseCurve(), getOrahGammaToneCurve<12>(), "Tone curve data");
  ENSURE_EQ(*panoDef->getVideoInput(1).getValueBasedResponseCurve(), getOrahGammaToneCurve<12>(), "Tone curve data");
  ENSURE_EQ(*panoDef->getVideoInput(2).getValueBasedResponseCurve(), getOrahGammaToneCurve<12>(), "Tone curve data");
  ENSURE_EQ(*panoDef->getVideoInput(3).getValueBasedResponseCurve(), getOrahGammaToneCurve<12>(), "Tone curve data");

  {
    Input::MetadataChunk metadata;
    metadata.toneCurve.push_back({{0, {frameRate.frameToTimestamp(2000), getGammaToneCurve(1.5)}},
                                  {1, {frameRate.frameToTimestamp(2000), getGammaToneCurve(1.5)}},
                                  {2, {frameRate.frameToTimestamp(2000), getGammaToneCurve(1.5)}},
                                  {3, {frameRate.frameToTimestamp(2000), getGammaToneCurve(1.5)}}});

    applyToneCurveToPanoAtFrame(1001, metadata);
  }

  ENSURE_EQ(*panoDef->getVideoInput(0).getValueBasedResponseCurve(), getOrahGammaToneCurve<12>(), "Tone curve data");
  ENSURE_EQ(*panoDef->getVideoInput(1).getValueBasedResponseCurve(), getOrahGammaToneCurve<12>(), "Tone curve data");
  ENSURE_EQ(*panoDef->getVideoInput(2).getValueBasedResponseCurve(), getOrahGammaToneCurve<12>(), "Tone curve data");
  ENSURE_EQ(*panoDef->getVideoInput(3).getValueBasedResponseCurve(), getOrahGammaToneCurve<12>(), "Tone curve data");

  updatePanoNoNewData(2000);

  ENSURE_EQ(*panoDef->getVideoInput(0).getValueBasedResponseCurve(), getOrahGammaToneCurve<15>(), "Tone curve data");
  ENSURE_EQ(*panoDef->getVideoInput(1).getValueBasedResponseCurve(), getOrahGammaToneCurve<15>(), "Tone curve data");
  ENSURE_EQ(*panoDef->getVideoInput(2).getValueBasedResponseCurve(), getOrahGammaToneCurve<15>(), "Tone curve data");
  ENSURE_EQ(*panoDef->getVideoInput(3).getValueBasedResponseCurve(), getOrahGammaToneCurve<15>(), "Tone curve data");

  {
    Input::MetadataChunk metadata;
    metadata.toneCurve.push_back({{0, {frameRate.frameToTimestamp(3000), getGammaToneCurve(0.9)}},
                                  {1, {frameRate.frameToTimestamp(3000), getGammaToneCurve(0.9)}},
                                  {2, {frameRate.frameToTimestamp(3000), getGammaToneCurve(0.9)}},
                                  {3, {frameRate.frameToTimestamp(3000), getGammaToneCurve(0.9)}}});

    applyToneCurveToPanoAtFrame(2500, metadata);
  }

  ENSURE_EQ(*panoDef->getVideoInput(0).getValueBasedResponseCurve(), getOrahGammaToneCurve<15>(), "Tone curve data");
  ENSURE_EQ(*panoDef->getVideoInput(1).getValueBasedResponseCurve(), getOrahGammaToneCurve<15>(), "Tone curve data");
  ENSURE_EQ(*panoDef->getVideoInput(2).getValueBasedResponseCurve(), getOrahGammaToneCurve<15>(), "Tone curve data");
  ENSURE_EQ(*panoDef->getVideoInput(3).getValueBasedResponseCurve(), getOrahGammaToneCurve<15>(), "Tone curve data");

  updatePanoNoNewData(3000);

  ENSURE_EQ(*panoDef->getVideoInput(0).getValueBasedResponseCurve(), getOrahGammaToneCurve<9>(), "Tone curve data");
  ENSURE_EQ(*panoDef->getVideoInput(1).getValueBasedResponseCurve(), getOrahGammaToneCurve<9>(), "Tone curve data");
  ENSURE_EQ(*panoDef->getVideoInput(2).getValueBasedResponseCurve(), getOrahGammaToneCurve<9>(), "Tone curve data");
  ENSURE_EQ(*panoDef->getVideoInput(3).getValueBasedResponseCurve(), getOrahGammaToneCurve<9>(), "Tone curve data");

  updatePanoNoNewData(1000);

  // do we drop data at some point?
  //
  // tests somewhat undefined behavior, may need to adapt test to implementation details
  // we expect that the metadata for time 1000 has been dropped at this point
  // and thus the tone curves are not updated when requesting pano def at frame 1000
  // if they were still available, gamma should be 1.2
  ENSURE_EQ(*panoDef->getVideoInput(0).getValueBasedResponseCurve(), getOrahGammaToneCurve<9>(), "Tone curve data");
  ENSURE_EQ(*panoDef->getVideoInput(1).getValueBasedResponseCurve(), getOrahGammaToneCurve<9>(), "Tone curve data");
  ENSURE_EQ(*panoDef->getVideoInput(2).getValueBasedResponseCurve(), getOrahGammaToneCurve<9>(), "Tone curve data");
  ENSURE_EQ(*panoDef->getVideoInput(3).getValueBasedResponseCurve(), getOrahGammaToneCurve<9>(), "Tone curve data");
}

void benchmarkAddingData(int iterations) {
  std::unique_ptr<Core::PanoDefinition> panoDef = getTestPanoDef();

  FrameRate frameRate{25, 1};

  std::vector<std::pair<int, GPU::Buffer<const uint32_t>>> frames{{0, {}}};

  Exposure::MetadataProcessor mp;

  auto applyToneCurveToPanoAtFrame = [&mp, &frameRate, &panoDef](frameid_t currentStitchingFrame,
                                                                 const Input::MetadataChunk& metadata) {
    std::unique_ptr<Core::PanoDefinition> updated =
        mp.createUpdatedPano(metadata, *panoDef, frameRate, currentStitchingFrame);
    if (updated) {
      panoDef.reset(updated.release());
    }
  };

  auto updatePanoNoNewData = [&mp, &frameRate, &panoDef](frameid_t currentStitchingFrame) {
    std::unique_ptr<Core::PanoDefinition> updated =
        mp.createUpdatedPano({}, *panoDef, frameRate, currentStitchingFrame);
    if (updated) {
      panoDef.reset(updated.release());
    }
  };

  for (int i = 0; i < iterations; i++) {
    Input::MetadataChunk metadata;
    metadata.toneCurve.push_back({{0, {frameRate.frameToTimestamp(i * 2), getGammaToneCurve(1.2)}},
                                  {1, {frameRate.frameToTimestamp(i * 2), getGammaToneCurve(1.5)}},
                                  {2, {frameRate.frameToTimestamp(i * 2), getGammaToneCurve(0.9)}},
                                  {3, {frameRate.frameToTimestamp(i * 2), getLinearToneCurve()}}});

    applyToneCurveToPanoAtFrame(i * 2, metadata);
    updatePanoNoNewData(i * 2 + 1);
  }
}

}  // namespace Testing
}  // namespace VideoStitch

int main() {
  VideoStitch::Testing::initTest();

  VideoStitch::Testing::testAppendToneCurve();

  VideoStitch::Testing::testAppendToneCurveMoreSensors();

  VideoStitch::Testing::testPruning();

  VideoStitch::Testing::benchmarkAddingData(10);

  return 0;
}
