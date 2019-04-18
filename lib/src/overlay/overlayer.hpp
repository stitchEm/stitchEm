// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "libvideostitch/overlay.hpp"

#include "common/angles.hpp"
#include "core/transformGeoParams.hpp"
#include "gpu/allocator.hpp"
#include "overlay/oglShader.hpp"
#include "parse/json.hpp"
#include "util/pngutil.hpp"

#include "libvideostitch/parse.hpp"
#include "libvideostitch/panoDef.hpp"
#include "libvideostitch/frame.hpp"

#include <iostream>
#include <locale.h>
#include <math.h>
#include <memory>
#include <string>
#include <vector>

#if _MSC_VER
// To disable warnings on the external glm library
#pragma warning(push)
#pragma warning(disable : 4201)
#pragma warning(disable : 4310)
#include <glm/ext.hpp>
#include <glm/mat4x4.hpp>
#include <glm/gtc/matrix_transform.hpp>
#pragma warning(pop)
#else
#include <glm/ext.hpp>
#include <glm/mat4x4.hpp>
#include <glm/gtc/matrix_transform.hpp>
#endif

namespace VideoStitch {
namespace Input {
class VideoReader;
}

namespace GPU {

struct PanoParams {
  int64_t panoTexWidth;
  int64_t panoTexHeight;
  FrameRate panoFrameRate;
  frameid_t panoFrameId;
};

class OverlayTex {
 public:
  ~OverlayTex();

  int cubemapFboId = -1;
  int cubemapTexId = -1;

  int equirectFboId = -1;
  int equirectTexId = -1;
  int equirectPboId = -1;

  int textFboId = -1;

  int panoSrcTexId = -1;
  int panoSrcPboId = -1;

  std::vector<std::pair<const Core::OverlayInputDefinition*, GLuint>> inputTexIds;

  mtime_t date = 0;

  int64_t panoSrcTexWidth = 0;
  int64_t panoSrcTexHeight = 0;
};

class VS_EXPORT Overlayer::Pimpl {
 public:
  Pimpl();
  ~Pimpl();

  void initialize(const Core::PanoDefinition* pano, const FrameRate& frameRate);
  void computeOverlay(std::shared_ptr<Core::PanoOpenGLSurface> surf, std::shared_ptr<Core::PanoOpenGLSurface> oglSurf,
                      mtime_t date);
  void renderOverlay(int winWidth, int winHeight);

 private:
  friend class Overlayer;

  bool createCubemapSurfaceFBAndTexture();
  bool createEquirectSurfaceFBAndTexture();
  bool createPanoSrcTexture();
  bool createInputTexture(const Core::OverlayInputDefinition* input);
  bool createSphereAoAndBo();

  bool createShader(OGLShader**, const std::string& vShaderCode, const std::string& fShaderCode,
                    const std::string& gShaderCode);

  OverlayTex olTex;
  GLuint sphereVbo, sphereVao;
  OGLShader *sphereShader, *equirectShader, *texturedQuadShader;
  PanoParams pParams;
  Core::QuaternionCurve* stitcherOrientationCurve;
  Core::QuaternionCurve* stitcherStabilizationCurve;

  Pimpl(const Pimpl&);
  const Pimpl& operator=(const Pimpl&);
};

}  // namespace GPU
}  // namespace VideoStitch
