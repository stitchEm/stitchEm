// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "overlay/overlayer.hpp"

#include "libvideostitch/input.hpp"
#include "libvideostitch/inputDef.hpp"
#include "libvideostitch/inputFactory.hpp"

#include "core/readerController.hpp"
#include "core/bufferedReader.hpp"
#include "libvideostitch/gpu_device.hpp"

#include "libvideostitch/overlayInputDef.hpp"

static const int numParallels(80);
static const int numMeridians(60);
static const int cubemapSize(1024);

namespace VideoStitch {
namespace GPU {
static std::string sphereVertexShader = R"(
#version 330 core
layout(location = 0) in vec3 position;
layout(location = 1) in vec2 texCoord;
out vec2 v_texcoord;

void main() {
  gl_Position = vec4(position, 1);
  v_texcoord = texCoord;
}
)";

static std::string sphereFragmentShader = R"(
#version 330 core
uniform sampler2D sampler;
in vec2 g_texcoord;
out vec4 color;
void main() {
  color = texture(sampler, g_texcoord);
}
)";

static std::string sphereGeometryShader = R"(
#version 330 core
uniform mat4 mvp_matrices[6];
layout(triangles) in;
layout(triangle_strip, max_vertices = 18) out;
in vec2 v_texcoord[];
out vec2 g_texcoord;

void main() {
  for (int layer = 0; layer < 6; ++layer) {
    gl_Layer = layer;
    for (int i = 0; i < 3; ++i) {
      gl_Position = mvp_matrices[layer] * gl_in[i].gl_Position;
      g_texcoord = v_texcoord[i];
      EmitVertex();
    }
    EndPrimitive();
  }
}
)";

static std::string texturedQuadVertexShader = R"(
#version 330 core
void main() { }
)";

static std::string texturedQuadFragmentShader = R"(
#version 330 core
uniform sampler2D sampler;
uniform float alphaValue;
in vec2 texcoord;
out vec4 color;

void main() {
  color = texture(sampler, texcoord);
  color.a *= alphaValue;
}
)";

static std::string texturedQuadGeometryShader = R"(
#version 330 core
uniform mat4 mvp_matrices[6];
layout(points) in;
layout(triangle_strip, max_vertices = 24) out;
out vec2 texcoord;

void main() {
  for (int layer = 0; layer < 6; ++layer) {
    gl_Layer = layer;

    gl_Position = mvp_matrices[layer] * vec4(1.0, 1.0, 0.0, 1.0);
    texcoord = vec2(1.0, 1.0);
    EmitVertex();

    gl_Position = mvp_matrices[layer] * vec4(1.0, -1.0, 0.0, 1.0);
    texcoord = vec2(1.0, 0.0);
    EmitVertex();

    gl_Position = mvp_matrices[layer] * vec4(-1.0, 1.0, 0.0, 1.0);
    texcoord = vec2(0.0, 1.0);
    EmitVertex();

    gl_Position = mvp_matrices[layer] * vec4(-1.0, -1.0, 0.0, 1.0);
    texcoord = vec2(0.0, 0.0);
    EmitVertex();

    EndPrimitive();
  }
}
)";

static std::string equirectVertexShader = R"(
#version 330 core
void main() { }
)";

static std::string equirectFragmentShader = R"(
#version 330 core
uniform samplerCube envMap;
in vec2 texcoord;
out vec4 color;

void main() {
  vec2 a = texcoord * vec2(3.14159265, 1.57079633);
  vec2 c = cos(a), s = sin(a);
  color = texture(envMap, vec3(vec2(c.x, s.x) * (-c.y), s.y));
}
)";

static std::string equirectGeometryShader = R"(
#version 330 core
layout(points) in;
layout(triangle_strip, max_vertices = 4) out;
out vec2 texcoord;

void main() {
  gl_Position = vec4(1.0, 1.0, 0.5, 1.0);
  texcoord = vec2(1.0, 1.0);
  EmitVertex();

  gl_Position = vec4(-1.0, 1.0, 0.5, 1.0);
  texcoord = vec2(-1.0, 1.0);
  EmitVertex();

  gl_Position = vec4(1.0, -1.0, 0.5, 1.0);
  texcoord = vec2(1.0, -1.0);
  EmitVertex();

  gl_Position = vec4(-1.0, -1.0, 0.5, 1.0);
  texcoord = vec2(-1.0, -1.0);
  EmitVertex();

  EndPrimitive();
}
)";

void generateModelMatrix(const glm::vec3 trans, const glm::vec3 scale, const float yaw, const float pitch,
                         const float roll, glm::mat4& Model);
void generateModelViewProjections(const glm::mat4 model, const glm::mat4 mgo, glm::mat4* mvps);

OverlayTex::~OverlayTex() {
  glDeleteTextures(1, (GLuint*)&panoSrcTexId);
  glDeleteTextures(1, (GLuint*)&equirectTexId);
  glDeleteTextures(1, (GLuint*)&cubemapTexId);

  for (auto& lt : inputTexIds) {
    delete lt.first;
    glDeleteTextures(1, (GLuint*)&lt.second);
  }

  glDeleteFramebuffers(1, (GLuint*)&cubemapFboId);
  glDeleteFramebuffers(1, (GLuint*)&equirectFboId);
  glDeleteFramebuffers(1, (GLuint*)&textFboId);
}

Overlayer::Pimpl::Pimpl()
    : olTex(),
      sphereShader(nullptr),
      equirectShader(nullptr),
      texturedQuadShader(nullptr),
      stitcherOrientationCurve(nullptr),
      stitcherStabilizationCurve(nullptr) {}

Overlayer::Pimpl::~Pimpl() {
  delete sphereShader;
  delete equirectShader;
  delete texturedQuadShader;
  delete stitcherOrientationCurve;
  delete stitcherStabilizationCurve;
}
Overlayer::Overlayer() : pimpl(new Pimpl) {}

Overlayer::~Overlayer() { delete pimpl; }

void Overlayer::initialize(const Core::PanoDefinition* pano, const FrameRate& frameRate) {
  pimpl->initialize(pano, frameRate);
}

void Overlayer::Pimpl::initialize(const Core::PanoDefinition* pano, const FrameRate& frameRate) {
  glewExperimental = GL_TRUE;
  GLenum glerr = glewInit();
  if (glerr != GL_NO_ERROR) {
    std::ostringstream oss;
    oss << "SetupFailure: Unable to initialize glew";
    Logger::error("overlay") << oss.str() << std::endl;
  }

  pParams.panoTexWidth = pano->getWidth();
  pParams.panoTexHeight = pano->getHeight();
  pParams.panoFrameRate = frameRate;

  // init the OLTex;
  // load inputs
  const overlayreaderid_t numOverlays = pano->numOverlays();
  for (overlayreaderid_t i = 0; i < numOverlays; i++) {
    VideoStitch::Core::OverlayInputDefinition* overlayDef = pano->getOverlay(i).clone();
    createInputTexture(overlayDef);
  }

  if (stitcherOrientationCurve) {
    delete stitcherOrientationCurve;
    stitcherOrientationCurve = nullptr;
  }
  stitcherOrientationCurve = pano->getGlobalOrientation().clone();

  if (stitcherStabilizationCurve) {
    delete stitcherStabilizationCurve;
    stitcherStabilizationCurve = nullptr;
  }
  stitcherStabilizationCurve = pano->getStabilization().clone();

  createPanoSrcTexture();
  createCubemapSurfaceFBAndTexture();
  createEquirectSurfaceFBAndTexture();
  createSphereAoAndBo();

  createShader(&sphereShader, sphereVertexShader, sphereFragmentShader, sphereGeometryShader);
  createShader(&equirectShader, equirectVertexShader, equirectFragmentShader, equirectGeometryShader);
  createShader(&texturedQuadShader, texturedQuadVertexShader, texturedQuadFragmentShader, texturedQuadGeometryShader);
}

void Overlayer::renderOverlay(int winWidth, int winHeight) { pimpl->renderOverlay(winWidth, winHeight); }

void Overlayer::Pimpl::renderOverlay(int winWidth, int winHeight) {
  glBindFramebuffer(GL_READ_FRAMEBUFFER, olTex.equirectFboId);
  glReadBuffer(GL_COLOR_ATTACHMENT0);
  glBlitFramebuffer(0, 0, (GLint)pParams.panoTexWidth, (GLint)pParams.panoTexHeight, 0, 0, (GLint)winWidth,
                    (GLint)winHeight, GL_COLOR_BUFFER_BIT, GL_LINEAR);
}

void Overlayer::computeOverlay(std::shared_ptr<Core::PanoOpenGLSurface> surf,
                               std::shared_ptr<Core::PanoOpenGLSurface> oglSurf, mtime_t date) {
  pimpl->computeOverlay(surf, oglSurf, date);
}

void Overlayer::Pimpl::computeOverlay(std::shared_ptr<Core::PanoOpenGLSurface> surf,
                                      std::shared_ptr<Core::PanoOpenGLSurface> oglSurf, mtime_t date) {
  olTex.panoSrcPboId = surf->pixelbuffer;
  olTex.panoSrcTexWidth = surf->getWidth();
  olTex.panoSrcTexHeight = surf->getHeight();

  pParams.panoFrameId = pParams.panoFrameRate.timestampToFrame(date);

  // Step 0: unpack surf pbo to texture
  glEnable(GL_CULL_FACE);
  {
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, olTex.panoSrcPboId);
    glBindTexture(GL_TEXTURE_2D, olTex.panoSrcTexId);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, (GLsizei)olTex.panoSrcTexWidth, (GLsizei)olTex.panoSrcTexHeight, GL_RGBA,
                    GL_UNSIGNED_BYTE, NULL);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
  }

  glBindFramebuffer(GL_FRAMEBUFFER, olTex.cubemapFboId);
  glViewport(0, 0, cubemapSize, cubemapSize);
  glClearColor(0.6f, 0.6f, 0.0f, 1.0f);
  glClear(GL_COLOR_BUFFER_BIT);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  // Step 1: render a cube map texture of the world
  {
    sphereShader->use();
    glBindVertexArray(sphereVao);
    glBindTexture(GL_TEXTURE_2D, olTex.panoSrcTexId);
    glUniform1i(glGetUniformLocation(sphereShader->Program, "sampler"), 0);

    // model matrix : an identity matrix (sphere model will be at the origin)
    glm::mat4 model = glm::mat4(1.0f);
    glm::mat4 mgo = glm::mat4(1.0f);
    glm::mat4 mvps[6];

    generateModelViewProjections(model, mgo, mvps);
    glUniformMatrix4fv(glGetUniformLocation(sphereShader->Program, "mvp_matrices"), 6, GL_FALSE,
                       glm::value_ptr(mvps[0]));

    glDrawArrays(GL_TRIANGLE_STRIP, 0, 2 * (numParallels + 1) * (numMeridians + 1));
  }

  // Step 2: project the overlay objects
  glDisable(GL_CULL_FACE);
  {
    for (auto& frame : olTex.inputTexIds) {
      double alpha = frame.first->getAlphaCurve().at(pParams.panoFrameId);
      if (alpha < std::numeric_limits<double>::epsilon()) {
        continue;
      } else if (alpha > 1.0) {
        alpha = 1.0;
      }

      texturedQuadShader->use();

      glm::mat4 model;
      glm::mat4 mgo;
      glm::mat4 mvps[6];
      double yaw, pitch, roll;
      float widthScale, heightScale;
      int width = (int)frame.first->getWidth();
      int height = (int)frame.first->getHeight();
      float scale = (float)frame.first->getScaleCurve().at(pParams.panoFrameId);
      frame.first->getRotationCurve().at(pParams.panoFrameId).toEuler(yaw, pitch, roll);

      if (frame.first->getGlobalOrientationApplied()) {
        double yawG, pitchG, rollG;
        auto orientation =
            stitcherStabilizationCurve->at(pParams.panoFrameId) * stitcherOrientationCurve->at(pParams.panoFrameId);
        orientation.toEuler(yawG, pitchG, rollG);
        generateModelMatrix(glm::vec3(0.0), glm::vec3(1.0), (float)-rollG, (float)-pitchG, (float)yawG, mgo);
      } else {
        mgo = glm::mat4(1.0);
      }

      float whRatio = (float)height / (float)width;
      if (whRatio < 1.) {
        widthScale = scale;
        heightScale = scale * whRatio;
      } else {
        widthScale = scale / whRatio;
        heightScale = scale;
      }

      generateModelMatrix(glm::vec3((float)frame.first->getTransXCurve().at(pParams.panoFrameId),
                                    (float)frame.first->getTransYCurve().at(pParams.panoFrameId),
                                    (float)frame.first->getTransZCurve().at(pParams.panoFrameId)),
                          glm::vec3(widthScale, heightScale, 0.0), (float)yaw, (float)pitch, (float)roll, model);

      generateModelViewProjections(model, mgo, mvps);
      glUniformMatrix4fv(glGetUniformLocation(texturedQuadShader->Program, "mvp_matrices"), 6, GL_FALSE,
                         glm::value_ptr(mvps[0]));
      glBindTexture(GL_TEXTURE_2D, frame.second);
      glUniform1i(glGetUniformLocation(texturedQuadShader->Program, "sampler"), 0);
      glUniform1f(glGetUniformLocation(texturedQuadShader->Program, "alphaValue"), (float)alpha);

      glDrawArrays(GL_POINTS, 0, 1);
    }

    glActiveTexture(GL_TEXTURE0);
  }

  // Step 3: reproject the cube map to an equirectangular view
  glEnable(GL_CULL_FACE);
  {
    glBindFramebuffer(GL_FRAMEBUFFER, olTex.equirectFboId);
    glViewport(0, 0, (GLsizei)olTex.panoSrcTexWidth, (GLsizei)olTex.panoSrcTexHeight);
    glBindTexture(GL_TEXTURE_CUBE_MAP, olTex.cubemapTexId);
    glUniform1i(glGetUniformLocation(equirectShader->Program, "envMap"), 0);

    equirectShader->use();
    glDrawArrays(GL_POINTS, 0, 1);
  }

  glBindVertexArray(0);
  glBindFramebuffer(GL_FRAMEBUFFER, 0);

  // transfer to pbo
  glBindTexture(GL_TEXTURE_2D, olTex.equirectTexId);
  glBindBuffer(GL_PIXEL_PACK_BUFFER, oglSurf->pixelbuffer);
  glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
  glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);

  olTex.date = date;
}

bool Overlayer::Pimpl::createCubemapSurfaceFBAndTexture() {
  // clear error flag before mapping to CUDA/OpenCL
  GLenum glerr = glGetError();
  while (glerr != GL_NO_ERROR) {
    glerr = glGetError();
  }

  // Framebuffers
  glGenFramebuffers(1, (GLuint*)&olTex.cubemapFboId);
  glBindFramebuffer(GL_FRAMEBUFFER, olTex.cubemapFboId);

  // Create a cubemap color attachment texture
  glGenTextures(1, (GLuint*)&olTex.cubemapTexId);
  glBindTexture(GL_TEXTURE_CUBE_MAP, olTex.cubemapTexId);
  glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

  glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);

  glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X, 0, GL_RGBA8, (GLsizei)cubemapSize, (GLsizei)cubemapSize, 0, GL_RGBA,
               GL_UNSIGNED_BYTE, NULL);
  glTexImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_X, 0, GL_RGBA8, (GLsizei)cubemapSize, (GLsizei)cubemapSize, 0, GL_RGBA,
               GL_UNSIGNED_BYTE, NULL);
  glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_Y, 0, GL_RGBA8, (GLsizei)cubemapSize, (GLsizei)cubemapSize, 0, GL_RGBA,
               GL_UNSIGNED_BYTE, NULL);
  glTexImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_Y, 0, GL_RGBA8, (GLsizei)cubemapSize, (GLsizei)cubemapSize, 0, GL_RGBA,
               GL_UNSIGNED_BYTE, NULL);
  glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_Z, 0, GL_RGBA8, (GLsizei)cubemapSize, (GLsizei)cubemapSize, 0, GL_RGBA,
               GL_UNSIGNED_BYTE, NULL);
  glTexImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_Z, 0, GL_RGBA8, (GLsizei)cubemapSize, (GLsizei)cubemapSize, 0, GL_RGBA,
               GL_UNSIGNED_BYTE, NULL);

  glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, olTex.cubemapTexId, 0);

  // Now that we actually created the framebuffer and added all attachments we want to check if it is actually complete
  // now
  if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) return false;
  glBindFramebuffer(GL_FRAMEBUFFER, 0);

  glerr = glGetError();
  if (glerr != GL_NO_ERROR) {
    return false;
  }

  return true;
}

bool Overlayer::Pimpl::createEquirectSurfaceFBAndTexture() {
  // clear error flag before mapping to CUDA/OpenCL
  GLenum glerr = glGetError();
  while (glerr != GL_NO_ERROR) {
    glerr = glGetError();
  }

  int64_t bufSize = pParams.panoTexWidth * pParams.panoTexHeight * 4;

  if (bufSize > std::numeric_limits<GLsizei>::max()) {
    std::ostringstream oss;
    oss << "Could not allocate OpenGL Surface of size " << bufSize
        << ". Maximum supported texture size: " << std::numeric_limits<GLsizei>::max();
    Logger::error("overlay") << oss.str() << std::endl;
    return false;
  }

  // Framebuffers
  glGenFramebuffers(1, (GLuint*)&olTex.equirectFboId);
  glBindFramebuffer(GL_FRAMEBUFFER, olTex.equirectFboId);

  // Create a cubemap color attachment texture
  glGenTextures(1, (GLuint*)&olTex.equirectTexId);
  glBindTexture(GL_TEXTURE_2D, olTex.equirectTexId);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, (GLsizei)pParams.panoTexWidth, (GLsizei)pParams.panoTexHeight, 0, GL_RGBA,
               GL_UNSIGNED_BYTE, NULL);

  glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, olTex.equirectTexId, 0);

  glGenBuffers(1, (GLuint*)&olTex.equirectPboId);
  glBindBuffer(GL_ARRAY_BUFFER, olTex.equirectPboId);
  glBufferData(GL_ARRAY_BUFFER, bufSize, NULL, GL_STREAM_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  // Now that we actually created the framebuffer and added all attachments we want to check if it is actually complete
  // now
  if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) return false;
  glBindFramebuffer(GL_FRAMEBUFFER, 0);

  glerr = glGetError();
  if (glerr != GL_NO_ERROR) {
    return false;
  }

  return true;
}

bool Overlayer::Pimpl::createInputTexture(const Core::OverlayInputDefinition* input) {
  GLenum glerr = glGetError();
  while (glerr != GL_NO_ERROR) {
    glerr = glGetError();
  }

  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

  Input::DefaultReaderFactory factory(-1, -1);
  readerid_t readerId = 0;
  Potential<Input::Reader> potReader = factory.create(readerId, *input);
  if (!potReader.ok()) {
    std::ostringstream oss;
    oss << "Could not grab input from " << input->getDisplayName();
    Logger::error("overlay") << oss.str() << std::endl;
    return false;
  }

  Input::VideoReader* reader = potReader.release()->getVideoReader();
  if (reader->getSpec().format != RGBA) {
    Logger::error("overlay") << "Only RGBA pixel format is supported by overlay, please modify input image format."
                             << std::endl;
    return false;
  }

  mtime_t pts;
  unsigned char* frame = new unsigned char[input->getWidth() * input->getHeight() * 4];
  Input::ReadStatus stat = reader->readFrame(pts, frame);
  if (!stat.ok()) {
    Logger::error("overlay") << "ould not read logo input." << std::endl;
    return false;
  }

  GLuint imageTexId;
  glGenTextures(1, (GLuint*)&imageTexId);
  glBindTexture(GL_TEXTURE_2D, imageTexId);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, (GLsizei)input->getWidth(), (GLsizei)input->getHeight(), 0, GL_RGBA,
               GL_UNSIGNED_BYTE, frame);
  glGenerateMipmap(GL_TEXTURE_2D);

  olTex.inputTexIds.push_back(std::make_pair(input, imageTexId));

  delete reader;
  delete[] frame;

  glerr = glGetError();
  if (glerr != GL_NO_ERROR) {
    return false;
  }

  return true;
}

bool Overlayer::Pimpl::createPanoSrcTexture() {
  GLenum glerr = glGetError();
  while (glerr != GL_NO_ERROR) {
    glerr = glGetError();
  }

  glGenTextures(1, (GLuint*)&olTex.panoSrcTexId);
  glBindTexture(GL_TEXTURE_2D, olTex.panoSrcTexId);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, (GLsizei)pParams.panoTexWidth, (GLsizei)pParams.panoTexHeight, 0, GL_RGBA,
               GL_UNSIGNED_BYTE, 0);
  glBindTexture(GL_TEXTURE_2D, 0);

  glBindTexture(GL_TEXTURE_2D, 0);

  glerr = glGetError();
  if (glerr != GL_NO_ERROR) {
    return false;
  }
  return true;
}

bool Overlayer::Pimpl::createSphereAoAndBo() {
  GLenum glerr = glGetError();
  while (glerr != GL_NO_ERROR) {
    glerr = glGetError();
  }

  const float radius = 20.;

  std::vector<GLfloat> vertices((numParallels + 1) * (numMeridians + 1) * 10);

  float hfov = 360.f, vfov = 180.f;
  float rHFov = hfov * (float)M_PI / 180.f;
  float rVFov = vfov * (float)M_PI / 180.f;
  int k = 0;

  for (int i = 0; i <= numParallels; i++) {
    float v0 = (i - 1) / (float)(numParallels);
    float lat0 = rVFov * (-0.5f + v0);
    float z0 = sinf(lat0);
    float zr0 = cosf(lat0);

    float v1 = i / (float)(numParallels);
    float lat1 = rVFov * (-0.5f + v1);
    float z1 = sinf(lat1);
    float zr1 = cosf(lat1);

    for (int j = 0; j <= numMeridians; j++) {
      float u = j / (float)numMeridians;
      float lng = rHFov * u;
      float x = cosf(lng);
      float y = sinf(lng);

      vertices[k++] = x * zr0 * radius;  // X
      vertices[k++] = y * zr0 * radius;  // Y
      vertices[k++] = z0 * radius;       // Z

      vertices[k++] = u;   // U
      vertices[k++] = v0;  // V

      vertices[k++] = x * zr1 * radius;  // X
      vertices[k++] = y * zr1 * radius;  // Y
      vertices[k++] = z1 * radius;       // Z

      vertices[k++] = u;   // U
      vertices[k++] = v1;  // V
    }
  }

  glGenVertexArrays(1, &sphereVao);
  glGenBuffers(1, &sphereVbo);
  glBindVertexArray(sphereVao);
  glBindBuffer(GL_ARRAY_BUFFER, sphereVbo);
  glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_STATIC_DRAW);
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(GLfloat), (GLvoid*)0);
  glEnableVertexAttribArray(1);
  glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(GLfloat), (GLvoid*)(3 * sizeof(GLfloat)));
  glBindVertexArray(0);

  glerr = glGetError();
  if (glerr != GL_NO_ERROR) {
    return false;
  }
  return true;
}

bool Overlayer::Pimpl::createShader(OGLShader** shader, const std::string& vShader, const std::string& fShader,
                                    const std::string& gShader) {
  GLenum glerr = glGetError();
  while (glerr != GL_NO_ERROR) {
    glerr = glGetError();
  }

  setlocale(LC_NUMERIC, "C");
  if (*shader) {
    delete *shader;
  }
  *shader = new OGLShader(vShader.c_str(), fShader.c_str(), gShader.c_str());
  setlocale(LC_ALL, "");

  glerr = glGetError();
  if (glerr != GL_NO_ERROR) {
    return false;
  }

  return true;
}

void generateModelMatrix(const glm::vec3 trans, const glm::vec3 scale, const float yaw, const float pitch,
                         const float roll, glm::mat4& model) {
  glm::mat4 Model_translated = glm::translate(glm::mat4(), trans);
  glm::mat4 Model_translated_roll = glm::rotate(Model_translated, roll, glm::vec3(0.0, 0.0, 1.0));
  glm::mat4 Model_translated_roll_pitch = glm::rotate(Model_translated_roll, pitch, glm::vec3(0.0, 1.0, 0.0));
  glm::mat4 Model_translated_roll_pitch_yaw = glm::rotate(Model_translated_roll_pitch, yaw, glm::vec3(1.0, 0.0, 0.0));

  model = glm::scale(Model_translated_roll_pitch_yaw, scale);
}

void generateModelViewProjections(const glm::mat4 model, const glm::mat4 mgo, glm::mat4* mvps) {
  // Projection matrix : 90 Field of View, 1:1 ratio, display range : 0.1 unit <-> 100 units
  glm::mat4 Projection = glm::perspective(glm::radians(90.0f), 1.f, 0.1f, 100.0f);

  // Face +X
  {
    // Camera matrix
    glm::mat4 View = glm::lookAt(glm::vec3(0, 0, 0),  // Camera is at (0,0,0), in World Space
                                 glm::vec3(1, 0, 0),  // and looks at the axis +x
                                 glm::vec3(0, -1, 0)  // Head is up
    );
    // Our ModelViewProjection : multiplication of our 3 matrices
    mvps[0] = Projection * View * mgo * model;  // Remember, matrix multiplication is the other way around
  }

  // Face -X
  {
    // Camera matrix
    glm::mat4 View = glm::lookAt(glm::vec3(0, 0, 0),   // Camera is at (0,0,0), in World Space
                                 glm::vec3(-1, 0, 0),  // and looks at the axis -x
                                 glm::vec3(0, -1, 0)   // Head is up
    );
    // Our ModelViewProjection : multiplication of our 3 matrices
    mvps[1] = Projection * View * mgo * model;  // Remember, matrix multiplication is the other way around
  }

  // Face +Y
  {
    // Camera matrix
    glm::mat4 View = glm::lookAt(glm::vec3(0, 0, 0),  // Camera is at (0,0,0), in World Space
                                 glm::vec3(0, 1, 0),  // and looks at the axis +y
                                 glm::vec3(0, 0, 1)   // Head is up
    );
    // Our ModelViewProjection : multiplication of our 3 matrices
    mvps[2] = Projection * View * mgo * model;  // Remember, matrix multiplication is the other way around
  }

  // Face -Y
  {
    // Camera matrix
    glm::mat4 View = glm::lookAt(glm::vec3(0, 0, 0),   // Camera is at (0,0,0), in World Space
                                 glm::vec3(0, -1, 0),  // and looks at the axis -y
                                 glm::vec3(0, 0, -1)   // Head is up
    );
    // Our ModelViewProjection : multiplication of our 3 matrices
    mvps[3] = Projection * View * mgo * model;  // Remember, matrix multiplication is the other way around
  }

  // Face +Z
  {
    // Camera matrix
    glm::mat4 View = glm::lookAt(glm::vec3(0, 0, 0),  // Camera is at (0,0,0), in World Space
                                 glm::vec3(0, 0, 1),  // and looks at the axis +z
                                 glm::vec3(0, -1, 0)  // Head is up
    );
    // Our ModelViewProjection : multiplication of our 3 matrices
    mvps[4] = Projection * View * mgo * model;  // Remember, matrix multiplication is the other way around
  }

  // Face -Z
  {
    // Camera matrix
    glm::mat4 View = glm::lookAt(glm::vec3(0, 0, 0),   // Camera is at (0,0,0), in World Space
                                 glm::vec3(0, 0, -1),  // and looks at the axis -z
                                 glm::vec3(0, -1, 0)   // Head is up
    );
    // Our ModelViewProjection : multiplication of our 3 matrices
    mvps[5] = Projection * View * mgo * model;  // Remember, matrix multiplication is the other way around
  }
}
}  // namespace GPU
}  // namespace VideoStitch
