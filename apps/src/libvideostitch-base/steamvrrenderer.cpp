// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "steamvrrenderer.hpp"
#include "texture.hpp"

using namespace lineag;
static const unsigned int DEFAULT_WIDTH(1280);
static const unsigned int DEFAULT_HEIGHT(720);

void ThreadSleep(unsigned long nMilliseconds) {
#if defined(_WIN32)
  ::Sleep(nMilliseconds);
#elif defined(POSIX)
  usleep(nMilliseconds * 1000);
#endif
}

SteamVRRenderer::SteamVRRenderer()
    : Renderer(),
      hmd(nullptr),
      renderModels(nullptr),
      windowWidth(DEFAULT_WIDTH),
      windowHeight(DEFAULT_HEIGHT),
      openglInitialized(false),
      lensProgramID(0),
      controllerTransformProgramID(0),
      renderModelProgramID(0),
      controllerMatrixLocation(0),
      renderModelMatrixLocation(0),
      renderWidth(0),
      renderHeight(0),
      nearClip(0.f),
      farClip(0.f),
      lensVAO(0),
      glIDVertBuffer(0),
      glIDIndexBuffer(0),
      indexSize(0),
      glControllerVertBuffer(0),
      controllerVAO(0),
      controllerVertcount(0) {}

SteamVRRenderer::~SteamVRRenderer() {
  if (!openglInitialized) {
    return;
  }
}

// -------------- Render Models -------------------------------

CGLRenderModel::CGLRenderModel(const std::string &renderModelName)
    : glIndexBuffer(0), glVertArray(0), glVertBuffer(0), glTexture(0), modelName(renderModelName) {}

CGLRenderModel::~CGLRenderModel() { cleanup(); }

bool CGLRenderModel::init(const vr::RenderModel_t &vrModel, const vr::RenderModel_TextureMap_t &vrDiffuseTexture) {
  initializeOpenGLFunctions();

  // create and bind a VAO to hold state for this model
  glGenVertexArrays(1, &glVertArray);
  glBindVertexArray(glVertArray);

  // Populate a vertex buffer
  glGenBuffers(1, &glVertBuffer);
  glBindBuffer(GL_ARRAY_BUFFER, glVertBuffer);
  glBufferData(GL_ARRAY_BUFFER, sizeof(vr::RenderModel_Vertex_t) * vrModel.unVertexCount, vrModel.rVertexData,
               GL_STATIC_DRAW);

  // Identify the components in the vertex buffer
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(vr::RenderModel_Vertex_t),
                        (void *)offsetof(vr::RenderModel_Vertex_t, vPosition));
  glEnableVertexAttribArray(1);
  glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(vr::RenderModel_Vertex_t),
                        (void *)offsetof(vr::RenderModel_Vertex_t, vNormal));
  glEnableVertexAttribArray(2);
  glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(vr::RenderModel_Vertex_t),
                        (void *)offsetof(vr::RenderModel_Vertex_t, rfTextureCoord));

  // Create and populate the index buffer
  glGenBuffers(1, &glIndexBuffer);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, glIndexBuffer);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(uint16_t) * vrModel.unTriangleCount * 3, vrModel.rIndexData,
               GL_STATIC_DRAW);

  glBindVertexArray(0);

  // create and populate the texture
  glGenTextures(1, &glTexture);
  glBindTexture(GL_TEXTURE_2D, glTexture);

  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, vrDiffuseTexture.unWidth, vrDiffuseTexture.unHeight, 0, GL_RGBA,
               GL_UNSIGNED_BYTE, vrDiffuseTexture.rubTextureMapData);

  // If this renders black ask McJohn what's wrong.
  glGenerateMipmap(GL_TEXTURE_2D);

  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);

  GLfloat largest;
  glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, &largest);
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, largest);

  glBindTexture(GL_TEXTURE_2D, 0);

  vertexCount = vrModel.unTriangleCount * 3;

  return true;
}

void CGLRenderModel::cleanup() {
  if (glVertBuffer) {
    glDeleteBuffers(1, &glIndexBuffer);
    glDeleteBuffers(1, &glVertArray);
    glDeleteBuffers(1, &glVertBuffer);
    glIndexBuffer = 0;
    glVertArray = 0;
    glVertBuffer = 0;
  }
}

void CGLRenderModel::draw() {
  glBindVertexArray(glVertArray);
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, glTexture);
  glDrawElements(GL_TRIANGLES, vertexCount, GL_UNSIGNED_SHORT, 0);
  glBindVertexArray(0);
}

// -------------- Service initialization ----------------------

bool SteamVRRenderer::initializeSteamVR() {
  // Loading the SteamVR Runtime
  vr::EVRInitError error = vr::VRInitError_None;
  hmd = vr::VR_Init(&error, vr::VRApplication_Scene);

  if (error != vr::VRInitError_None) {
    hmd = nullptr;
    const QString mess = "Unable to init VR runtime: %0";
    emit logMessage(mess.arg(vr::VR_GetVRInitErrorAsEnglishDescription(error)), VideoStitch::LOG_ERROR);
    return false;
  }

  renderModels = (vr::IVRRenderModels *)vr::VR_GetGenericInterface(vr::IVRRenderModels_Version, &error);
  if (!renderModels) {
    hmd = nullptr;
    vr::VR_Shutdown();
    const QString mess = "Unable to get render model interface: %0";
    emit logMessage(mess.arg(vr::VR_GetVRInitErrorAsEnglishDescription(error)), VideoStitch::LOG_ERROR);
    return false;
  }

  if (!vr::VRCompositor()) {
    const QString mess = "Compositor initialization failed. See log file for details";
    emit logMessage(mess, VideoStitch::LOG_ERROR);
    return false;
  }
  return true;
}

void SteamVRRenderer::uninitializeSteamVR() { vr::VR_Shutdown(); }

// ----------------- OpenGL rendering configuration ------------

void SteamVRRenderer::configureRendering(int width, int height) {
  Q_UNUSED(width);
  Q_UNUSED(height);

  Renderer::initialize();

  nearClip = 0.1f;
  farClip = 30.0f;

  createAllShaders();
  setupCameras();
  setupStereoRenderTargets();
  setupDistortion();
  setupRenderModels();

  Q_ASSERT(!glGetError());
  openglInitialized = true;
}

void SteamVRRenderer::setupCameras() {
  mat4ProjectionLeft = getHMDMatrixProjectionEye(vr::Eye_Left);
  mat4ProjectionRight = getHMDMatrixProjectionEye(vr::Eye_Right);
  mat4eyePosLeft = getHMDMatrixPoseEye(vr::Eye_Left);
  mat4eyePosRight = getHMDMatrixPoseEye(vr::Eye_Right);
}

Matrix4 SteamVRRenderer::getHMDMatrixProjectionEye(vr::Hmd_Eye eye) {
  if (!hmd) {
    return Matrix4();
  }

  vr::HmdMatrix44_t mat = hmd->GetProjectionMatrix(eye, nearClip, farClip);
  return Matrix4(mat.m[0][0], mat.m[1][0], mat.m[2][0], mat.m[3][0], mat.m[0][1], mat.m[1][1], mat.m[2][1], mat.m[3][1],
                 mat.m[0][2], mat.m[1][2], mat.m[2][2], mat.m[3][2], mat.m[0][3], mat.m[1][3], mat.m[2][3],
                 mat.m[3][3]);
}

Matrix4 SteamVRRenderer::getHMDMatrixPoseEye(vr::Hmd_Eye eye) {
  if (!hmd) {
    return Matrix4();
  }

  vr::HmdMatrix34_t matEyeRight = hmd->GetEyeToHeadTransform(eye);
  Matrix4 matrixObj(matEyeRight.m[0][0], matEyeRight.m[1][0], matEyeRight.m[2][0], 0.0, matEyeRight.m[0][1],
                    matEyeRight.m[1][1], matEyeRight.m[2][1], 0.0, matEyeRight.m[0][2], matEyeRight.m[1][2],
                    matEyeRight.m[2][2], 0.0, matEyeRight.m[0][3], matEyeRight.m[1][3], matEyeRight.m[2][3], 1.0f);

  return matrixObj.invert();
}

bool SteamVRRenderer::setupStereoRenderTargets() {
  if (!hmd) {
    return false;
  }

  hmd->GetRecommendedRenderTargetSize(&renderWidth, &renderHeight);
  createFrameBuffer(renderWidth, renderHeight, leftEyeDesc);
  createFrameBuffer(renderWidth, renderHeight, rightEyeDesc);
  return true;
}

bool SteamVRRenderer::createFrameBuffer(int width, int height, FramebufferDesc &framebufferDesc) {
  glGenFramebuffers(1, &framebufferDesc.renderFramebufferId);
  glBindFramebuffer(GL_FRAMEBUFFER, framebufferDesc.renderFramebufferId);

  glGenRenderbuffers(1, &framebufferDesc.depthBufferId);
  glBindRenderbuffer(GL_RENDERBUFFER, framebufferDesc.depthBufferId);
  glRenderbufferStorageMultisample(GL_RENDERBUFFER, 4, GL_DEPTH_COMPONENT, width, height);
  glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, framebufferDesc.depthBufferId);

  glGenTextures(1, &framebufferDesc.renderTextureId);
  glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, framebufferDesc.renderTextureId);
  glTexImage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, 4, GL_RGBA8, width, height, true);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D_MULTISAMPLE,
                         framebufferDesc.renderTextureId, 0);

  glGenFramebuffers(1, &framebufferDesc.resolveFramebufferId);
  glBindFramebuffer(GL_FRAMEBUFFER, framebufferDesc.resolveFramebufferId);

  glGenTextures(1, &framebufferDesc.resolveTextureId);
  glBindTexture(GL_TEXTURE_2D, framebufferDesc.resolveTextureId);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, framebufferDesc.resolveTextureId, 0);

  // check FBO status
  GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
  if (status != GL_FRAMEBUFFER_COMPLETE) {
    return false;
  }

  glBindFramebuffer(GL_FRAMEBUFFER, 0);

  return true;
}

void SteamVRRenderer::setupDistortion() {
  if (!hmd) {
    return;
  }

  GLushort lensGridSegmentCountH = 43;
  GLushort lensGridSegmentCountV = 43;

  float w = (float)(1.0 / float(lensGridSegmentCountH - 1));
  float h = (float)(1.0 / float(lensGridSegmentCountV - 1));
  float u, v = 0;

  std::vector<VertexDataLens> vertices;
  VertexDataLens vert;

  // left eye distortion verts
  float Xoffset = -1;
  for (int y = 0; y < lensGridSegmentCountV; ++y) {
    for (int x = 0; x < lensGridSegmentCountH; ++x) {
      u = x * w;
      v = 1 - y * h;
      vert.position = Vector2(Xoffset + u, -1 + 2 * y * h);
      vr::DistortionCoordinates_t *dc0;
      hmd->ComputeDistortion(vr::Eye_Left, u, v, dc0);
      vert.texCoordRed = Vector2(dc0->rfRed[0], 1 - dc0->rfRed[1]);
      vert.texCoordGreen = Vector2(dc0->rfGreen[0], 1 - dc0->rfGreen[1]);
      vert.texCoordBlue = Vector2(dc0->rfBlue[0], 1 - dc0->rfBlue[1]);
      vertices.push_back(vert);
    }
  }

  // right eye distortion verts
  Xoffset = 0;
  for (int y = 0; y < lensGridSegmentCountV; ++y) {
    for (int x = 0; x < lensGridSegmentCountH; ++x) {
      u = x * w;
      v = 1 - y * h;
      vert.position = Vector2(Xoffset + u, -1 + 2 * y * h);
      vr::DistortionCoordinates_t *dc0;
      hmd->ComputeDistortion(vr::Eye_Right, u, v, dc0);
      vert.texCoordRed = Vector2(dc0->rfRed[0], 1 - dc0->rfRed[1]);
      vert.texCoordGreen = Vector2(dc0->rfGreen[0], 1 - dc0->rfGreen[1]);
      vert.texCoordBlue = Vector2(dc0->rfBlue[0], 1 - dc0->rfBlue[1]);
      vertices.push_back(vert);
    }
  }

  std::vector<GLushort> indices;
  GLushort a, b, c, d;

  GLushort offset = 0;
  for (GLushort y = 0; y < lensGridSegmentCountV - 1; ++y) {
    for (GLushort x = 0; x < lensGridSegmentCountH - 1; ++x) {
      a = lensGridSegmentCountH * y + x + offset;
      b = lensGridSegmentCountH * y + x + 1 + offset;
      c = (y + 1) * lensGridSegmentCountH + x + 1 + offset;
      d = (y + 1) * lensGridSegmentCountH + x + offset;
      indices.push_back(a);
      indices.push_back(b);
      indices.push_back(c);
      indices.push_back(a);
      indices.push_back(c);
      indices.push_back(d);
    }
  }

  offset = lensGridSegmentCountH * lensGridSegmentCountV;
  for (GLushort y = 0; y < lensGridSegmentCountV - 1; ++y) {
    for (GLushort x = 0; x < lensGridSegmentCountH - 1; ++x) {
      a = lensGridSegmentCountH * y + x + offset;
      b = lensGridSegmentCountH * y + x + 1 + offset;
      c = (y + 1) * lensGridSegmentCountH + x + 1 + offset;
      d = (y + 1) * lensGridSegmentCountH + x + offset;
      indices.push_back(a);
      indices.push_back(b);
      indices.push_back(c);
      indices.push_back(a);
      indices.push_back(c);
      indices.push_back(d);
    }
  }
  indexSize = (unsigned int)indices.size();

  glGenVertexArrays(1, &lensVAO);
  glBindVertexArray(lensVAO);

  glGenBuffers(1, &glIDVertBuffer);
  glBindBuffer(GL_ARRAY_BUFFER, glIDVertBuffer);
  glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(VertexDataLens), &vertices[0], GL_STATIC_DRAW);

  glGenBuffers(1, &glIDIndexBuffer);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, glIDIndexBuffer);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(GLushort), &indices[0], GL_STATIC_DRAW);

  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(VertexDataLens), (void *)offsetof(VertexDataLens, position));

  glEnableVertexAttribArray(1);
  glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(VertexDataLens),
                        (void *)offsetof(VertexDataLens, texCoordRed));

  glEnableVertexAttribArray(2);
  glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(VertexDataLens),
                        (void *)offsetof(VertexDataLens, texCoordGreen));

  glEnableVertexAttribArray(3);
  glVertexAttribPointer(3, 2, GL_FLOAT, GL_FALSE, sizeof(VertexDataLens),
                        (void *)offsetof(VertexDataLens, texCoordBlue));

  glBindVertexArray(0);

  glDisableVertexAttribArray(0);
  glDisableVertexAttribArray(1);
  glDisableVertexAttribArray(2);
  glDisableVertexAttribArray(3);

  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

void SteamVRRenderer::setupRenderModels() {
  memset(trackedDeviceToRenderModel, 0, sizeof(trackedDeviceToRenderModel));
  if (!hmd) {
    return;
  }

  for (uint32_t trackedDevice = vr::k_unTrackedDeviceIndex_Hmd + 1; trackedDevice < vr::k_unMaxTrackedDeviceCount;
       ++trackedDevice) {
    if (!hmd->IsTrackedDeviceConnected(trackedDevice)) {
      continue;
    }
    setupRenderModelForTrackedDevice(trackedDevice);
  }
}

std::string getTrackedDeviceString(vr::IVRSystem *hmd, vr::TrackedDeviceIndex_t device, vr::TrackedDeviceProperty prop,
                                   vr::TrackedPropertyError *error = nullptr) {
  uint32_t requiredBufferLen = hmd->GetStringTrackedDeviceProperty(device, prop, nullptr, 0, error);
  if (requiredBufferLen == 0) {
    return std::string();
  }

  char *buffer = new char[requiredBufferLen];
  requiredBufferLen = hmd->GetStringTrackedDeviceProperty(device, prop, buffer, requiredBufferLen, error);
  std::string result = buffer;
  delete[] buffer;
  return result;
}

void SteamVRRenderer::setupRenderModelForTrackedDevice(vr::TrackedDeviceIndex_t trackedDeviceIndex) {
  if (trackedDeviceIndex >= vr::k_unMaxTrackedDeviceCount) {
    return;
  }

  // try to find a model we've already set up
  std::string renderModelName = getTrackedDeviceString(hmd, trackedDeviceIndex, vr::Prop_RenderModelName_String);
  CGLRenderModel *renderModel = findOrLoadRenderModel(renderModelName.c_str());
  if (!renderModel) {
    std::string trackingSystemName =
        getTrackedDeviceString(hmd, trackedDeviceIndex, vr::Prop_TrackingSystemName_String);
    const QString mess = "Unable to load render model for tracked device %0 %1 %2";
    emit logMessage(mess.arg(trackedDeviceIndex).arg(trackingSystemName.c_str()).arg(renderModelName.c_str()),
                    VideoStitch::LOG_ERROR);
  } else {
    trackedDeviceToRenderModel[trackedDeviceIndex] = renderModel;
    showTrackedDevice[trackedDeviceIndex] = true;
  }
}

CGLRenderModel *SteamVRRenderer::findOrLoadRenderModel(const char *renderModelName) {
  CGLRenderModel *renderModel = nullptr;
  for (std::vector<CGLRenderModel *>::iterator i = vecRenderModels.begin(); i != vecRenderModels.end(); ++i) {
    if (!stricmp((*i)->getName().c_str(), renderModelName)) {
      renderModel = *i;
      break;
    }
  }

  // load the model if we didn't find one
  if (!renderModel) {
    vr::RenderModel_t *model;
    vr::EVRRenderModelError error;
    while (1) {
      error = vr::VRRenderModels()->LoadRenderModel_Async(renderModelName, &model);
      if (error != vr::VRRenderModelError_Loading) break;
      ThreadSleep(1);
    }

    if (error != vr::VRRenderModelError_None) {
      QString mess;
      mess.append("Unable to load render model ")
          .append(renderModelName)
          .append(" - ")
          .append(vr::VRRenderModels()->GetRenderModelErrorNameFromEnum(error))
          .append("\n");
      emit logMessage(mess, VideoStitch::LOG_ERROR);
      return nullptr;  // move on to the next tracked device
    }

    vr::RenderModel_TextureMap_t *texture;
    while (1) {
      error = vr::VRRenderModels()->LoadTexture_Async(model->diffuseTextureId, &texture);
      if (error != vr::VRRenderModelError_Loading) break;
      ThreadSleep(1);
    }

    if (error != vr::VRRenderModelError_None) {
      QString mess;
      mess.append("Unable to load render texture id:")
          .append(model->diffuseTextureId)
          .append(" for render model ")
          .append(renderModelName)
          .append("\n");
      emit logMessage(mess, VideoStitch::LOG_ERROR);
      vr::VRRenderModels()->FreeRenderModel(model);
      return nullptr;  // move on to the next tracked device
    }

    renderModel = new CGLRenderModel(renderModelName);
    if (!renderModel->init(*model, *texture)) {
      QString mess;
      mess.append("Unable to create GL model from render model ").append(renderModelName).append("\n");
      emit logMessage(mess, VideoStitch::LOG_ERROR);
      delete renderModel;
      renderModel = nullptr;
    } else {
      vecRenderModels.push_back(renderModel);
    }
    vr::VRRenderModels()->FreeRenderModel(model);
    vr::VRRenderModels()->FreeTexture(texture);
  }
  return renderModel;
}

GLuint SteamVRRenderer::compileGLShader(const char *shaderName, const char *vertexShaderSource,
                                        const char *fragmentShaderSource) {
  const GLuint programID = glCreateProgram();
  const GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
  glShaderSource(vertexShader, 1, &vertexShaderSource, nullptr);
  glCompileShader(vertexShader);

  GLint vertexShaderCompiled = GL_FALSE;
  glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &vertexShaderCompiled);
  if (vertexShaderCompiled != GL_TRUE) {
    QString mess;
    mess.append(shaderName).append(" - Unable to compile vertex shader ").append(vertexShader).append("!\n");
    emit logMessage(mess, VideoStitch::LOG_ERROR);
    glDeleteProgram(programID);
    glDeleteShader(vertexShader);
    return 0;
  }
  glAttachShader(programID, vertexShader);
  glDeleteShader(vertexShader);  // the program hangs onto this once it's attached

  GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
  glShaderSource(fragmentShader, 1, &fragmentShaderSource, nullptr);
  glCompileShader(fragmentShader);

  GLint fragmentShaderCompiled = GL_FALSE;
  glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &fragmentShaderCompiled);
  if (fragmentShaderCompiled != GL_TRUE) {
    QString mess;
    mess.append(shaderName).append(" - Unable to compile fragment shader ").append(fragmentShader).append("!\n");
    emit logMessage(mess, VideoStitch::LOG_ERROR);
    glDeleteProgram(programID);
    glDeleteShader(fragmentShader);
    return 0;
  }

  glAttachShader(programID, fragmentShader);
  glDeleteShader(fragmentShader);  // the program hangs onto this once it's attached
  glLinkProgram(programID);

  GLint programSuccess = GL_TRUE;
  glGetProgramiv(programID, GL_LINK_STATUS, &programSuccess);
  if (programSuccess != GL_TRUE) {
    const QString mess = "%0 - Error linking program %1";
    emit logMessage(mess.arg(shaderName).arg(programID), VideoStitch::LOG_ERROR);
    glDeleteProgram(programID);
    return 0;
  }

  glUseProgram(programID);
  glUseProgram(0);

  return programID;
}

bool SteamVRRenderer::createAllShaders() {
  controllerTransformProgramID = compileGLShader("Controller",

                                                 // vertex shader
                                                 "#version 410\n"
                                                 "uniform mat4 matrix;\n"
                                                 "layout(location = 0) in vec4 position;\n"
                                                 "layout(location = 1) in vec3 v3ColorIn;\n"
                                                 "out vec4 v4Color;\n"
                                                 "void main()\n"
                                                 "{\n"
                                                 "	v4Color.xyz = v3ColorIn; v4Color.a = 1.0;\n"
                                                 "	gl_Position = matrix * position;\n"
                                                 "}\n",

                                                 // fragment shader
                                                 "#version 410\n"
                                                 "in vec4 v4Color;\n"
                                                 "out vec4 outputColor;\n"
                                                 "void main()\n"
                                                 "{\n"
                                                 "   outputColor = v4Color;\n"
                                                 "}\n");
  controllerMatrixLocation = glGetUniformLocation(controllerTransformProgramID, "matrix");
  if (controllerMatrixLocation == -1) {
    const QString mess = "Unable to find matrix uniform in controller shader";
    emit logMessage(mess, VideoStitch::LOG_ERROR);
    return false;
  }

  renderModelProgramID = compileGLShader("render model",

                                         // vertex shader
                                         "#version 410\n"
                                         "uniform mat4 matrix;\n"
                                         "layout(location = 0) in vec4 position;\n"
                                         "layout(location = 1) in vec3 v3NormalIn;\n"
                                         "layout(location = 2) in vec2 v2TexCoordsIn;\n"
                                         "out vec2 v2TexCoord;\n"
                                         "void main()\n"
                                         "{\n"
                                         "	v2TexCoord = v2TexCoordsIn;\n"
                                         "	gl_Position = matrix * vec4(position.xyz, 1);\n"
                                         "}\n",

                                         // fragment shader
                                         "#version 410 core\n"
                                         "uniform sampler2D diffuse;\n"
                                         "in vec2 v2TexCoord;\n"
                                         "out vec4 outputColor;\n"
                                         "void main()\n"
                                         "{\n"
                                         "   outputColor = texture( diffuse, v2TexCoord);\n"
                                         "}\n"

  );
  renderModelMatrixLocation = glGetUniformLocation(renderModelProgramID, "matrix");
  if (renderModelMatrixLocation == -1) {
    QString mess;
    mess.append("Unable to find matrix uniform in render model shader\n");
    emit logMessage(mess, VideoStitch::LOG_ERROR);
    return false;
  }

  lensProgramID =
      compileGLShader("Distortion",

                      // vertex shader
                      "#version 410 core\n"
                      "layout(location = 0) in vec4 position;\n"
                      "layout(location = 1) in vec2 v2UVredIn;\n"
                      "layout(location = 2) in vec2 v2UVGreenIn;\n"
                      "layout(location = 3) in vec2 v2UVblueIn;\n"
                      "noperspective  out vec2 v2UVred;\n"
                      "noperspective  out vec2 v2UVgreen;\n"
                      "noperspective  out vec2 v2UVblue;\n"
                      "void main()\n"
                      "{\n"
                      "	v2UVred = v2UVredIn;\n"
                      "	v2UVgreen = v2UVGreenIn;\n"
                      "	v2UVblue = v2UVblueIn;\n"
                      "	gl_Position = position;\n"
                      "}\n",

                      // fragment shader
                      "#version 410 core\n"
                      "uniform sampler2D mytexture;\n"

                      "noperspective  in vec2 v2UVred;\n"
                      "noperspective  in vec2 v2UVgreen;\n"
                      "noperspective  in vec2 v2UVblue;\n"

                      "out vec4 outputColor;\n"

                      "void main()\n"
                      "{\n"
                      "	float fBoundsCheck = ( (dot( vec2( lessThan( v2UVgreen.xy, vec2(0.05, 0.05)) ), vec2(1.0, "
                      "1.0))+dot( vec2( greaterThan( v2UVgreen.xy, vec2( 0.95, 0.95)) ), vec2(1.0, 1.0))) );\n"
                      "	if( fBoundsCheck > 1.0 )\n"
                      "	{ outputColor = vec4( 0, 0, 0, 1.0 ); }\n"
                      "	else\n"
                      "	{\n"
                      "		float red = texture(mytexture, v2UVred).x;\n"
                      "		float green = texture(mytexture, v2UVgreen).y;\n"
                      "		float blue = texture(mytexture, v2UVblue).z;\n"
                      "		outputColor = vec4( red, green, blue, 1.0  ); }\n"
                      "}\n");

  return controllerTransformProgramID != 0 && renderModelProgramID != 0 && lensProgramID != 0;
}

// ------------------ Main rendering loop -----------------------

void SteamVRRenderer::render() {
  // for now as fast as possible
  if (hmd) {
    drawControllers();
    renderStereoTargets();
    renderDistortion();

    vr::Texture_t leftEyeTexture = {(void *)leftEyeDesc.resolveTextureId, vr::TextureType_OpenGL, vr::ColorSpace_Gamma};
    vr::VRCompositor()->Submit(vr::Eye_Left, &leftEyeTexture);
    vr::Texture_t rightEyeTexture = {(void *)rightEyeDesc.resolveTextureId, vr::TextureType_OpenGL,
                                     vr::ColorSpace_Gamma};
    vr::VRCompositor()->Submit(vr::Eye_Right, &rightEyeTexture);
  }

  // Spew out the controller and pose count whenever they change.
  if (trackedControllerCount != trackedControllerCount_Last || validPoseCount != validPoseCount_Last) {
    validPoseCount_Last = validPoseCount;
    trackedControllerCount_Last = trackedControllerCount;
    QString mess;
    mess.append("PoseCount:")
        .append(validPoseCount)
        .append("(")
        .append(poseClasses.c_str())
        .append(") Controllers:")
        .append(trackedControllerCount)
        .append("\n");
    emit logMessage(mess, VideoStitch::LOG_NOTICE);
  }

  updateHMDMatrixPose();
}

void SteamVRRenderer::renderStereoTargets() {
  glClearColor(0.15f, 0.15f, 0.18f, 1.0f);  // nice background color, but not black
  glEnable(GL_MULTISAMPLE);

  // Left Eye
  glBindFramebuffer(GL_FRAMEBUFFER, leftEyeDesc.renderFramebufferId);
  glViewport(0, 0, renderWidth, renderHeight);
  renderVideoFrame(vr::Eye_Left);
  glBindFramebuffer(GL_FRAMEBUFFER, 0);

  glDisable(GL_MULTISAMPLE);

  glBindFramebuffer(GL_READ_FRAMEBUFFER, leftEyeDesc.renderFramebufferId);
  glBindFramebuffer(GL_DRAW_FRAMEBUFFER, leftEyeDesc.resolveFramebufferId);

  glBlitFramebuffer(0, 0, renderWidth, renderHeight, 0, 0, renderWidth, renderHeight, GL_COLOR_BUFFER_BIT, GL_LINEAR);

  glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
  glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);

  glEnable(GL_MULTISAMPLE);

  // Right Eye
  glBindFramebuffer(GL_FRAMEBUFFER, rightEyeDesc.renderFramebufferId);
  glViewport(0, 0, renderWidth, renderHeight);
  renderVideoFrame(vr::Eye_Right);
  glBindFramebuffer(GL_FRAMEBUFFER, 0);

  glDisable(GL_MULTISAMPLE);

  glBindFramebuffer(GL_READ_FRAMEBUFFER, rightEyeDesc.renderFramebufferId);
  glBindFramebuffer(GL_DRAW_FRAMEBUFFER, rightEyeDesc.resolveFramebufferId);

  glBlitFramebuffer(0, 0, renderWidth, renderHeight, 0, 0, renderWidth, renderHeight, GL_COLOR_BUFFER_BIT, GL_LINEAR);

  glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
  glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
}

void SteamVRRenderer::renderVideoFrame(vr::Hmd_Eye eye) {
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glEnable(GL_DEPTH_TEST);

  // render the spherical frame
  {
    const Matrix4 mvp = getCurrentViewProjectionMatrix(eye);
    QMatrix4x4 qt_mvp;
    qt_mvp.setRow(0, QVector4D(mvp.get()[0], mvp.get()[4], mvp.get()[8], mvp.get()[12]));
    qt_mvp.setRow(1, QVector4D(mvp.get()[1], mvp.get()[5], mvp.get()[9], mvp.get()[13]));
    qt_mvp.setRow(2, QVector4D(mvp.get()[2], mvp.get()[6], mvp.get()[10], mvp.get()[14]));
    qt_mvp.setRow(3, QVector4D(mvp.get()[3], mvp.get()[7], mvp.get()[11], mvp.get()[15]));

    // our world does not use the same coordinate system as OpenGL
    qt_mvp.rotate(90.0, 1.0, 0.0, 0.0);
    // rotate to have the center of our rendered panorama in the window be at the center in VR
    qt_mvp.rotate(90.0, 0.0, 0.0, 1.0);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    switch (Texture::getLeft().getType()) {
      case Texture::Type::PANORAMIC:
        Texture::getLeft().latePanoramaDef();

        sphereProgram.bind();
        if (eye == vr::Eye_Right && (Texture::getLeft().id != Texture::getRight().id) &&
            Texture::getRight().id != Texture::ID_NONE) {
          Texture::getRight().latePanoramaDef();
          std::lock_guard<std::mutex> lk(*Texture::getRight().lock.get());
          glBindBuffer(GL_PIXEL_UNPACK_BUFFER, Texture::getRight().pixelBuffer);
          glBindTexture(GL_TEXTURE_2D, Texture::getRight().id);
          glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, Texture::getRight().getWidth(), Texture::getRight().getHeight(),
                          GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

          sphereProgram.setUniformValue("texture", 0);
          sphereProgram.setUniformValue("mvp_matrix", qt_mvp);
          sphereProgram.release();

          Renderer::renderSphere();
        } else if (Texture::getLeft().id != Texture::ID_NONE) {
          std::lock_guard<std::mutex> lk(*Texture::getLeft().lock.get());
          glBindBuffer(GL_PIXEL_UNPACK_BUFFER, Texture::getLeft().pixelBuffer);
          glBindTexture(GL_TEXTURE_2D, Texture::getLeft().id);
          glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, Texture::getLeft().getWidth(), Texture::getLeft().getHeight(),
                          GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

          sphereProgram.setUniformValue("texture", 0);
          sphereProgram.setUniformValue("mvp_matrix", qt_mvp);
          sphereProgram.release();

          Renderer::renderSphere();
        } else {
          placeHolderTex.bind();
        }
        break;
      case Texture::Type::CUBEMAP:
      case Texture::Type::EQUIANGULAR_CUBEMAP:
        Q_ASSERT(false);
        // XXX TODO FIXME
        break;
    }
  }

  bool isInputAvailable = hmd->IsInputAvailable();

  if (isInputAvailable) {
    // draw the controller axis lines
    glUseProgram(controllerTransformProgramID);
    glUniformMatrix4fv(controllerMatrixLocation, 1, GL_FALSE, getCurrentViewProjectionMatrix(eye).get());
    glBindVertexArray(controllerVAO);
    glDrawArrays(GL_LINES, 0, controllerVertcount);
    glBindVertexArray(0);
  }

  // ----- Render Model rendering -----
  glUseProgram(renderModelProgramID);

  for (uint32_t trackedDevice = 0; trackedDevice < vr::k_unMaxTrackedDeviceCount; ++trackedDevice) {
    if (!trackedDeviceToRenderModel[trackedDevice] || !showTrackedDevice[trackedDevice]) {
      continue;
    }

    const vr::TrackedDevicePose_t &pose = trackedDevicePose[trackedDevice];
    if (!pose.bPoseIsValid) {
      continue;
    }

    if (!isInputAvailable &&
        hmd->GetTrackedDeviceClass(trackedDevice) == vr::TrackedDeviceClass_Controller) {
      continue;
    }

    const Matrix4 &deviceToTracking = devicePose[trackedDevice];
    const Matrix4 mvp = getCurrentViewProjectionMatrix(eye) * deviceToTracking;
    glUniformMatrix4fv(renderModelMatrixLocation, 1, GL_FALSE, mvp.get());

    trackedDeviceToRenderModel[trackedDevice]->draw();
  }

  glUseProgram(0);
}

Matrix4 SteamVRRenderer::getCurrentViewProjectionMatrix(vr::Hmd_Eye eye) {
  Matrix4 mvp;
  if (eye == vr::Eye_Left) {
    mvp = mat4ProjectionLeft * mat4eyePosLeft * mat4HMDPose;
  } else if (eye == vr::Eye_Right) {
    mvp = mat4ProjectionRight * mat4eyePosRight * mat4HMDPose;
  }
  return mvp;
}

void SteamVRRenderer::renderDistortion() {
  glDisable(GL_DEPTH_TEST);
  glViewport(0, 0, windowWidth, windowHeight);

  glBindVertexArray(lensVAO);
  glUseProgram(lensProgramID);

  // render left lens (first half of index array )
  glBindTexture(GL_TEXTURE_2D, leftEyeDesc.resolveTextureId);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
  glDrawElements(GL_TRIANGLES, indexSize / 2, GL_UNSIGNED_SHORT, 0);

  // render right lens (second half of index array )
  glBindTexture(GL_TEXTURE_2D, rightEyeDesc.resolveTextureId);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
  glDrawElements(GL_TRIANGLES, indexSize / 2, GL_UNSIGNED_SHORT, (const void *)(indexSize));

  glBindVertexArray(0);
  glUseProgram(0);
}

void SteamVRRenderer::drawControllers() {
  // don't draw controllers if somebody else has input focus
  if (!hmd->IsInputAvailable()) {
    return;
  }

  std::vector<float> vertdataarray;

  controllerVertcount = 0;
  trackedControllerCount = 0;

  for (vr::TrackedDeviceIndex_t trackedDevice = vr::k_unTrackedDeviceIndex_Hmd + 1;
       trackedDevice < vr::k_unMaxTrackedDeviceCount; ++trackedDevice) {
    if (!hmd->IsTrackedDeviceConnected(trackedDevice)) {
      continue;
    }

    if (hmd->GetTrackedDeviceClass(trackedDevice) != vr::TrackedDeviceClass_Controller) {
      continue;
    }

    trackedControllerCount += 1;

    if (!trackedDevicePose[trackedDevice].bPoseIsValid) {
      continue;
    }

    const Matrix4 &mat = devicePose[trackedDevice];
    const Vector4 center = mat * Vector4(0, 0, 0, 1);

    for (int i = 0; i < 3; ++i) {
      Vector3 color(0, 0, 0);
      Vector4 point(0, 0, 0, 1);
      point[i] += 0.05f;  // offset in X, Y, Z
      color[i] = 1.0;     // R, G, B
      point = mat * point;
      vertdataarray.push_back(center.x);
      vertdataarray.push_back(center.y);
      vertdataarray.push_back(center.z);
      vertdataarray.push_back(color.x);
      vertdataarray.push_back(color.y);
      vertdataarray.push_back(color.z);
      vertdataarray.push_back(point.x);
      vertdataarray.push_back(point.y);
      vertdataarray.push_back(point.z);
      vertdataarray.push_back(color.x);
      vertdataarray.push_back(color.y);
      vertdataarray.push_back(color.z);

      controllerVertcount += 2;
    }

    Vector4 start = mat * Vector4(0, 0, -0.02f, 1);
    Vector4 end = mat * Vector4(0, 0, -39.f, 1);
    Vector3 color(.92f, .92f, .71f);
    vertdataarray.push_back(start.x);
    vertdataarray.push_back(start.y);
    vertdataarray.push_back(start.z);
    vertdataarray.push_back(color.x);
    vertdataarray.push_back(color.y);
    vertdataarray.push_back(color.z);
    vertdataarray.push_back(end.x);
    vertdataarray.push_back(end.y);
    vertdataarray.push_back(end.z);
    vertdataarray.push_back(color.x);
    vertdataarray.push_back(color.y);
    vertdataarray.push_back(color.z);

    controllerVertcount += 2;
  }

  // Setup the VAO the first time through.
  if (controllerVAO == 0) {
    glGenVertexArrays(1, &controllerVAO);
    glBindVertexArray(controllerVAO);

    glGenBuffers(1, &glControllerVertBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, glControllerVertBuffer);

    const GLuint stride = 2 * 3 * sizeof(float);
    GLuint offset = 0;

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, (const void *)offset);

    offset += sizeof(Vector3);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, (const void *)offset);

    glBindVertexArray(0);
  }

  glBindBuffer(GL_ARRAY_BUFFER, glControllerVertBuffer);

  // set vertex data if we have some
  if (vertdataarray.size() > 0) {
    //$ TODO: Use glBufferSubData for this...
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * vertdataarray.size(), &vertdataarray[0], GL_STREAM_DRAW);
  }
}

// ------------------------- Tracking ----------------------

void SteamVRRenderer::updateHMDMatrixPose() {
  if (!hmd) {
    return;
  }

  vr::VRCompositor()->WaitGetPoses(trackedDevicePose, vr::k_unMaxTrackedDeviceCount, nullptr, 0);

  // We just got done with the glFinish - the seconds since last vsync should be 0.
  const float secondsSinceLastVsync = 0.0f;
  const float frameDuration =
      1.0f / hmd->GetFloatTrackedDeviceProperty(vr::k_unTrackedDeviceIndex_Hmd, vr::Prop_DisplayFrequency_Float);
  const float secondsUntilPhotons =
      frameDuration - secondsSinceLastVsync +
      hmd->GetFloatTrackedDeviceProperty(vr::k_unTrackedDeviceIndex_Hmd, vr::Prop_SecondsFromVsyncToPhotons_Float);
  hmd->GetDeviceToAbsoluteTrackingPose(vr::TrackingUniverseStanding, secondsUntilPhotons, trackedDevicePose,
                                       vr::k_unMaxTrackedDeviceCount);

  validPoseCount = 0;
  poseClasses.clear();
  for (int device = 0; device < vr::k_unMaxTrackedDeviceCount; ++device) {
    if (trackedDevicePose[device].bPoseIsValid) {
      validPoseCount++;
      devicePose[device] = convertSteamVRMatrixToMatrix4(trackedDevicePose[device].mDeviceToAbsoluteTracking);
      if (devClassChar[device] == 0) {
        switch (hmd->GetTrackedDeviceClass(device)) {
          case vr::TrackedDeviceClass_Controller:
            devClassChar[device] = 'C';
            break;
          case vr::TrackedDeviceClass_HMD:
            devClassChar[device] = 'H';
            break;
          case vr::TrackedDeviceClass_Invalid:
            devClassChar[device] = 'I';
            break;
          case vr::TrackedDeviceClass_GenericTracker:
            devClassChar[device] = 'O';
            break;
          case vr::TrackedDeviceClass_TrackingReference:
            devClassChar[device] = 'T';
            break;
          default:
            devClassChar[device] = '?';
            break;
        }
      }
      poseClasses += devClassChar[device];
    }
  }

  if (trackedDevicePose[vr::k_unTrackedDeviceIndex_Hmd].bPoseIsValid) {
    mat4HMDPose = devicePose[vr::k_unTrackedDeviceIndex_Hmd].invert();
  }
}

Matrix4 SteamVRRenderer::convertSteamVRMatrixToMatrix4(const vr::HmdMatrix34_t &matPose) {
  Matrix4 matrixObj(matPose.m[0][0], matPose.m[1][0], matPose.m[2][0], 0.0, matPose.m[0][1], matPose.m[1][1],
                    matPose.m[2][1], 0.0, matPose.m[0][2], matPose.m[1][2], matPose.m[2][2], 0.0, matPose.m[0][3],
                    matPose.m[1][3], matPose.m[2][3], 1.0f);
  return matrixObj;
}
