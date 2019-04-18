// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "renderer.hpp"
#include "vectors.hpp"
#include "matrices.hpp"
#include "common-config.hpp"
#include <openvr.h>

using namespace lineag;

class CGLRenderModel : public VideoStitchQOpenGLFunctions {
 public:
  explicit CGLRenderModel(const std::string &renderModelName);
  ~CGLRenderModel();

  bool init(const vr::RenderModel_t &vrModel, const vr::RenderModel_TextureMap_t &vrDiffuseTexture);
  void cleanup();
  void draw();
  const std::string &getName() const { return modelName; }

 private:
  GLuint glVertBuffer;
  GLuint glIndexBuffer;
  GLuint glVertArray;
  GLuint glTexture;
  GLsizei vertexCount;
  std::string modelName;
};

class VS_COMMON_EXPORT SteamVRRenderer : public Renderer {
  Q_OBJECT
 public:
  SteamVRRenderer();
  ~SteamVRRenderer();

  bool initializeSteamVR();
  void uninitializeSteamVR();
  void render();
  void configureRendering(int width, int height);

 private:
  struct FramebufferDesc {
    GLuint depthBufferId;
    GLuint renderTextureId;
    GLuint renderFramebufferId;
    GLuint resolveTextureId;
    GLuint resolveFramebufferId;
  };

  struct VertexDataLens {
    Vector2 position;
    Vector2 texCoordRed;
    Vector2 texCoordGreen;
    Vector2 texCoordBlue;
  };

  // -- Setup

  void setupCameras();
  bool setupStereoRenderTargets();
  void setupDistortion();
  void setupRenderModels();
  void setupRenderModelForTrackedDevice(vr::TrackedDeviceIndex_t trackedDeviceIndex);

  GLuint compileGLShader(const char *shaderName, const char *vertexShader, const char *fragmentShader);
  bool createAllShaders();

  Matrix4 getHMDMatrixProjectionEye(vr::Hmd_Eye);
  Matrix4 getHMDMatrixPoseEye(vr::Hmd_Eye);
  bool createFrameBuffer(int w, int h, FramebufferDesc &);
  CGLRenderModel *findOrLoadRenderModel(const char *renderModelName);

  // -- Rendering

  void drawControllers();
  void renderStereoTargets();
  void renderVideoFrame(vr::Hmd_Eye);
  void renderDistortion();
  Matrix4 getCurrentViewProjectionMatrix(vr::Hmd_Eye);

  // -- Tracking

  void updateHMDMatrixPose();
  Matrix4 convertSteamVRMatrixToMatrix4(const vr::HmdMatrix34_t &matPose);

  // -- State

  vr::IVRSystem *hmd;
  vr::IVRRenderModels *renderModels;
  vr::TrackedDevicePose_t trackedDevicePose[vr::k_unMaxTrackedDeviceCount];
  Matrix4 devicePose[vr::k_unMaxTrackedDeviceCount];
  bool showTrackedDevice[vr::k_unMaxTrackedDeviceCount];

  uint32_t windowWidth;
  uint32_t windowHeight;

  bool openglInitialized;

  GLuint lensProgramID;
  GLuint controllerTransformProgramID;
  GLuint renderModelProgramID;

  GLint controllerMatrixLocation;
  GLint renderModelMatrixLocation;

  uint32_t renderWidth;
  uint32_t renderHeight;
  float nearClip;
  float farClip;

  GLuint lensVAO;
  GLuint glIDVertBuffer;
  GLuint glIDIndexBuffer;
  unsigned int indexSize;

  Matrix4 mat4HMDPose;
  Matrix4 mat4eyePosLeft;
  Matrix4 mat4eyePosRight;

  Matrix4 mat4ProjectionCenter;
  Matrix4 mat4ProjectionLeft;
  Matrix4 mat4ProjectionRight;

  FramebufferDesc leftEyeDesc;
  FramebufferDesc rightEyeDesc;

  std::vector<CGLRenderModel *> vecRenderModels;
  CGLRenderModel *trackedDeviceToRenderModel[vr::k_unMaxTrackedDeviceCount];

  // controllers rendering
  int trackedControllerCount = 0;
  int trackedControllerCount_Last = -1;
  int validPoseCount = 0;
  int validPoseCount_Last = -1;
  GLuint glControllerVertBuffer;
  GLuint controllerVAO;
  unsigned int controllerVertcount;

  // tracking
  std::string poseClasses;                           // what classes we saw poses for this frame
  char devClassChar[vr::k_unMaxTrackedDeviceCount];  // for each device, a character representing its class
};
