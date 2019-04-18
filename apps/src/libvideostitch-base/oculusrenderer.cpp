// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "oculusrenderer.hpp"

#include "texture.hpp"

#include <Extras/OVR_Math.h>

#include <QTimer>

OculusRenderer::OculusRenderer() : Renderer(), mirrorTexture(nullptr), nbRenderedFrames(0), openglInitialized(false) {}

bool OculusRenderer::initializeOculus() {
  if (ovr_Initialize(nullptr) != ovrSuccess) {
    ovrErrorInfo err;
    ovr_GetLastErrorInfo(&err);
    qDebug() << err.ErrorString;
    return false;
  }
  ovrResult result = ovr_Create(&session, &luid);
  if (result != ovrSuccess) {
    ovr_Shutdown();
    qDebug() << "Cannot create the debug HMD";
    return false;
  }

  hmdDesc = ovr_GetHmdDesc(session);
  eyeRenderDesc[0] = ovr_GetRenderDesc(session, ovrEye_Left, hmdDesc.DefaultEyeFov[0]);
  eyeRenderDesc[1] = ovr_GetRenderDesc(session, ovrEye_Right, hmdDesc.DefaultEyeFov[1]);

  // Get eye poses, feeding in correct IPD offset
  if (Texture::getLeft().id == Texture::getRight().id) {
    // For monoscopic rendering,
    // use a vector that is the average of the two vectors for both eyes.
    viewOffset[ovrEye_Left] = (OVR::Vector3f(eyeRenderDesc[ovrEye_Left].HmdToEyeOffset) +
                               OVR::Vector3f(eyeRenderDesc[ovrEye_Right].HmdToEyeOffset)) /
                              2;
    viewOffset[ovrEye_Right] = viewOffset[ovrEye_Left];
  } else {
    viewOffset[ovrEye_Left] = eyeRenderDesc[ovrEye_Left].HmdToEyeOffset;
    viewOffset[ovrEye_Right] = eyeRenderDesc[ovrEye_Right].HmdToEyeOffset;
  }

  return true;
}

OculusRenderer::~OculusRenderer() {
  if (!openglInitialized) {
    return;
  }

  if (mirrorTexture) {
    ovr_DestroyMirrorTexture(session, mirrorTexture);
    mirrorTexture = nullptr;
  }

  for (int eye = 0; eye < 2; ++eye) {
    delete eyeRenderTexture[eye];
    delete eyeDepthBuffer[eye];
  }

  ovr_Destroy(session);
  ovr_Shutdown();
}

void OculusRenderer::configureRendering(int width, int height, bool mirror) {
  Renderer::initialize();

  // Make eye render buffers
  for (int i = 0; i < 2; i++) {
    ovrSizei idealTextureSize = ovr_GetFovTextureSize(session, (ovrEyeType)i, hmdDesc.DefaultEyeFov[i], 1);
    eyeRenderTexture[i] = new TextureBuffer(session, idealTextureSize);
    eyeDepthBuffer[i] = new DepthBuffer(eyeRenderTexture[i]->GetSize());
  }

  if (mirror) {
    ovrMirrorTextureDesc desc;
    desc.Format = OVR_FORMAT_R8G8B8A8_UNORM_SRGB;
    desc.Width = width;
    desc.Height = height;
    ovr_CreateMirrorTextureGL(session, &desc, &mirrorTexture);
  }

  Q_ASSERT(!glGetError());
  openglInitialized = true;
  emit renderingConfigured();
}

static const OVR::Vector3f eyePosition(0.0f, 0.0f, 0.0f);

void OculusRenderer::render() {
  // compute the anticipated pose
  ovrPosef eyeRenderPose[2];
  double sensorSampleTime = 0;
  ovr_GetEyePoses(session, 0, ovrTrue, viewOffset, eyeRenderPose, &sensorSampleTime);
  notifyOrientation(eyeRenderPose[0].Orientation);

  for (int eye = 0; eye < 2; eye++) {
    // switch to eye render target
    eyeRenderTexture[eye]->SetAndClearRenderSurface(eyeDepthBuffer[eye]);

    // compute the view and projection, convert to Qt matrix
    OVR::Matrix4f rollPitchYaw = OVR::Matrix4f(eyeRenderPose[eye].Orientation);
    OVR::Vector3f up = rollPitchYaw.Transform(OVR::Vector3f(0, 1, 0));
    OVR::Vector3f forward = rollPitchYaw.Transform(OVR::Vector3f(0, 0, -1));
    OVR::Matrix4f view = OVR::Matrix4f::LookAtRH(eyePosition, forward, up);

    OVR::Matrix4f proj = ovrMatrix4f_Projection(eyeRenderDesc[eye].Fov, 0.2f, 1000.0f, ovrProjection_None);

    // our world does not use the same coordinate system as OpenGL
    OVR::Matrix4f mvp = proj * view * OVR::Matrix4f::RotationX((float)M_PI_2);

    QMatrix4x4 mvp_matrix;
    mvp_matrix.setRow(0, QVector4D(mvp.M[0][0], mvp.M[0][1], mvp.M[0][2], mvp.M[0][3]));
    mvp_matrix.setRow(1, QVector4D(mvp.M[1][0], mvp.M[1][1], mvp.M[1][2], mvp.M[1][3]));
    mvp_matrix.setRow(2, QVector4D(mvp.M[2][0], mvp.M[2][1], mvp.M[2][2], mvp.M[2][3]));
    mvp_matrix.setRow(3, QVector4D(mvp.M[3][0], mvp.M[3][1], mvp.M[3][2], mvp.M[3][3]));
    // rotate to have the center of our rendered panorama in the window be at the center in VR
    mvp_matrix.rotate(90.0, 0.0, 0.0, 1.0);

    // render the spherical frame
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    switch (Texture::getLeft().getType()) {
      case Texture::Type::PANORAMIC:
        sphereProgram.bind();
        Texture::getLeft().latePanoramaDef();
        if (eye && (Texture::getLeft().id != Texture::getRight().id) && Texture::getRight().id != Texture::ID_NONE) {
          Texture::getRight().latePanoramaDef();
          std::lock_guard<std::mutex> lk(*Texture::getRight().lock.get());
          glBindBuffer(GL_PIXEL_UNPACK_BUFFER, Texture::getRight().pixelBuffer);
          glBindTexture(GL_TEXTURE_2D, Texture::getRight().id);
          glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, Texture::getRight().getWidth(), Texture::getRight().getHeight(),
                          GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

          sphereProgram.setUniformValue("texture", 0);
          sphereProgram.setUniformValue("mvp_matrix", mvp_matrix);
          sphereProgram.release();

          Renderer::renderSphere();
        } else if (Texture::getLeft().id != Texture::ID_NONE) {
          std::lock_guard<std::mutex> lk(*Texture::getLeft().lock.get());
          glBindBuffer(GL_PIXEL_UNPACK_BUFFER, Texture::getLeft().pixelBuffer);
          glBindTexture(GL_TEXTURE_2D, Texture::getLeft().id);
          glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, Texture::getLeft().getWidth(), Texture::getLeft().getHeight(),
                          GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

          sphereProgram.setUniformValue("texture", 0);
          sphereProgram.setUniformValue("mvp_matrix", mvp_matrix);
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

    eyeRenderTexture[eye]->UnsetRenderSurface();
    eyeRenderTexture[eye]->commitTexture();
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
  }

  ovrViewScaleDesc viewScaleDesc;
  viewScaleDesc.HmdSpaceToWorldScaleInMeters = 1.0f;
  viewScaleDesc.HmdToEyeOffset[0] = viewOffset[0];
  viewScaleDesc.HmdToEyeOffset[1] = viewOffset[1];

  ovrLayerEyeFov ld;
  ld.Header.Type = ovrLayerType_EyeFov;
  ld.Header.Flags = ovrLayerFlag_TextureOriginAtBottomLeft;  // Because OpenGL.
  ld.SensorSampleTime = sensorSampleTime;

  for (int eye = 0; eye < 2; eye++) {
    ld.ColorTexture[eye] = eyeRenderTexture[eye]->textureSwapChain;
    ld.Viewport[eye] = OVR::Recti(eyeRenderTexture[eye]->GetSize());
    ld.Fov[eye] = hmdDesc.DefaultEyeFov[eye];
    ld.RenderPose[eye] = eyeRenderPose[eye];
  }

  ovrLayerHeader* layers = &ld.Header;
  ovr_SubmitFrame(session, 0, &viewScaleDesc, &layers, 1);

  ++nbRenderedFrames;
  float fps = nbRenderedFrames / (float)renderTimer.elapsed() * 1000.0f;
  emit fpsChanged(QString::number(fps, 'f', 0).rightJustified(2, '0'));
}

void OculusRenderer::notifyOrientation(ovrQuatf newOrientation) {
  OVR::Quatf orientation(newOrientation);
  float yaw = 0.0f;
  float pitch = 0.0f;
  float roll = 0.0f;
  orientation.GetEulerAngles<OVR::Axis_Y, OVR::Axis_X, OVR::Axis_Z>(&yaw, &pitch, &roll);
  emit orientationChanged(yaw + M_PI, -pitch, roll);
}

GLuint OculusRenderer::getMirrorTextureId() const {
  GLuint texId = 0u;
  ovr_GetMirrorTextureBufferGL(session, mirrorTexture, &texId);
  return texId;
}
