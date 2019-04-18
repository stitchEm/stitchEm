// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "libvideostitch/allocator.hpp"

#include <QElapsedTimer>
#include <QOpenGLWidget>
#include <QOpenGLFunctions>
#include <QOpenGLTexture>

#include <mutex>
#include <vector>

class VS_GUI_EXPORT GenericVideoWidget : public QOpenGLWidget,
                                         public QOpenGLFunctions,
                                         public VideoStitch::Core::SourceRenderer {
  Q_OBJECT
 public:
  explicit GenericVideoWidget(QWidget* parent = nullptr);
  virtual ~GenericVideoWidget();

  void clearTextures();

  /**
   * @brief setName
   * @param n
   */
  void setName(const std::string& n);

  /**
   * @brief getName
   * @return
   */
  std::string getName() const override;

  /**
   * @brief Returns the number of textures to render.
   * @return Number of textures.
   */
  int getNumbTextures();

 signals:
  void gotFrame(mtime_t);

 protected:
  void initializeGL() override;
  void resizeGL(int w, int h) override;
  void paintFrame(GLuint texture, float quadWidth, float quadHeight, float verticalMargin, float horizontalMargin);

  std::mutex textureMutex;
  std::map<int, std::shared_ptr<VideoStitch::Core::SourceOpenGLSurface>> textures;
  mtime_t ref_vs;  // the initial time of the sequence (playing from a seek point)
  QOpenGLTexture placeHolderTex;

  QElapsedTimer clk;
  std::string name;
};
