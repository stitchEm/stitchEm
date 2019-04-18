// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "genericvideowidget.hpp"

/**
 * @brief Renders a single input source.
 */
class VS_GUI_EXPORT SingleVideoWidget : public GenericVideoWidget {
  Q_OBJECT
 public:
  /**
   * @brief Creates the render for a given input source.
   * @param id The input source id.
   * @param parent The parent widget.
   */
  explicit SingleVideoWidget(const int id, QWidget *parent = nullptr);
  virtual ~SingleVideoWidget();

  virtual void paintGL() override;

  /**
   * @brief render
   */
  void render(std::shared_ptr<VideoStitch::Core::SourceOpenGLSurface> surf, mtime_t date) override;

 private:
  const int inputId;
};
