// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "genericvideowidget.hpp"

class VS_GUI_EXPORT MultiVideoWidget : public GenericVideoWidget {
  Q_OBJECT
 public:
  explicit MultiVideoWidget(QWidget *parent = nullptr);
  virtual ~MultiVideoWidget();

  void syncOn();
  void syncOff();

  virtual void paintGL() override;

  /**
   * @brief render
   */
  virtual void render(std::shared_ptr<VideoStitch::Core::SourceOpenGLSurface> surf, mtime_t date) override;

 private:
  bool sync = true;
};
