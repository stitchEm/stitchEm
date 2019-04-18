// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once
#include <QWidget>
#include <QListWidgetItem>
#include <memory>

#include "computationwidget.hpp"
#include "itoolwidget.hpp"
#include "libvideostitch/inputDef.hpp"

namespace Ui {
class BlendingMaskWidget;
}

class BlendingMaskWidget : public ComputationWidget, public IToolWidget {
  Q_OBJECT

 public:
  explicit BlendingMaskWidget(QWidget* parent = nullptr);
  ~BlendingMaskWidget();
 public slots:
  virtual void onProjectOpened(ProjectDefinition* project) override;
  virtual void updateSequence(const QString, const QString) override {}
  void startBlendingMaskComputation();
  void onEnableBlendingMaskChecked(const bool show);
  void onEnableInterpolationChecked(const bool show);
  VideoStitch::Status* computeBlendingMask(std::shared_ptr<VideoStitch::Ptv::Value> blendingMaskConfig);
  void refresh(mtime_t date);
  void removeFrame();
  void clearFrames();
  void clearListFrame();
  void seekFrame(QListWidgetItem* item);
  void setFrame(frameid_t frame);
  void frameItemSelected();
  void setFrameText();

 signals:
  void reqApplyBlendingMask(VideoStitch::Core::PanoDefinition* panoDef = nullptr);
  /**
   * @brief Sends a signal to seek to the frame given in parameter
   * @param frameToSeekTo Frame to seek to.
   */
  void reqSeek(frameid_t frameToSeekTo);

 protected:
  virtual QString getAlgorithmName() const override;
  virtual void manageComputationResult(bool hasBeenCancelled, VideoStitch::Status* status) override;
  virtual void reset() override {}

 private:
  void addFrame(const frameid_t frame);

  VideoStitch::Core::PanoDefinition* panoDef;
  VideoStitch::Core::PanoDefinition* oldPanoDef;
  frameid_t currentFrame;
  Ui::BlendingMaskWidget* ui;
};
