// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "libvideostitch-gui/common.hpp"
#include "libvideostitch-gui/videostitcher/projectdefinition.hpp"
#include "libvideostitch/allocator.hpp"
#include "libvideostitch/input.hpp"
#include "libvideostitch-gui/widgets/singlevideowidget.hpp"

namespace Ui {
class CropWidget;
}

class ProjectDefinition;
class CropInputTab;
class QHBoxLayout;

/**
 * @brief A widget to manager inputs crop edition.
 */
class VS_GUI_EXPORT CropWidget : public QWidget {
  Q_OBJECT

 public:
  explicit CropWidget(QWidget* parent = nullptr);
  /**
   * @brief Constructor
   * @param p A ProjectDefinition pointer.
   * @param f An input format.
   * @param parent A parent widget.
   */
  explicit CropWidget(ProjectDefinition* p, InputLensClass::LensType t, const int extended, QWidget* parent = nullptr);
  ~CropWidget();

  /**
   * @brief Creates one tab per input and sets up the input extractors.
   */
  void initializeTabs();
  /**
   * @brief Deletes tha tabs and removes the input extractors.
   */
  void deinitializeTabs();

  /**
   * @brief Sets the current crop value.
   */
  void applyCrop();

  /**
   * @brief Resets every input crop value.
   */
  void setDefaultCrop();

  void setProject(ProjectDefinition* p);
  void setLensType(InputLensClass::LensType t);
  void setWidgetExtension(int extended);

  QHBoxLayout* getHorizontalLayout();

 signals:
  /**
   * @brief Applies a list of crop values to the inputs.
   * @param crops A list of crops. Should contain the same amount of elements than the inputs.
   * @param lensType The lens type.
   */
  void reqApplyCrops(const QVector<Crop>& crops, const InputLensClass::LensType lensType);

  void reqRegisterRender(std::shared_ptr<VideoStitch::Core::SourceRenderer> renderer, const int inputId);

  void reqUnregisterRender(const QString name, const int inputId);

  void reextract(SignalCompressionCaps* = nullptr);

 private slots:
  /**
   * @brief Locks or unlocks the current tab crop values into the other tabs.
   * @param block True for blocking.
   */
  void onCropLocked(const bool block);

  void onTabChanged(const int index);

 private:
  void disconnectFrom(CropInputTab* mainInputTab);
  void connectTo(CropInputTab* mainInputTab);
  int getAvailableHeight() const;
  CropInputTab* getTabWidget(const int index) const;
  QVector<Crop> getCrops() const;

  InputLensClass::LensType lensType;
  QScopedPointer<Ui::CropWidget> ui;
  ProjectDefinition* project;
  int height;
  int blockIndex;
  int oldFrameAction;
};
