// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "libvideostitch-gui/centralwidget/ifreezablewidget.hpp"
#include "libvideostitch-gui/centralwidget/icentraltabwidget.hpp"
#include "libvideostitch-gui/widgets/stylablewidget.hpp"

#include "libvideostitch/allocator.hpp"
#include "libvideostitch/input.hpp"
#include "../common.hpp"

#include <QDragMoveEvent>

#define HIDE_MASK_FEATURE

namespace Ui {
class SourceWidget;
}

class ProjectDefinition;
class MultiVideoWidget;
class SignalCompressionCaps;

/**
 * @brief Widget that shall be displayed in the source tab.
 */
class VS_GUI_EXPORT SourceWidget : public IFreezableWidget, public ICentralTabWidget {
  Q_OBJECT
  Q_MAKE_STYLABLE
  Q_PROPERTY(int numbSources READ getNumbSources)

 public:
  explicit SourceWidget(const bool drops, QWidget *parent = nullptr);
  ~SourceWidget();

  MultiVideoWidget &getMultiVideoWidget();
  int getNumbSources() const;
  virtual bool allowsPlayback() const override { return true; }

 public slots:

  /**
   * @brief Clears the current rendered thumbnails.
   */
  void clearThumbnails();

  /**
   * @brief Create all the thumbnail according to the readers.
   * @param inputs List of inputs description.
   * @param renderers List of renderers.
   */
  void createThumbnails(std::vector<std::tuple<VideoStitch::Input::VideoReader::Spec, bool, std::string>> inputs,
                        std::vector<std::shared_ptr<VideoStitch::Core::SourceRenderer>> *renderers);

  void setProject(ProjectDefinition *p);
  void clearProject();

 protected:
  /**
   * @brief Slot called when files are dropped into the widget.
   * @param e Drop event that should be processed.
   */
  virtual void dropEvent(QDropEvent *e) override { emit sendDropEvent(e); }

  /**
   * @brief Slot called a drag event has moved in the widget.
   * @param e Drag event that should be processed.
   */
  virtual void dragMoveEvent(QDragMoveEvent *e) override { e->accept(); }

  /**
   * @brief Slot called when a drag event entered into the widget.
   * @param e Drag event that should be processed.
   */
  virtual void dragEnterEvent(QDragEnterEvent *e) override { e->acceptProposedAction(); }

  /**
   * @brief Changes the widget's stats to the given state.
   * @param s State you want to switch to.
   */
  void changeState(GUIStateCaps::State s) override;

 signals:
  void reqUpdateMasks();
  void sendDropEvent(QDropEvent *);
  void notifyUploadError(const VideoStitch::Status &status, bool needToExit) const;
  void reqResetDimensions(unsigned panoramaWidth, unsigned panoramaHeight, const QStringList &inputNames);
  void reqReextract(SignalCompressionCaps * = nullptr);

 protected slots:
  inline void clearScreenshot() override {}

 protected:
  inline void freeze() override {}
  inline void unfreeze() override {}
  inline void showGLView() override {}
  inline void connectToDeviceWriter() override {}

 private slots:
  /**
   * @brief Callback called when the masl have been toggle back on or off.
   * @param show true = the masks should be displayed / false = hide the masks.
   */
  void maskToggled(bool show);  // TODO: VSA-486 implement this
  /**
   * @brief Updates the mask according to the data sent by the extractor.
   * @param index Index of the display on which the mask should be overlayed.
   * @param maskData Raw pixel data representing the mask.
   * @param width Width of the picture.
   * @param height Height of the picture.
   */
  void updateMask(int index, unsigned char *maskData, int width, int height);
  /**
   * @brief Updates the mask according to the QImage sent from the extractor
   * @param index Index of the thumbnail you need to print the mask on.
   * @param mask Mask you need to print.
   */
  void updateMask(int index, QImage *mask);
  void onUploaderError(const VideoStitch::Status &errorStatus, bool needToExit);

 private:
  Ui::SourceWidget *ui;
  ProjectDefinition *project;
  std::shared_ptr<MultiVideoWidget> viewPtr;
};
