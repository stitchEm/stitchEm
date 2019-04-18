// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "libvideostitch-gui/utils/panoutilities.hpp"

#include <QDialog>

namespace Ui {
class ResetDimensionsDialog;
}

/**
 * @brief Class used when the stitcher fails to init due to a lack of memory. It prompts the user to type another size
 * for its panorama.
 */
class VS_GUI_EXPORT ResetDimensionsDialog : public QDialog {
  Q_OBJECT

 public:
  /**
   * @brief Constructor of the Reset dimension dialog. Using references to the pano dimensions, and has the bounds of
   * these values in parameters
   * @param ptvWhichFailed Path of the ptv which failed to load
   * @param initialPanoWidth Initial panorama width
   * @param initialPanoHeight Initial panorama height
   * @param parent Parent widget (needed for heap allocation)
   */
  explicit ResetDimensionsDialog(const QString ptvWhichFailed, unsigned int initialPanoWidth,
                                 unsigned int initialPanoHeight, QWidget *parent);
  ~ResetDimensionsDialog();

  unsigned int getNewPanoWidth() const;
  unsigned int getNewPanoHeight() const;

 private:
  void updateSizeSpinBoxes(VideoStitch::Util::PanoSize size);

 private slots:
  void onWidthChanged();
  void onHeightChanged();

 private:
  QScopedPointer<Ui::ResetDimensionsDialog> ui;
};
