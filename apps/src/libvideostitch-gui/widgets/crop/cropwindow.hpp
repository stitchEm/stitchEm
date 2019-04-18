// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "cropwidget.hpp"

#include <QDialog>

/**
 * @brief A widget to manager inputs crop edition.
 */
class VS_GUI_EXPORT CropWindow : public QDialog {
  Q_OBJECT

 public:
  /**
   * @brief Constructor
   * @param p A ProjectDefinition pointer.
   * @param f An input format.
   * @param parent A parent widget.
   */
  explicit CropWindow(ProjectDefinition* p, InputLensClass::LensType t, const int extended, QWidget* parent = nullptr);
  ~CropWindow();

  CropWidget& getCropWidget();

 protected:
  virtual void accept() override;
  virtual void reject() override;

 private:
  CropWidget cropWidget;
};
