// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef VSPATHEDIT_HPP
#define VSPATHEDIT_HPP
#include <QLineEdit>
#include <QCompleter>
#include <QDirModel>

/**
 * @brief Class which derivates from QLineEdit and which is used to display an auto-completing field when the user types
 * a path.
 */
class VS_GUI_EXPORT VSPathEdit : public QLineEdit {
  Q_OBJECT
 public:
  explicit VSPathEdit(QWidget* parent = 0) : QLineEdit(parent) {
    pathCompleter = new QCompleter(this);
    dirModel = new QDirModel(pathCompleter);
    pathCompleter->setModel(dirModel);
    setCompleter(pathCompleter);
  }

 private:
  QCompleter* pathCompleter;
  QDirModel* dirModel;
};

#endif  // VSPATHEDIT_HPP
