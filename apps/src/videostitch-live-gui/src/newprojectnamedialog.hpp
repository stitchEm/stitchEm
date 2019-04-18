// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef NEWPROJECTNAMEDIALOG_H
#define NEWPROJECTNAMEDIALOG_H

#include "generic/genericdialog.hpp"

class QLineEdit;
class NewProjectNameDialog : public GenericDialog {
  Q_OBJECT
 public:
  explicit NewProjectNameDialog(QWidget* const parent = nullptr);

  ~NewProjectNameDialog();

 private:
  QLineEdit* lineName;

  QSpacerItem* spacerTop;

 private slots:
  void onButtonAcceptNameClicked();

  void onNameEdited(const QString& name);

 signals:
  void notifySetProjectName(const QString& name);
};

#endif  // NEWPROJECTNAMEDIALOG_H
