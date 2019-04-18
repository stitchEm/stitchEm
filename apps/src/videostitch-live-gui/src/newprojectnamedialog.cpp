// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "newprojectnamedialog.hpp"
#include "guiconstants.hpp"
#include <QLineEdit>

static const unsigned int NAME_MAX_CHARS(30);

NewProjectNameDialog::NewProjectNameDialog(QWidget* const parent)
    : GenericDialog(tr("New project name:"), QString(), GenericDialog::DialogMode::ACCEPT_CANCEL, parent),
      lineName(new QLineEdit(this)),
      spacerTop(new QSpacerItem(0, BUTTON_SIDE, QSizePolicy::Expanding, QSizePolicy::Fixed)) {
  connect(this, &GenericDialog::notifyAcceptClicked, this, &NewProjectNameDialog::onButtonAcceptNameClicked);
  connect(lineName, &QLineEdit::textEdited, this, &NewProjectNameDialog::onNameEdited);
  lineName->setObjectName("lineName");
  lineName->setFixedHeight(LINEEDIT_HEIGHT);
  lineName->setMaxLength(NAME_MAX_CHARS);
  lineName->setFocus();
  dialogLayout->insertSpacerItem(1, spacerTop);
  dialogLayout->insertWidget(2, lineName);
  buttonAccept->setEnabled(false);
}

NewProjectNameDialog::~NewProjectNameDialog() {}

void NewProjectNameDialog::onButtonAcceptNameClicked() { emit notifySetProjectName(lineName->text()); }

void NewProjectNameDialog::onNameEdited(const QString&) { buttonAccept->setEnabled(!lineName->text().isEmpty()); }
