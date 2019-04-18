// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "resizepanoramadialog.hpp"
#include "configurations/resizepanoramawidget.hpp"

ResizePanoramaDialog::ResizePanoramaDialog(const unsigned width, const unsigned height, QWidget* const parent)
    : GenericDialog(tr("Critical error"), QString(), GenericDialog::DialogMode::NEXT, parent),
      resetWidget(new ResizePanoramaWidget(width, height, this)) {
  dialogLayout->insertWidget(1, resetWidget);

  connect(buttonAccept, &QPushButton::clicked, this, &ResizePanoramaDialog::onAcceptClicked);
  connect(buttonNext, &QPushButton::clicked, this, &ResizePanoramaDialog::onNextClicked);
  connect(buttonCancel, &QPushButton::clicked, this, &ResizePanoramaDialog::onCancelClicked);
}

void ResizePanoramaDialog::onAcceptClicked() {
  emit notifyPanoValuesSet(resetWidget->panoSizeSelector->getWidth(), resetWidget->panoSizeSelector->getHeight());
  close();
}

void ResizePanoramaDialog::onNextClicked() {
  resetWidget->stackedWidget->setCurrentIndex(1);
  setDialogMode(DialogMode::ACCEPT_CANCEL);
}

void ResizePanoramaDialog::onCancelClicked() {
  buttonAccept->hide();
  buttonNext->show();
  close();
}
