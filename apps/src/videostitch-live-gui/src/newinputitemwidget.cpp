// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "newinputitemwidget.hpp"
#include "ui_newinputitemwidget.h"
#include <QMovie>

static const QString LOADING_ICON(":/live/icons/assets/icon/live/loadconnecting.gif");
static const QString OK_ICON(":/live/icons/assets/icon/live/check.png");
static const QString FAIL_ICON(":/live/icons/assets/icon/live/close.png");

NewInputItemWidget::NewInputItemWidget(const QString url, const int id, QWidget* const parent)
    : QWidget(parent),
      ui(new Ui::NewInputItemWidget),
      widgetId(id),
      movieLoading(new QMovie(LOADING_ICON, nullptr, this)) {
  ui->setupUi(this);
  ui->lineURL->setText(url);
  ui->labelField->setText(tr("URL %0:").arg(id + 1));
  connect(ui->lineURL, &QLineEdit::textChanged, this, &NewInputItemWidget::onUrlChanged);
}

NewInputItemWidget::~NewInputItemWidget() { delete ui; }

bool NewInputItemWidget::hasValidUrl() const { return urlCheckStatus == UrlStatus::Verified; }

QByteArray NewInputItemWidget::getUrl() const { return ui->lineURL->text().toLatin1(); }

int NewInputItemWidget::getId() const { return widgetId; }

void NewInputItemWidget::setUrl(const QString url) { ui->lineURL->setText(url); }

void NewInputItemWidget::onTestFinished(const bool success) {
  ui->labelStatusIcon->setPixmap(success ? QPixmap(OK_ICON) : QPixmap(FAIL_ICON));
  movieLoading->stop();
  ui->lineURL->setEnabled(true);
  urlCheckStatus = (success ? UrlStatus::Verified : UrlStatus::Failed);
  emit notifyUrlValidityChanged();
}

void NewInputItemWidget::onTestClicked() {
  ui->labelStatusIcon->show();
  ui->labelStatusIcon->setMovie(movieLoading);
  ui->lineURL->setEnabled(false);
  movieLoading->start();
  emit notifyTestActivated(widgetId, ui->lineURL->text());
}

void NewInputItemWidget::onUrlChanged() {
  ui->labelStatusIcon->clear();
  urlCheckStatus = UrlStatus::Unknown;
  emit notifyUrlContentChanged();
}
