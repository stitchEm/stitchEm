// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef NEWINPUTITEMWIDGET_HPP
#define NEWINPUTITEMWIDGET_HPP

#include <QWidget>

namespace Ui {
class NewInputItemWidget;
}

class NewInputItemWidget : public QWidget {
  enum UrlStatus { Unknown, Verified, Failed };

  Q_OBJECT
 public:
  explicit NewInputItemWidget(const QString url, const int id, QWidget* const parent = nullptr);
  ~NewInputItemWidget();
  bool hasValidUrl() const;
  QByteArray getUrl() const;
  int getId() const;

 public slots:
  void setUrl(const QString url);
  void onTestFinished(const bool success);
  void onTestClicked();

 signals:
  void notifyUrlValidityChanged();
  void notifyUrlContentChanged();
  void notifyTestActivated(const int id, const QString name);

 private slots:
  void onUrlChanged();

 private:
  Ui::NewInputItemWidget* ui;
  const int widgetId;
  QMovie* movieLoading;
  UrlStatus urlCheckStatus = UrlStatus::Unknown;
};

#endif  // NEWINPUTITEMWIDGET_HPP
