// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "computationwidget.hpp"
#include "itoolwidget.hpp"
#include <memory>

class AutoSelectSpinBox;
class QCheckBox;
class SignalCompressionCaps;

namespace Ui {
class SynchronizationWidget;
}

class SynchronizationWidget : public ComputationWidget, public IToolWidget {
  Q_OBJECT

 public:
  explicit SynchronizationWidget(QWidget* parent = nullptr);
  ~SynchronizationWidget();
  void changeAllValues(QVector<int> newOffsetValues, QVector<bool> newChecked);

 signals:
  void reqApplySynchronization(SignalCompressionCaps*, VideoStitch::Core::PanoDefinition*);
  void reqTogglePlay();

 public slots:
  virtual void onProjectOpened(ProjectDefinition* newProject) override;
  virtual void clearProject() override;
  virtual void updateSequence(const QString start, const QString stop) override;

  void startAudio();
  void startMotion();
  void startFlash();

 protected:
  virtual QString getAlgorithmName() const override;
  virtual void manageComputationResult(bool hasBeenCancelled, VideoStitch::Status* status) override;
  virtual void reset() override {}

 private slots:
  /**
   * @brief resetOffsets Resets the offsets to 0
   */
  void resetOffsets();
  /**
   * @brief offsetValueChanged Signal emitted when a spinbox value has changed
   */
  void offsetValueChanged();

 private:
  void buildOffsetWidgets(bool preserveProjectParameters);
  virtual bool eventFilter(QObject* obj, QEvent* event) override;
  QList<AutoSelectSpinBox*> getLinkedSpinBoxes() const;
  void submit();
  void connectAllSpinBoxes(bool connect);
  std::shared_ptr<VideoStitch::Ptv::Value> createConfig() const;
  VideoStitch::Status* computation(const std::string& algoString,
                                   std::shared_ptr<VideoStitch::Ptv::Value> synchronizationConfig);

  enum class OffsetColumn : int { Name = 0, OffsetSpinBox, Link };

  QScopedPointer<Ui::SynchronizationWidget> ui;
  QList<AutoSelectSpinBox*> spinBoxes;  // Owned by the QTableWidget
  QList<QCheckBox*> checkBoxes;         // Owned by the QTableWidget
  QVector<int> currentValues;           // This is usefull only for manual edition, to know the diff
  std::unique_ptr<VideoStitch::Ptv::Value> panoValue;
  VideoStitch::Core::PanoDefinition* panoDef;
  std::shared_ptr<SignalCompressionCaps> compressor;
};
