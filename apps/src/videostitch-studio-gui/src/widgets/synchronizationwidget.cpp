// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "synchronizationwidget.hpp"
#include "ui_synchronizationwidget.h"

#include "commands/synchronizationoffsetschangedcommand.hpp"
#include "videostitcher/globalpostprodcontroller.hpp"

#include "libvideostitch-gui/caps/signalcompressioncaps.hpp"
#include "libvideostitch-gui/mainwindow/LibLogHelpers.hpp"
#include "libvideostitch-gui/mainwindow/ui_header/progressreporterwrapper.hpp"
#include "libvideostitch-gui/mainwindow/objectutil.hpp"
#include "libvideostitch-gui/mainwindow/msgboxhandlerhelper.hpp"
#include "libvideostitch-gui/mainwindow/timeconverter.hpp"
#include "libvideostitch-gui/widgets/autoselectspinbox.hpp"
#include "libvideostitch-base/file.hpp"
#include "libvideostitch/inputDef.hpp"
#include "libvideostitch/logging.hpp"

#include <QCheckBox>

SynchronizationWidget::SynchronizationWidget(QWidget* parent)
    : ComputationWidget(parent),
      ui(new Ui::SynchronizationWidget),
      panoDef(nullptr),
      compressor(SignalCompressionCaps::createOwned()) {
  ui->setupUi(this);
  ui->syncTabWidget->setCurrentWidget(ui->automaticSyncTab);
  ui->flashSyncButton->setVisible(VideoStitch::GPU::getFramework() != VideoStitch::Discovery::Framework::OpenCL);

  connect(ui->audioSyncButton, &QPushButton::clicked, this, &SynchronizationWidget::startAudio);
  connect(ui->motionSyncButton, &QPushButton::clicked, this, &SynchronizationWidget::startMotion);
  connect(ui->flashSyncButton, &QPushButton::clicked, this, &SynchronizationWidget::startFlash);
  connect(ui->resetAutoButton, &QPushButton::clicked, this, &SynchronizationWidget::resetOffsets);
  connect(ui->resetManualButton, &QPushButton::clicked, this, &SynchronizationWidget::resetOffsets);

  ui->offsetTab->horizontalHeader()->setSectionResizeMode(int(OffsetColumn::Name), QHeaderView::Stretch);
  ui->offsetTab->horizontalHeader()->setSectionResizeMode(int(OffsetColumn::OffsetSpinBox),
                                                          QHeaderView::ResizeToContents);
  ui->offsetTab->horizontalHeader()->setSectionResizeMode(int(OffsetColumn::Link), QHeaderView::ResizeToContents);
  ui->offsetTab->horizontalHeader()->setDefaultAlignment(Qt::AlignLeft | Qt::AlignVCenter);
  installEventFilter(this);
}

SynchronizationWidget::~SynchronizationWidget() { delete panoDef; }

// ----------------------- Manual offsets edition ----------------------------

void SynchronizationWidget::offsetValueChanged() {
  // Retrieve the sender before to disconnect the spin boxes
  AutoSelectSpinBox* senderSpinbox = qobject_cast<AutoSelectSpinBox*>(sender());

  connectAllSpinBoxes(false);  // It will be reactivated by the undo command bellow

  // Compute the initial diff and cancel the change
  int currentSenderValue = currentValues.at(spinBoxes.indexOf(senderSpinbox));
  int initialDiff = senderSpinbox->value() - currentSenderValue;
  senderSpinbox->setValue(currentSenderValue);

  // Compute the spin boxes to update
  QList<AutoSelectSpinBox*>
      activeSpinBoxes;  // Active spin boxes are the current spin box and eventually the other linked spin boxes
  QList<AutoSelectSpinBox*> linkedSpinBoxes = getLinkedSpinBoxes();
  if (linkedSpinBoxes.contains(senderSpinbox)) {
    activeSpinBoxes = linkedSpinBoxes;
  } else {
    activeSpinBoxes.append(senderSpinbox);
  }

  // Compute the final diffs
  int diffForActiveSpinBoxes = 0;
  int diffForOtherSpinBoxes = 0;
  if (initialDiff >= 0) {
    diffForActiveSpinBoxes = initialDiff;
  } else {  // Negative diff
    diffForActiveSpinBoxes = initialDiff;

    for (auto activeSpinBox : activeSpinBoxes) {
      int maxDiffForThisSpinBox = initialDiff;
      if (activeSpinBox->value() < qAbs(initialDiff)) {
        maxDiffForThisSpinBox = -activeSpinBox->value();
      }
      diffForActiveSpinBoxes = qMax(diffForActiveSpinBoxes, maxDiffForThisSpinBox);
    }

    diffForOtherSpinBoxes = diffForActiveSpinBoxes - initialDiff;
  }

  // Finally, build the undo command
  QVector<int> newValues;
  foreach (auto spinBox, spinBoxes) {
    if (activeSpinBoxes.contains(spinBox)) {
      newValues.append(spinBox->value() + diffForActiveSpinBoxes);
    } else {
      newValues.append(spinBox->value() + diffForOtherSpinBoxes);
    }
  }
  QVector<bool> checkedStatuses;
  foreach (const QCheckBox* checkBox, checkBoxes) { checkedStatuses.append(checkBox->isChecked()); }
  SynchronizationOffsetsChangedCommand* command = new SynchronizationOffsetsChangedCommand(
      newValues, currentValues, checkedStatuses, checkedStatuses, spinBoxes.indexOf(senderSpinbox), this);
  qApp->findChild<QUndoStack*>()->push(command);
}

void SynchronizationWidget::buildOffsetWidgets(bool preserveProjectParameters) {
  // Keep the old check states in case we have the same project
  QVector<Qt::CheckState> oldStates;
  if (preserveProjectParameters) {
    foreach (const QCheckBox* box, checkBoxes) { oldStates.append(box->checkState()); }
  }

  spinBoxes.clear();
  checkBoxes.clear();
  currentValues.clear();
  ui->offsetTab->clearContents();
  ui->offsetTab->setRowCount(0);

  if (!project || !project->isInit()) {
    return;
  }

  std::vector<frameid_t> lastFrames = GlobalPostProdController::getInstance().getController()->getLastFrames();
  auto nbInputs = project->getNumInputs();
  Q_ASSERT(lastFrames.size() == (size_t)nbInputs);

  ui->offsetTab->setRowCount(int(nbInputs));

  for (int index = 0; index < nbInputs; ++index) {
    const VideoStitch::Core::InputDefinition& inputDef = project->getPanoConst()->getInput(index);

    QString boxName =
        QString(" %0 - %1").arg(index).arg(File::strippedName(QString::fromStdString(inputDef.getDisplayName())));
    QLabel* label = new QLabel(boxName, ui->offsetTab);
    ui->offsetTab->setCellWidget(index, int(OffsetColumn::Name), label);

    auto offset = inputDef.getFrameOffset();
    AutoSelectSpinBox* newSpinBox = new AutoSelectSpinBox(ui->offsetTab);
    newSpinBox->setFocusPolicy(Qt::ClickFocus);
    newSpinBox->setAutoSelectOnFocus(true);
    newSpinBox->setFrame(false);
    newSpinBox->setSuffix(tr(" frames"));
    newSpinBox->setMinimum(
        -1);  // Because the user can decrease a spinbox at 0; it will increase the other spinbox values
    newSpinBox->setMaximum(int(lastFrames.at(index)));
    newSpinBox->setValue(offset);
    newSpinBox->installEventFilter(this);
    spinBoxes.append(newSpinBox);
    currentValues.append(offset);
    ui->offsetTab->setCellWidget(index, int(OffsetColumn::OffsetSpinBox), newSpinBox);

    QWidget* checkBoxContainer = new QWidget(ui->offsetTab);
    QHBoxLayout* layout = new QHBoxLayout(ui->offsetTab);
    layout->setAlignment(Qt::AlignHCenter);
    layout->setContentsMargins(0, 0, 0, 0);
    QCheckBox* newCheckBox = new QCheckBox(checkBoxContainer);
    newCheckBox->setObjectName("link-check-box");
    newCheckBox->setFocusPolicy(Qt::TabFocus);
    newCheckBox->setCheckState(oldStates.value(int(index), Qt::Unchecked));
    newCheckBox->installEventFilter(this);
    checkBoxes.append(newCheckBox);
    layout->addWidget(newCheckBox);
    checkBoxContainer->setLayout(layout);
    ui->offsetTab->setCellWidget(index, int(OffsetColumn::Link), checkBoxContainer);
  }

  if (nbInputs > 0) {
    QStringList offsetTableHeaders;
    //: please translate these strings, used as table headers
    offsetTableHeaders << tr("Input") << tr("Offset") << tr("Link");
    ui->offsetTab->setHorizontalHeaderLabels(offsetTableHeaders);
  }

  connectAllSpinBoxes(true);
}

bool SynchronizationWidget::eventFilter(QObject* obj, QEvent* event) {
  if (event->type() == QEvent::KeyPress) {
    if (static_cast<QKeyEvent*>(event)->key() == Qt::Key_Space) {
      emit reqTogglePlay();
    }
  }
  return QWidget::eventFilter(obj, event);
}

QList<AutoSelectSpinBox*> SynchronizationWidget::getLinkedSpinBoxes() const {
  QList<AutoSelectSpinBox*> linkedSpinBoxes;
  int nb = spinBoxes.count();
  for (int index = 0; index < nb; ++index) {
    if (checkBoxes.at(index)->isChecked()) {
      linkedSpinBoxes.append(spinBoxes.at(index));
    }
  }
  return linkedSpinBoxes;
}

// --------------------------- Asynchronous algorithm implementation ---------------------------

void SynchronizationWidget::startAudio() {
  startComputationOf(std::bind(&SynchronizationWidget::computation, this, "sound_offset_align", createConfig()));
}

void SynchronizationWidget::startMotion() {
  startComputationOf(std::bind(&SynchronizationWidget::computation, this, "motion_synchronization", createConfig()));
}

void SynchronizationWidget::startFlash() {
  startComputationOf(std::bind(&SynchronizationWidget::computation, this, "flash_synchronization", createConfig()));
}

VideoStitch::Status* SynchronizationWidget::computation(
    const std::string& algoStr, std::shared_ptr<VideoStitch::Ptv::Value> synchronizationConfig) {
  VideoStitch::Potential<VideoStitch::Util::Algorithm> fStatus =
      VideoStitch::Util::Algorithm::create(algoStr, synchronizationConfig.get());
  if (!fStatus.ok()) {
    panoValue.reset();
    return new VideoStitch::Status(
        VideoStitch::Origin::SynchronizationAlgorithm, VideoStitch::ErrType::UnsupportedAction,
        tr("Could not initialize the synchronization algorithm").toStdString(), fStatus.status());
  }
  algo.reset(fStatus.release());

  delete panoDef;
  panoDef = project->getPanoConst().get()->clone();
  VideoStitch::Potential<VideoStitch::Ptv::Value> ret = algo->apply(panoDef, getReporter());

  if (ret.status().ok()) {
    if (ret.object() && ret->has("lowConfidence") && ret->get("lowConfidence")->asBool()) {
      VideoStitch::Logger::get(VideoStitch::Logger::Info) << "Low confidence Motion Synchro" << std::endl;
    }
  }

  panoValue.reset(ret.release());
  return new VideoStitch::Status(ret.status());
}

QString SynchronizationWidget::getAlgorithmName() const { return tr("Synchronization"); }

void SynchronizationWidget::manageComputationResult(bool hasBeenCancelled, VideoStitch::Status* status) {
  if (status->ok()) {
    connectAllSpinBoxes(false);  // It will be reactivated by the undo command bellow
    QVector<int> newValues, oldValues;
    for (readerid_t i = 0; i < panoDef->numInputs(); ++i) {
      newValues.append(panoDef->getInput(i).getFrameOffset());
      oldValues.append(spinBoxes.at(int(i))->value());
    }

    QVector<bool> oldChecked, newChecked;
    foreach (const QCheckBox* checkBox, checkBoxes) {
      oldChecked.append(checkBox->isChecked());
      newChecked.append(checkBox->isChecked());
    }
    SynchronizationOffsetsChangedCommand* command =
        new SynchronizationOffsetsChangedCommand(newValues, oldValues, newChecked, oldChecked, -1, this);
    qApp->findChild<QUndoStack*>()->push(command);
  } else {
    if (!hasBeenCancelled) {
      MsgBoxHandlerHelper::genericErrorMessage({VideoStitch::Origin::SynchronizationAlgorithm,
                                                VideoStitch::ErrType::RuntimeError, "Synchronization failed", *status});
    }
  }
  panoValue.reset();
  delete status;
}

void SynchronizationWidget::resetOffsets() {
  connectAllSpinBoxes(false);  // It will be reactivated by the undo command bellow

  QVector<bool> oldChecked;
  foreach (const QCheckBox* checkBox, checkBoxes) { oldChecked.append(checkBox->isChecked()); }
  const QVector<int> newValues = QVector<int>(spinBoxes.count(), 0);
  const QVector<bool> newChecked = QVector<bool>(spinBoxes.count(), false);

  SynchronizationOffsetsChangedCommand* command =
      new SynchronizationOffsetsChangedCommand(newValues, currentValues, newChecked, oldChecked, -1, this);
  qApp->findChild<QUndoStack*>()->push(command);
}

void SynchronizationWidget::changeAllValues(QVector<int> newOffsetValues, QVector<bool> newChecked) {
  connectAllSpinBoxes(false);
  int minInputs = qMin(spinBoxes.count(), newOffsetValues.count());
  for (int i = 0; i < minInputs; ++i) {
    checkBoxes.at(i)->setChecked(newChecked.at(i));
    spinBoxes.at(i)->setValue(newOffsetValues.at(i));
  }
  currentValues = newOffsetValues;
  connectAllSpinBoxes(true);
  submit();
}

void SynchronizationWidget::onProjectOpened(ProjectDefinition* newProject) {
  bool sameProject = newProject == project;
  ComputationWidget::onProjectOpened(newProject);
  buildOffsetWidgets(sameProject);
}

void SynchronizationWidget::clearProject() {
  ComputationWidget::clearProject();
  spinBoxes.clear();
  checkBoxes.clear();
  currentValues.clear();
  ui->offsetTab->clearContents();
  ui->offsetTab->setRowCount(0);
}

void SynchronizationWidget::updateSequence(const QString start, const QString stop) {
  ui->timeSequence->sequenceUpdated(start, stop);
}

// ------------------------------ Helpers --------------------------------------

void SynchronizationWidget::connectAllSpinBoxes(bool connect) {
  foreach (AutoSelectSpinBox* spin, spinBoxes) {
    VideoStitch::Helper::toggleConnect(connect, spin, SIGNAL(valueChanged(int)), this, SLOT(offsetValueChanged()),
                                       Qt::UniqueConnection);
  }
}

std::shared_ptr<VideoStitch::Ptv::Value> SynchronizationWidget::createConfig() const {
  std::shared_ptr<VideoStitch::Ptv::Value> syncConfig(VideoStitch::Ptv::Value::emptyObject());
  syncConfig->get("first_frame")->asInt() = project->getFirstFrame();
  syncConfig->get("last_frame")->asInt() = project->getLastFrame();
  return syncConfig;
}

// send the signal to update the readers
// according to the current checkboxes' values
void SynchronizationWidget::submit() {
  VideoStitch::Core::PanoDefinition* panorama = project->getPanoConst().get()->clone();
  for (readerid_t i = 0; i < panorama->numInputs(); ++i) {
    panorama->getInput(i).setFrameOffset(spinBoxes.at(int(i))->value());
  }
  emit reqApplySynchronization(compressor->add(), panorama);
}
