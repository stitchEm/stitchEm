// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "libvideostitch-gui/caps/guistatecaps.hpp"
#include "libvideostitch-gui/widgets/stylablewidget.hpp"

#include "libvideostitch/algorithm.hpp"
#include "libvideostitch/status.hpp"

#include <QFutureWatcher>
#include <QPointer>

#include <functional>

class ProjectDefinition;
class PostProdProjectDefinition;
class ModalProgressDialog;
class ProgressReporterWrapper;

/**
 * @brief This class is a base class dedicated to the usage of VideoStitch::Util::Algorithm.
 *        When you need to use a VideoStitch::Util::Algorithm, display a progress bar for this algorithm, and lock your
 * application this is the class you need to use. To do so, fill the widget with the parameters you need for your
 * algorithm, and call the method startComputationOf() with your computation function as a parameter.
 *
 *        You should call your initialization instructions before startComputationOf(), and your cleanup instructions in
 * manageComputationResult(). These methods will handle themselves what happened to the progress reporter (if it has
 * been closed to cancel the algorithm). These methods are called in the main thread (thread 0).
 */
class ComputationWidget : public QWidget, public GUIStateCaps {
  Q_OBJECT
  Q_MAKE_STYLABLE

 public:
  explicit ComputationWidget(QWidget* parent = nullptr);
  virtual ~ComputationWidget();

 public slots:
  virtual void onProjectOpened(ProjectDefinition* p);
  virtual void clearProject();

  /**
   * @brief Changes the widget's stats to the given state.
   * @param s State you want to switch to.
   */

  virtual void changeState(GUIStateCaps::State s) override;

  /**
   * @brief Updates the time sequence
   * @param start Starting time
   * @param stop Ending time
   */
  virtual void updateSequence(const QString start, const QString stop) = 0;

 signals:
  /**
   * @brief Requests the State Manager to initiate a specific state transition.
   * @param s is the requested state.
   */
  void reqChangeState(GUIStateCaps::State s) override;
  void reqPause();

 protected:
  ProgressReporterWrapper* getReporter() const;
  virtual QString getAlgorithmName() const = 0;
  /**
   * @brief Manages the GUI part (eg progress dialog) and runs the computation function in an asynchronous thread.
   */
  void startComputationOf(std::function<VideoStitch::Status*()> function);
  /**
   * @brief Method called when the algorithm ends. The cleanup instructions should be there
   * @param hasBeenCancelled tells if the user cancelled the algorithm.
   * @param status You'll have to delete it yourself.
   */
  virtual void manageComputationResult(bool hasBeenCancelled, VideoStitch::Status* status) = 0;

 protected:
  QScopedPointer<VideoStitch::Util::Algorithm> algo;
  QPointer<PostProdProjectDefinition> project;

 private slots:
  void finishComputation();

 private:
  QScopedPointer<ModalProgressDialog> algorithmProgressReporterDialog;
  QFutureWatcher<VideoStitch::Status*> asyncTaskWatcher;
};
