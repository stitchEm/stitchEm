// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "postprodprojectdefinition.hpp"

#include "libvideostitch-gui/videostitcher/stitchercontroller.hpp"

#include "libvideostitch-base/projection.hpp"

#include <QUrl>

class PostProdStitcherController : public StitcherController {
  Q_OBJECT
 public:
  explicit PostProdStitcherController(DeviceDefinition&);

  virtual ~PostProdStitcherController();
  frameid_t getLastStitchableFrame() const;
  std::vector<frameid_t> getLastFrames() const;
  /**
   * @brief The controller has an audio pipeline?
   * @return True if there is at least one valid audio reader.
   */
  bool hasAudioInput() const;
  bool allowsYPRModifications() const;
  VideoStitch::Status resetPano();

 signals:
  void reqWarnWrongInputSize(unsigned widthIs, unsigned heightIs, unsigned widthShouldBe, unsigned heightShouldbe);
  void reqCheckInputsDialog(QString& newFolder);
  void mergerNotSupportedByGpu(const std::string& merger, bool& fallBackToLinearBlending);
  // Broadcast the state of the orientation animation (allowed/disallowed)
  void projectOrientable(bool);
  // Update the OpenGL ouput to the correct projection
  // Without that, zooming/rotating/dragging wouldn't work as expected
  // since the geometric transformations involved are linked to the projection
  void reqChangeProjection(VideoStitch::Projection proj, double fov);
  void panoChanged(qint64 lastStitchableFrame, qint64 currentFrame);
  void statusBarUpdate(QString message);
  void calibrationApplied();
  void blendingMaskApplied();
  void exposureApplied();
  void advancedBlendingApplied();
  void resetMergerApplied();
  void resetAdvancedBlendingApplied();
  void snapshotsDone(QStringList);

  void reqUpdateCurve(VideoStitch::Core::Curve* curve, CurveGraphicsItem::Type type, int inputId = -1);
  void reqUpdateQuaternionCurve(VideoStitch::Core::QuaternionCurve* curve, CurveGraphicsItem::Type type,
                                int inputId = -1);
  void reqCurvesChanged(SignalCompressionCaps*,
                        std::vector<std::pair<VideoStitch::Core::Curve*, CurveGraphicsItem::Type> >,
                        std::vector<std::pair<VideoStitch::Core::QuaternionCurve*, CurveGraphicsItem::Type> >);
  void reqUpdatePhotometry();
  void notifyInputsOpened();

 public slots:
  void stitch(frameid_t, SignalCompressionCaps* comp = nullptr);
  void restitch(frameid_t frame, SignalCompressionCaps* comp = nullptr);
  void extract(frameid_t, SignalCompressionCaps* comp = nullptr);
  void reextract(frameid_t, SignalCompressionCaps* comp = nullptr);
  void stitchAndExtract(frameid_t, SignalCompressionCaps* comp = nullptr);

  void applyExposure(VideoStitch::Core::PanoDefinition*);
  void applyPhotometricCalibration(VideoStitch::Core::PanoDefinition*);
  void applyStabilization(VideoStitch::Core::PanoDefinition*);
  void applyCalibration(VideoStitch::Core::PanoDefinition*);
  void applyBlendingMask(VideoStitch::Core::PanoDefinition*);
  void applyAdvancedBlending(VideoStitch::Core::PanoDefinition*);
  void applySynchronization(SignalCompressionCaps*, VideoStitch::Core::PanoDefinition*);
  void applyTemplate(VideoStitch::Core::PanoDefinition*);
  void applyExternalCalibration(VideoStitch::Core::PanoDefinition*);

  void importCalibration(const QString&);
  void importTemplate(const QString&);
  void finishOrientation(int frame, VideoStitch::Quaternion<double> orientation,
                         VideoStitch::Core::QuaternionCurve* curve);
  void configureRig(const VideoStitch::Core::StereoRigDefinition::Orientation orientation,
                    const VideoStitch::Core::StereoRigDefinition::Geometry geometry, const double diameter,
                    const double ipd, const QVector<int> leftInputs, const QVector<int> rightInputs);

  void setProjection(VideoStitch::Projection proj, double fov);
  void ensureProjectionIsValid();
  void openInputs(QList<QUrl> urls, int customWidth, int customHeight);

  void reset() override;
  void onResetRig() override;
  void resetProject() override;
  void clearCalibration();
  void resetMerger(SignalCompressionCaps* comp = nullptr);
  void resetAdvancedBlending(SignalCompressionCaps* comp = nullptr);
  void snapshotSources(const QString& directory, bool forCalibration);
  void selectOrientation();
  void switchOutputAndRestitch(const QString&);
  void applyCrops(const QVector<Crop>& crops, const InputLensClass::LensType lensType) override;

 protected:
  /**
   * @brief Get a reference on the current project.
   * To publicly get the project, connect to the signal StitcherController::projectInitialized(ProjectDefinition*)
   */
  const ProjectDefinition* getProjectPtr() const override;

  /**
   * @brief Get a reference on the current project.
   * To publicly get the project, connect to the signal StitcherController::projectInitialized(ProjectDefinition*)
   */
  ProjectDefinition* getProjectPtr() override;
  void createProject() override;
  VideoStitch::Input::ReaderFactory* createReaderFactory() const override;

 private:
  void finishProjectOpening() override;
  bool checkProject() override;
  void informOnPano();
  bool seekOnSignal(frameid_t, SignalCompressionCaps*);

  PostProdProjectDefinition* project;  // owned
  VideoStitch::Projection currentProjection;
  VideoStitch::Projection previousProjection;
  QMap<VideoStitch::Projection, double> currentHFovs;
};
