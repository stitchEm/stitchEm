// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "libvideostitch/imageMergerFactory.hpp"
#include "libvideostitch/imageFlowFactory.hpp"
#include "libvideostitch/imageWarperFactory.hpp"
#include "libvideostitch/audioPipeDef.hpp"
#include "libvideostitch/object.hpp"
#include "libvideostitch/panoDef.hpp"
#include "libvideostitch/frame.hpp"
#include "libvideostitch/stereoRigDef.hpp"

/**
 * A project definition class.
 *
 * Contains everything that is saveable.
 * (Mostly the Panorama definition
 * and the blender configuration).
 *
 * The rest of the fields in ProjectDefinition
 * are transients.
 */
class VS_GUI_EXPORT MutableProjectDefinition : public VideoStitch::Ptv::Object {
 public:
  /**
   * Build/serialize from/to a Ptv::Value.
   * @param value Input value.
   * @return The parsed PostProdProjectDefinition, or NULL on error.
   */
  static MutableProjectDefinition* create(const VideoStitch::Ptv::Value& value);

  virtual VideoStitch::Ptv::Value* serialize() const;

  virtual ~MutableProjectDefinition();

  /**
   * Returns the buffer frame for the project.
   */
  int getBufferFrames() const { return bufferFrames; }

  /**
   * Returns the pano definition for the project.
   */
  VideoStitch::Core::PanoDefinition& getPanoDefinition() const { return *pano; }

  /**
   * Returns the audio pipe definition for the project.
   */
  VideoStitch::Core::AudioPipeDefinition& getAudioPipe() const { return *audioPipe; }

  /**
   * Returns the stereo rig definition for the project.
   */
  VideoStitch::Core::StereoRigDefinition* getStereoRigDefinition() const { return rig; }

  bool hasFileFormatChanged() const;
  void updateFileFormat();

  /**
   * @brief setAudioPipe writes the @audioPipe definition in the project.
   * @param audioPipe The audio pipeline definition. Shall not be NULL.
   */
  void setAudioPipe(VideoStitch::Core::AudioPipeDefinition* audioPipeDefinition);

  /**
   * @brief setPano writes the @pano definition in the project.
   * @param pano The panorama. Shall not be NULL.
   */
  void setPano(VideoStitch::Core::PanoDefinition* pano);

  /**
   * @brief setRig Writer a new StereoRigDefinition in the project
   * @param rig The rig definition
   */
  void setRig(VideoStitch::Core::StereoRigDefinition* rigDefinition);

  /**
   * Returns the merger factory for the project.
   */
  VideoStitch::Core::ImageMergerFactory& getImageMergerFactory() const { return *mergerFactory; }

  /**
   * Returns the warper factory for the project.
   */
  VideoStitch::Core::ImageWarperFactory& getImageWarperFactory() const { return *warperFactory; }

  /**
   * Returns the flow factory for the project.
   */
  VideoStitch::Core::ImageFlowFactory& getImageFlowFactory() const { return *flowFactory; }

  /**
   * @brief setMergerFactory writes the @mergerFactory definition in the project.
   * @param mergerFactory The merger factory. Shall not be NULL.
   */
  void setMergerFactory(VideoStitch::Core::ImageMergerFactory* mergerFactory);

  /**
   * @brief setWarperFactory writes the @warperFactory definition in the project.
   * @param warperFactory The warper factory. Shall not be NULL.
   */
  void setWarperFactory(VideoStitch::Core::ImageWarperFactory* warperFactory);

  /**
   * @brief setFlowFactory writes the @flowFactory definition in the project.
   * @param flowFactory The flow factory. Shall not be NULL.
   */
  void setFlowFactory(VideoStitch::Core::ImageFlowFactory* flowFactory);

 protected:
  MutableProjectDefinition(const MutableProjectDefinition&);

 private:
  // Pano definition.
  VideoStitch::Core::PanoDefinition* pano;
  // Audio pipe definition.
  VideoStitch::Core::AudioPipeDefinition* audioPipe;
  // Stereo rig definition (optional)
  VideoStitch::Core::StereoRigDefinition* rig;
  // Number of frames to buffer.
  int bufferFrames;
  // Merger factory.
  VideoStitch::Core::ImageMergerFactory* mergerFactory;
  VideoStitch::Core::ImageFlowFactory* flowFactory;
  VideoStitch::Core::ImageWarperFactory* warperFactory;
  QString libVersion;

  /**
   * Create from individual components. Ownership is transferred to the MutableProjectDefinition.
   * @param pano The panorama.
   * @param bufferFrames The number of frames to buffer.
   * @param mergerFactory The merger factory.
   * @param outputConfig The output writer configuration.
   */
  MutableProjectDefinition(VideoStitch::Core::PanoDefinition* pano, VideoStitch::Core::AudioPipeDefinition* audioPipe,
                           VideoStitch::Core::StereoRigDefinition* rig, int bufferFrames,
                           VideoStitch::Core::ImageMergerFactory* mergerFactory,
                           VideoStitch::Core::ImageWarperFactory* warperFactory,
                           VideoStitch::Core::ImageFlowFactory* flowFactory, QString libVersion);
};
