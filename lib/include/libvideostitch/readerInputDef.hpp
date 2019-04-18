// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef READER_INPUT_DEF_HPP_
#define READER_INPUT_DEF_HPP_

#include "config.hpp"
#include "curves.hpp"
#include "status.hpp"

#include <string>

namespace VideoStitch {

namespace Ptv {
class Value;
}

namespace Core {

class VS_EXPORT ReaderInputDefinition : public Ptv::Object {
 public:
  virtual ~ReaderInputDefinition();

  /**
   * Clones an ReaderInputDefinition java-style.
   * @return A similar ReaderInputDefinition. Ownership is given to the caller.
   */
  virtual ReaderInputDefinition* clone() const;

  /**
   * Build from a Ptv::Value.
   * @param value Input value.
   * @return The parsed ReaderInputDefinition, or NULL on error.
   */
  static ReaderInputDefinition* create(const Ptv::Value& value, bool enforceMandatoryFields = true);

  virtual Ptv::Value* serialize() const;

  /**
   * Comparison operator.
   */
  virtual bool operator==(const ReaderInputDefinition& other) const;

  /**
   * Validate that the input makes sense.
   * @param os The sink for error messages.
   * @return false of failure.
   */
  virtual bool validate(std::ostream& os) const;

  /**
   * Returns the reader config of this input.
   */
  virtual const Ptv::Value& getReaderConfig() const;

  /**
   * Returns the reader config pointer of this input.
   */
  virtual const Ptv::Value* getReaderConfigPtr() const;

  /**
   * Sets the reader config of this input.
   * @param config Configuration to set. Must not be NULL. Ownership is taken.
   */
  virtual void setReaderConfig(Ptv::Value* config);

  /**
   * Returns the width of this input, including the optionally cropped part.
   */
  virtual int64_t getWidth() const;
  /**
   * Returns the height of this input, including the optionally cropped part.
   */
  virtual int64_t getHeight() const;
  /**
   * Sets the width of this input, including the optionally cropped part.
   */
  virtual void setWidth(int64_t);
  /**
   * Returns the height of this input, including the optionally cropped part.
   */
  virtual void setHeight(int64_t);

  /**
   * Returns the frame offset, in frames.
   */
  virtual frameid_t getFrameOffset() const;

  /**
   * Sets the frame offset.
   * @param fo Frame offset.
   */
  virtual void setFrameOffset(int fo);

  /**
   * Human-readable name for this input.
   */
  virtual std::string getDisplayName() const;

  /**
   * Shortcut to set the reader config to a single filename.
   * @param fn filename.
   */
  virtual void setFilename(const std::string& fileName);

  /**
   * Sets the input enabled state
   * @param state True to enable.
   * @note setIsVideoEnabled() is deprecated because it may be used to enable video on an audio-only input
   * @note setIsAudioEnabled() is deprecated because it may be used to enable audio on a video-only input
   * @note setIsEnabled() is not functional yet, somebody please create a real enabling/disabling mechanism
   */
  virtual void setIsEnabled(bool state);

  /**
   * Returns true if the input is enabled.
   */
  virtual bool getIsEnabled() const;

  /**
   * Returns true if the input's video is enabled.
   */
  virtual bool getIsVideoEnabled() const;

  /**
   * Sets is video enabled.
   */
  virtual void setIsVideoEnabled(bool b);

  /**
   * Returns true if the input's audio is enabled.
   */
  virtual bool getIsAudioEnabled() const;

  /**
   * Sets is audio enabled.
   */
  virtual void setIsAudioEnabled(bool b);

 protected:
  /**
   * Build with the mandatory fields. The others take default values.
   */
  ReaderInputDefinition();

  /**
   * Disabled, use clone()
   */
  ReaderInputDefinition(const ReaderInputDefinition&) = delete;

  /**
   * Disabled, use clone()
   */
  ReaderInputDefinition& operator=(const ReaderInputDefinition&) = delete;

  /**
   * Parse from the given ptv. Values not specified are not overridden.
   * @param diff Input diff.
   * @param enforceMandatoryFields If false, ignore missing mandatory values.
   */
  Status applyDiff(const Ptv::Value& diff, bool enforceMandatoryFields);

  /**
   * Parse an Input from a pto line.
   * @param line The input line
   * @param prevInputs The vector of previous ReaderInputDefinitions for back references.
   * @return The parsed ReaderInputDefinition.
   */
  static Potential<Core::ReaderInputDefinition> parseFromPtoLine(
      char* line, const std::vector<Core::ReaderInputDefinition*>& prevInputs);

  /**
   * Parse from a pts line.
   * @param line The input line
   * @param prevInputs The vector of previous ReaderInputDefinitions for back references.
   * @return The parsed ReaderInputDefinition.
   */
  static Potential<Core::ReaderInputDefinition> parseFromPtsLine(
      char* line, const std::vector<Core::ReaderInputDefinition*>& prevInputs);

  /**
   * Clones an ReaderInputDefinition java-style.
   * @param dstDef A similar ReaderInputDefinition. Ownership is given to the caller.
   */
  void cloneTo(ReaderInputDefinition* dstDef) const;

 private:
  friend class PanoDefinition;
  friend Potential<ReaderInputDefinition> parseFromPtoLine(char*, const std::vector<ReaderInputDefinition*>&);

  class Pimpl;
  Pimpl* const pimpl;
};
}  // namespace Core
}  // namespace VideoStitch

#endif
