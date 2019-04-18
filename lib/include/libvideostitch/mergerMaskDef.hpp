// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "algorithm.hpp"
#include "parse.hpp"
#include "object.hpp"
#include <map>
#include <memory>

namespace VideoStitch {

namespace Ptv {
class Value;
}

namespace Core {

/**
 * @brief The merger mask definition
 */
class VS_EXPORT MergerMaskDefinition : public Ptv::Object {
 public:
  ~MergerMaskDefinition();
  /**
   * Represents an input mask.
   */
  struct InputIndexPixelData {
   public:
    InputIndexPixelData() : width(0), height(0) {}
    ~InputIndexPixelData();
    /**
     * Returns the mask's width.
     */
    int64_t getWidth() const { return width; }

    /**
     * Returns the mask's width.
     */
    int64_t getHeight() const { return height; }

    /**
     * Reallocate a buffer.
     * @param newWidth Width of the new buffer.
     * @param newHeight Height of the new buffer.
     * @param newData Pointer to the new data
     * @return true on success.
     */
    Status realloc(const frameid_t frameId, const int64_t newWidth, const int64_t newHeight,
                   const std::map<videoreaderid_t, std::string>& newData);

    /**
     * Clear all data.
     */
    void clear();

    /**
     * Get mask of an input.
     * @param frameId Index of frame
     * @param index Index of input.
     * @return Pointer to the mask
     */
    const std::string& getData(const frameid_t frameId, const size_t index);

    const std::map<frameid_t, std::map<videoreaderid_t, std::string>>& getData();

    /**
     * Get mask of an input.
     * @param frameId Index of frame
     * @return Pointer to the mask
     */
    const std::map<videoreaderid_t, std::string>& getData(const frameid_t frameId);

    /**
     * @return Number of inputs
     */
    size_t getDataCount(const frameid_t frameId);

    /**
     * @return The list of input indices
     */
    std::vector<videoreaderid_t> getInputIndices() const;

    /*
     * @return the full map at the current frame
     * depends on the number of precomputed frames, there might be 0 (no frame was computed), 1, or 2 outputs
     */
    std::vector<std::pair<frameid_t, std::map<videoreaderid_t, std::string>>> getFullData(const frameid_t frameId);

    /*
     * return Number of frame
     */
    size_t getFrameCount() const;

    /*
     * return FrameId of a certain index
     */
    std::vector<frameid_t> getFrameIds() const;

    void removeFrameIds(const std::vector<frameid_t>& frameIds);

    std::pair<frameid_t, frameid_t> getBoundedFrameIds(const frameid_t frameId) const;

   private:
    friend class MergerMaskDefinition;
    int64_t width;
    int64_t height;
    std::map<frameid_t, std::map<videoreaderid_t, std::string>> datas;
  };

  /**
   * Clones an MergerMaskDefinition java-style.
   * @return A similar MergerMaskDefinition. Ownership is given to the caller.
   */
  virtual MergerMaskDefinition* clone() const;

  /**
   * Build from a Ptv::Value.
   * @param value Input value.
   * @return The parsed MergerMaskDefinition.
   */
  static MergerMaskDefinition* create(const Ptv::Value& value);

  virtual Ptv::Value* serialize() const;

  /**
   * Check whether the mask is enabled
   */
  virtual bool getEnabled() const;

  /**
   * Check whether the mask interpolation is enabled
   */
  virtual bool getInterpolationEnabled() const;

  /**
   * Returns the width of the mask.
   */
  virtual int64_t getWidth() const;

  /**
   * Returns the height of the mask.
   */
  virtual int64_t getHeight() const;

  /**
   * Returns the mask orders.
   */
  virtual std::vector<size_t> getMasksOrder() const;

  /**
   * Returns the list of frames.
   */
  virtual std::vector<frameid_t> getFrameIds() const;

  /**
   * Sets the "enable" of this mask.
   */
  virtual void setEnabled(bool);

  /**
   * Sets the "interpolation" of this mask.
   */
  virtual void setInterpolationEnabled(bool);

  /**
   * Sets the width of this mask.
   */
  virtual void setWidth(int64_t);

  /**
   * Sets the height of this mask.
   */
  virtual void setHeight(int64_t);

  /**
   * Sets the mask orders of this mask.
   */
  virtual void setMasksOrder(std::vector<size_t>);

  /**
   * Returns the list of frames.
   */
  virtual void removeFrameIds(const std::vector<frameid_t>&);

  /**
   * Returns the raw pixel data for the mask of this input.
   * The data is invalidated by any call to a non-const method to the InputDefinition.
   * Note that the size is not necessarily that of the InputDefinition.
   * @return InputIndexPixelData() if there is no data, or if the data is invalid.
   */
  virtual const InputIndexPixelData& getInputIndexPixelData() const;

  /**
   * A helper that returns the raw pixel data for the mask of this input; or NULL if there is no mask.
   */
  virtual std::vector<std::pair<frameid_t, std::map<videoreaderid_t, std::string>>> getInputIndexPixelDataIfValid(
      const frameid_t frameId) const;

  /**
   * Returns true if there is no mask or if the mask data is valid (i.e. can be decoded and is of the correct size).
   */
  virtual bool validateInputIndexPixelData() const;

  /**
   * Set the raw pixel data for the mask of this input. *encodedMasks* is the encoded mask stored in the input space.
   * The data is invalidated by any call to a non-const method to the MergerMaskDefinition.
   * Note that the size is not necessarily that of the InputDefinition.
   * @return true if data is valid, false otherwise.
   */
  virtual Status setInputIndexPixelData(const std::map<videoreaderid_t, std::string>& encodedMasks,
                                        const uint64_t width, const uint64_t height, const frameid_t frameId);

  /**
   * Returns the scale factor of the input map resolution.
   * The blending mask is computed in the output space but stored in the input space.
   * One pixel from the input image might cover several pixels in the output panorama.
   * A factor > 1 allows sub-pixel accuracy of the input map to better preserve the original mask (in the output space).
   */
  virtual int getInputScaleFactor() const;

  /**
   * @brief Set the scale factor value
   */
  virtual void setInputScaleFactor(int);

 protected:
  /**
   * Build with the mandatory fields. The others take default values.
   */
  MergerMaskDefinition();

  /**
   * Disabled, use clone()
   */
  MergerMaskDefinition(const MergerMaskDefinition&) = delete;
  /**
   * Disabled, use clone()
   */
  MergerMaskDefinition& operator=(const MergerMaskDefinition&) = delete;

  /**
   * Parse from the given ptv. Values not specified are not overridden.
   * @param value Input value.
   * @param enforceMandatoryFields If false, ignore missing mandatory values.
   */
  Status applyDiff(const Ptv::Value& value, bool enforceMandatoryFields);

 private:
  class Pimpl;
  Pimpl* const pimpl;
};

}  // namespace Core
}  // namespace VideoStitch
