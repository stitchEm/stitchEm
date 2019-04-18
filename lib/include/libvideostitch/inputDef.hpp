// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef INPUT_DEF_HPP_
#define INPUT_DEF_HPP_

#include "config.hpp"
#include "curves.hpp"
#include "status.hpp"
#include "geometryDef.hpp"
#include "readerInputDef.hpp"

#include <array>

namespace VideoStitch {

namespace Ptv {
class Value;
}

namespace Core {
class GeometryDefinition;

/**
 * @brief An input stream representation class.
 */
class VS_EXPORT InputDefinition : public ReaderInputDefinition {
 public:
  /**
   * An identifier for a group of inputs
   * (inputs with a common time referential)
   */
  typedef int group_t;

  /**
   * Input format (projection).
   */
  enum class Format {
    Rectilinear = 0,
    CircularFisheye = 2,
    FullFrameFisheye = 3,
    Equirectangular = 4,
    CircularFisheye_Opt = 5,
    FullFrameFisheye_Opt = 6
  };

  /**
   * Legacy/Optimized lens model category
   */
  enum class LensModelCategory {
    Legacy,    // Rectilinear, CircularFisheye, FullFrameFisheye, Equirectangular
    Optimized  // CircularFisheye_Opt, FullFrameFisheye_Opt
  };

  /**
   * Photometric response type of the camera.
   */
  enum class PhotoResponse { LinearResponse, GammaResponse, EmorResponse, InvEmorResponse, CurveResponse };

  virtual ~InputDefinition();

  /**
   * Clones an InputDefinition java-style.
   * @return A similar InputDefinition. Ownership is given to the caller.
   */
  virtual InputDefinition* clone() const;

  /**
   * Build from a Ptv::Value.
   * @param value Input value.
   * @return The parsed InputDefinition, or nullptr on error.
   */
  static InputDefinition* create(const Ptv::Value& value, bool enforceMandatoryFields = true);

  virtual Ptv::Value* serialize() const;

  /**
   * Comparison operator.
   */
  virtual bool operator==(const InputDefinition& other) const;

  /**
   * Validate that the input makes sense.
   * @param os The sink for error messages.
   * @return false of failure.
   */
  virtual bool validate(std::ostream& os) const;

  /**
   * Convert a panotools int to an InputDefinition::Format.
   * @param ptFmt a PanoTools integer representing the format.
   * @param fmt VideoStitch output format.
   * @return false of the input format is not supported.
   */
  static bool fromPTFormat(const char* ptFmt, Format* fmt);

  /**
   * Converts a projection name to a InputDefinition::Format.
   * @param fmt projection name.
   * @param fmtOut On output, contains the format if successful.
   * @return false is the projection is unknown.
   */
  static bool getFormatFromName(const std::string& fmt, InputDefinition::Format& fmtOut);

  /**
   * Returns the data for the mask of this input.
   */
  virtual const std::string& getMaskData() const;

  /**
   * Returns true if masked pixels are taken into account when computing stitching seam geometry.
   */
  virtual bool deletesMaskedPixels() const;

  /**
   * Represents an input mask.
   */
  struct MaskPixelData {
   public:
    MaskPixelData() : width(0), height(0), data(NULL) {}
    ~MaskPixelData();
    /**
     * Returns the mask's width.
     */
    virtual int64_t getWidth() const { return width; }
    /**
     * Returns the mask's width.
     */
    virtual int64_t getHeight() const { return height; }
    /**
     * Reallocate a buffer.
     * @param newWidth Width of the new buffer.
     * @param newHeight Height of the new buffer.
     * @return true on success.
     */
    virtual bool realloc(int64_t newWidth, int64_t newHeight);
    /**
     * Returns the (const) data buffer.
     */
    virtual const unsigned char* getData() const { return data; }

   private:
    friend class InputDefinition;
    int64_t width;
    int64_t height;
    unsigned char* data;  // Owned.
  };

  /**
   * Returns the raw pixel data for the mask of this input.
   * The data is invalidated by any call to a non-const method to the InputDefinition.
   * Note that the size is not necessarily that of the InputDefinition.
   * @return MaskPixelData() if there is no data, or if the data is invalid.
   */
  virtual const MaskPixelData& getMaskPixelData() const;

  /**
   * A helper that returns the raw pixel data for the mask of this input; or NULL if there is no mask.
   * Functionally equivalent to:
   *   return validateMask() ? maskPixelDataCache.getData() : NULL;
   */
  virtual const unsigned char* getMaskPixelDataIfValid() const;

  /**
   * Returns true if there is no mask or if the mask data is valid (i.e. can be decoded and is of the correct size).
   */
  virtual bool validateMask() const;

  /**
   * Returns the group of this input.
   * Groups are readers with a common time referential.
   */
  virtual group_t getGroup() const;
  /**
   * Sets the group of this input.
   * Groups are readers with a common time referential.
   */
  virtual void setGroup(group_t);

  /**
   * Returns the width of this input, NOT including the optionally cropped part.
   */
  virtual int64_t getCroppedWidth() const;
  /**
   * Returns the height of this input, NOT including the optionally cropped part.
   */
  virtual int64_t getCroppedHeight() const;

  /**
   * Get the place the distortion take place
   * @return is the distortion in meter space
   */
  virtual bool getUseMeterDistortion() const;

  /**
   * Returns the input projection name.
   */
  static const char* getFormatName(Format fmt);

  /**
   * Returns the input projection.
   */
  virtual Format getFormat() const;

  /**
   * Sets the input projection.
   */
  virtual void setFormat(Format);

  /**
   * Returns the lens category
   */
  virtual LensModelCategory getLensModelCategory() const;

  /**
   * Returns true if parameters are relative to the cropped area.
   */
  virtual bool hasCroppedArea() const;

  /**
   * Returns the cost of current input
   */
  virtual double getSynchroCost() const;

  /**
   * Returns the input stack order.
   */
  virtual int getStack() const;

  /**
   * Sets the stack order. If two inputs have the same stack order, the first one is merged first.
   * @param value stack value to set.
   */
  virtual void setStack(int value);

  /**
   * Sets the mask data of this input (2-bit png).
   * @param data "file:/path/to/file", data or NULL for no mask.
   * @param len Length of @a data.
   */
  virtual void setMaskData(const std::string& maskData);

  /**
   * Sets whether the masked pixels are taken into account when computing stitching seam geometry.
   * @param value True to delete masked pixels (i.e. do as if they were not masked.).
   */
  virtual void setDeletesMaskedPixels(bool value);

  /**
   * Sets the mask data of this input from raw pixel data by compressing the data.
   * @param buffer Pixel data, of size @a maskWidth x @a maskHeight
   * @param maskWidth Width of @a buffer.
   * @param maskHeight Height of @a buffer.
   * @return false on failure.
   */
  virtual bool setMaskPixelData(const char* buffer, uint64_t maskWidth, uint64_t maskHeight);

  /**
   * Set the synchronization cost for current input
   * @param cost Cost of synchronization for current input
   */
  virtual void setSynchroCost(double cost);

  /**
   * green/red/blue compensation factor. (Linear scale, 1.0 means no correction)
   * For hugin, this is always 1. For PTGui, this is computed from red and blue so that the sum of the 3 components is 0
   * in log scale. We convert it on import.
   */
  DECLARE_CURVE(RedCB, double)
  DECLARE_CURVE(GreenCB, double)
  DECLARE_CURVE(BlueCB, double)

  /**
   * Exposure value, log scale.
   */
  DECLARE_CURVE(ExposureValue, double)

  /**
   * Geometries
   * @note Reseter method resetGeometries() not defined by the macro, but explicitly below with an argument
   */
  DECLARE_CURVE_WITHOUT_RESETER(Geometries, GeometryDefinition)

  /**
   * Resets the geometries to the given HFOV
   * @param HFOV Horizontal field of view
   */
  virtual void resetGeometries(const double HFOV);

  /**
   * Get the preprocessors config, or NULL.
   */
  virtual const Ptv::Value* getPreprocessors() const;

  // --------------------- Lens --------------------

  // The distortion can evolve in time for a better image
  // alignement, look in GeometryDef.

  /**
   * Set the place the distortion take place
   * @param meter is the distortion in meter space
   */
  virtual void setUseMeterDistortion(bool meter);

  // -------------------- Camera response function --------------

  /**
   * Returns the photometric response type.
   */
  virtual PhotoResponse getPhotoResponse() const;

  /**
   * Returns the coefficient of the 2nd eigenvector in the EMoR basis.
   * @note Only makes sense for EmorResponse photometric responses.
   */
  virtual double getEmorA() const;
  /**
   * Returns the coefficient of the 3rd eigenvector in the EMoR basis.
   * @note Only makes sense for EmorResponse photometric responses.
   */
  virtual double getEmorB() const;
  /**
   * Returns the coefficient of the 4th eigenvector in the EMoR basis.
   * @note Only makes sense for EmorResponse photometric reponses.
   */
  virtual double getEmorC() const;
  /**
   * Returns the coefficient of the 5th eigenvector in the EMoR basis.
   * @note Only makes sense for EmorResponse photometric responses.
   */
  virtual double getEmorD() const;
  /**
   * Returns the coefficient of the 6th eigenvector in the EMoR basis.
   * @note Only makes sense for EmorResponse photometric responses.
   */
  virtual double getEmorE() const;
  /**
   * Returns the photometric gamma.
   * @note Only makes sense for GammaResponse photometric responses.
   */
  virtual double getGamma() const;
  /**
   * Returns the value-based response curve
   * @note Only valid for CurveResponse photometric responses, otherwise may be NULL
   */
  virtual const std::array<uint16_t, 256>* getValueBasedResponseCurve() const;

  /**
   * Sets the camera response curve to an EMoR-parametrised curve
   */
  virtual void setEmorA(double emorA);
  /**
   * Sets the camera response curve to an EMoR-parametrised curve
   */
  virtual void setEmorB(double emorB);
  /**
   * Sets the camera response curve to an EMoR-parametrised curve
   */
  virtual void setEmorC(double emorC);
  /**
   * Sets the camera response curve to an EMoR-parametrised curve
   */
  virtual void setEmorD(double emorD);
  /**
   * Sets the camera response curve to an EMoR-parametrised curve
   */
  virtual void setEmorE(double emorE);
  /**
   * Sets the camera response to an EMoR-parametrised curve
   */
  virtual void setEmorPhotoResponse(double emorA, double emorB, double emorC, double emorD, double emorE);

  /**
   * Resets the photo response to its default
   */
  virtual void resetPhotoResponse();

  /**
   * Sets the photometric gamma.
   * @note Only makes sense for GammaResponse photometric responses.
   */
  virtual void setGamma(double gamma);
  /**
   * Sets the camera response to a value-based response curve
   */
  virtual void setValueBasedResponseCurve(const std::array<uint16_t, 256>& values);

  // ------------------ Vignetting -----------------------
  /**
   * Returns the constant vignetting coefficient.
   */
  virtual double getVignettingCoeff0() const;
  /**
   * Returns the linear vignetting coefficient.
   */
  virtual double getVignettingCoeff1() const;
  /**
   * Returns the quadratic vignetting coefficient.
   */
  virtual double getVignettingCoeff2() const;
  /**
   * Returns the order three vignetting coefficient.
   */
  virtual double getVignettingCoeff3() const;
  /**
   * Returns the vignetting center (X).
   */
  virtual double getVignettingCenterX() const;
  /**
   * Returns the vignetting center (Y).
   */
  virtual double getVignettingCenterY() const;

  /**
   * Sets the vignetting type to a parametrised radial vignetting
   */
  virtual void setVignettingCoeff0(double vignettingCoeff0);
  /**
   * Sets the vignetting type to a parametrised radial vignetting
   */
  virtual void setVignettingCoeff1(double vignettingCoeff1);
  /**
   * Sets the vignetting type to a parametrised radial vignetting
   */
  virtual void setVignettingCoeff2(double vignettingCoeff2);
  /**
   * Sets the vignetting type to a parametrised radial vignetting
   */
  virtual void setVignettingCoeff3(double vignettingCoeff3);
  /**
   * Sets the vignetting type to a parametrised radial vignetting
   */
  virtual void setVignettingCenterX(double vignettingCenterX);
  /**
   * Sets the vignetting type to a parametrised radial vignetting
   */
  virtual void setVignettingCenterY(double vignettingCenterY);
  /**
   * Sets the vignetting type to a parametrised radial vignetting
   */
  virtual void setRadialVignetting(double vignettingCoeff0, double vignettingCoeff1, double vignettingCoeff2,
                                   double vignettingCoeff3, double vignettingCenterX, double vignettingCenterY);
  /**
   * Resets radial vignetting parameters and disables vignetting
   */
  virtual void resetVignetting();

  /**
   * @brief getInputCenterX
   * @return input center X
   */
  virtual double getInputCenterX() const;

  /**
   * @brief getInputCenterY
   * @return input center Y
   */
  virtual double getInputCenterY() const;

  /**
   * @brief getCenterX
   * @param geometry used geometry input
   * @return lens center X coordinate
   */
  virtual double getCenterX(const GeometryDefinition& geometry) const;

  /**
   * @brief getCenterY
   * @param geometry used geometry input
   * @return lens center Y coordinate
   */
  virtual double getCenterY(const GeometryDefinition& geometry) const;

  // ------------------- Crop -----------------

  /**
   * Returns where to start using pixels, starting from the left border.
   * @note May be negative.
   */
  virtual int64_t getCropLeft() const;
  /**
   * Returns where to stop using pixels, starting from the left border.
   * @note May be larger than the actual width.
   */
  virtual int64_t getCropRight() const;
  /**
   * Returns where to start using pixels, starting from the top border.
   * @note May be negative.
   */
  virtual int64_t getCropTop() const;
  /**
   * Returns where to stop using pixels, starting from the top border.
   * @note May be larger than the actual height.
   */
  virtual int64_t getCropBottom() const;

  /**
   * Sets the input crop values and calculate the relative crop
   * @param left Crop left value
   */
  virtual void setCropLeft(int64_t left);
  /**
   * Sets the input crop values and calculate the relative crop
   * @param right Crop right value
   */
  virtual void setCropRight(int64_t right);
  /**
   * Sets the input crop values and calculate the relative crop
   * @param top Crop top value
   */
  virtual void setCropTop(int64_t top);
  /**
   * Sets the input crop values and calculate the relative crop
   * @param bottom Crop bottom value
   */
  virtual void setCropBottom(int64_t bottom);
  /**
   * Sets the input crop values and calculate the relative crop
   * @param left Crop left value
   * @param right Crop right value
   * @param top Crop top value
   * @param bottom Crop bottom value
   */
  virtual void setCrop(int64_t left, int64_t right, int64_t top, int64_t bottom);
  /**
   * Resets the crop value to the image bounds
   */
  virtual void resetCrop();

  /**
   * Resets all distortion parameters and the geometry and vignetting center
   */
  virtual void resetDistortion();

  /**
   * Compute what focal the input should have if its distortion were removed.
   */
  virtual double computeFocalWithoutDistortion() const;

 protected:
  /**
   * Build with the mandatory fields. The others take default values.
   */
  InputDefinition();

  /**
   * Disabled, use clone()
   */
  InputDefinition(const InputDefinition&) = delete;

  /**
   * Disabled, use clone()
   */
  InputDefinition& operator=(const InputDefinition&) = delete;

  /**
   * Parse from the given ptv. Values not specified are not overridden.
   * @param diff Input diff.
   * @param enforceMandatoryFields If false, ignore missing mandatory values.
   */
  Status applyDiff(const Ptv::Value& diff, bool enforceMandatoryFields);

  /**
   * @brief resetExposure Resets exposure curve values for this InputDefinition.
   */
  virtual void resetExposure();

  /**
   * Parse an Input from a pto line.
   * @param line The input line
   * @param prevInputs The vector of previous InputDefinitions for back references.
   * @return The parsed InputDefinition.
   */
  static Potential<Core::InputDefinition> parseFromPtoLine(char* line,
                                                           const std::vector<Core::InputDefinition*>& prevInputs);

  /**
   * Parse from a pts line.
   * @param line The input line
   * @param prevInputs The vector of previous InputDefinitions for back references.
   * @return The parsed InputDefinition.
   */
  static Potential<Core::InputDefinition> parseFromPtsLine(char* line,
                                                           const std::vector<Core::InputDefinition*>& prevInputs);

 private:
  friend class PanoDefinition;
  friend Potential<InputDefinition> parseFromPtoLine(char*, const std::vector<InputDefinition*>&);

  class Pimpl;
  Pimpl* const pimpl;

  // keep the compiler happy
  using ReaderInputDefinition::operator==;
};
}  // namespace Core
}  // namespace VideoStitch

#endif
