# PTS file format description

## Global

cameracurve <invEmor1> <invEmor2> <invEmor3> <invEmor4> <invEmor5>
Inverse Emor coefficients for camera response.

wbexposure <red correction> <blue correction> <ev correction> (log scale, 0 means no correction)
Global panorama exposure params.

exposurecorrection <enabled>
Is HDR exposure correction enabled ?

## Per-input

imgfile <width> <height> "<filename>"
Sets properties for next 'i' line.

viewpoint <tx> <ty> <tz> <pan> <tilt>
Sets viewpoint parameters for next 'i' line.

vignettingparams <vA> <vB> <vC> <vD> <vE>
Sets vignetting parameters for next 'i' line.

exposureparams <red correction> <blue correction> <flare> <ev correction> (log scale, 0 means no correction)
Sets exposure parameters for next 'i' line.
