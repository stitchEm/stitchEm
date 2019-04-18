Usage: autocrop-cmd -i <template> -o <output.json> [options]";
-i <template>: can be an image file / or directory that contains all inputs for detection.
               The input file can be PNG, JPEG or JPG.
-o <output.json>: the output json file to store all computed results.
[options] are
    d   : Set d to dump circle overlay on top of the input image.
          The new images will be dumpped at the same place with the input.

Output json format:
  A typical output field will contain:
  - "reader_config" 	: the input file path
  - "width"           : input file width
  - "height"          : input file height
  - "proj"            : type of projection, always "circular_fisheye" for now
  - "center_x"        : center of the circle in the X coordinate
  - "center_y"        : center of the circle in the Y coordinate
  - "radius"          : circle radius
  - "output_circle"   : path to the debug file if options -d was used
  - "crop_left"       : center_x - radius
  - "crop_right"      : center_x + radius
  - "crop_top"        : center_y - radius
  - "crop_bottom"     : center_y + radius

Example output:
  {
    "reader_config" : "\\Assets\\VideoStitch-assets\\autoCrop\\Orah\\orahFisheye.png",
    "width" : 1096,
    "height" : 822,
    "proj" : "circular_fisheye",
    "center_x" : 551,
    "center_y" : 405,
    "radius" : 635,
    "output_circle" : "\\Assets\\VideoStitch-assets\\autoCrop\\Orah\\orahFisheye.png_circle.jpg"
  }

Example usages:

- To find the circle of a single input image:
    autocrop-cmd -i "C:\\autoCrop\\input-07.jpg" -o "C:\\autoCrop\\output.json" -d

- To find the circle of all images inside a folder
    autocrop-cmd -i "C:\\autoCrop\\Orah" -o "C:\\autoCrop\\output.json" -d
