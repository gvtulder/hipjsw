# hipJSW – Measuring joint space width in hip X-ray images

**Automated hip joint space width measurements from pelvic radiographs.**

This utility automatically segments the hip joint and computes the hip joint space at various locations. Inputs can be pelvic X-ray images showing both hips or images of individual hips, in DICOM or JPEG format.

<img style="width:27%; height:auto;" alt="Full pelvic radiograph (© Nevit Dilmen, CC BY-SA 3.0)" title="Full pelvic radiograph (© Nevit Dilmen, CC BY-SA 3.0)" src="docs/images/Medical_X-Ray_imaging_SAL07_nevit.jpg?raw=true"> <img style="width:30%; height:auto;" alt="Analysis of right hip" title="Analysis of right hip" src="docs/images/jsw-Medical_X-Ray_imaging_SAL07_nevit.jpg-right.png?raw=true"> <img style="width:30%; height:auto;" alt="Analysis of right hip" title="Analysis of left hip" src="docs/images/jsw-Medical_X-Ray_imaging_SAL07_nevit.jpg-left.png?raw=true">

*Example scan [© Nevit Dilmen on Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Medical_X-Ray_imaging_XAN07_nevit.jpg), CC BY-SA 3.0.*

## Authors, validation paper, citation

This program was written by [Gijs van Tulder](https://www.vantulder.net/) at the [Delft University of Technology](https://www.tudelft.nl/ewi/over-de-faculteit/afdelingen/intelligent-systems/pattern-recognition-bioinformatics/pattern-recognition) in 2026 and is made available under the [GPLv3.0 license](https://www.gnu.org/licenses/gpl-3.0.en.html). Models were trained using data from the World COACH consortium.

See the following paper for a description and evaluation of the algorithm:

> *Automated joint space width measurements for the hip joint: an open-source method*
>
> Van den Berg M.A., van Tulder G., Ahedi H., Arden N., Bierma-Zeinstra S.M.A., Boer C.G., van Buuren M.M.A., Cicuttini F.M., Cootes T.F., Felson D.T., Gielis W.P., Heerey J.J., Jones G., Kemp J., Kluzek S., Lane N.E., Lindner C., Lynch J.A., van Meurs J.B.J., Mosler A., Nelson A.E.,  Nevitt M.C., Oei E.H., Riedstra N.S., Runhaar J., Tang J., Weinans H., Krijthe J.H., Boel F., Agricola R. (2026)
>
> ...

If you use this software in your research, please cite the above paper and refer to the software on Zenodo:

> ...

As the license implies, this software is not in any way approved for use in clinical practice. Use at your own risk and always validate the results if you care about their correctness.


## Outline of the program

1. **Input.** You can provide individual DICOM or JPEG images, point to a directory with images, or provide a CSV file listing images and other parameters.

2. **Hip detection.** Unless the hip location is given as input, a YOLO-style detection model detects the left and/or right hip joints in the image. See the note on [training and evaluation of the hip detection model](docs/hip_detection_evaluation.md) for more details.

   *Note:* If the hip detection fails to find the hip joints, you can provide the coordinates of the femoral head manually or by providing BoneFinder landmark points.

3. **Pixel spacing.** For DICOM images, the pixel spacing is derived from the DICOM headers. For JPEG images or other images without known pixel spacing, the pixel spacing is estimated automatically from the hip detection in step 1, assuming a fixed femoral head size of 55 mm.

   *Note:* If no pixel spacing is known or given, the joint space measurents will be relative to the estimated pixel spacing, and might deviate from the true measurements. Input the correct pixel spacing manually to get exact measurements.

4. **Image segmentation.** The area around the hip joint is segmented using a UNet-style convolutional neural network to detect the femoral head, sourcil, and joint space.

5. **Measurement.** The segmentation is analyzed to measure the joint space along the sourcil curve.

6. **Output.** The measurements can be saved in CSV or JSON format and visualized in a diagram.


## Installation

### pip / pipx / uv

Install the `hipjsw` package in a virtual environment or directly on your system, then run `hipjsw`:

```
pip install https://github.com/gvtulder/hipjsw.git
hipjsw --help
```

See below for additional command-line arguments.

### Podman / Docker container

Download and run directly from GitHub Packages:

```
podman run --rm ghcr.io/gvtulder/hipjsw:main --help
```

Make sure to mount the locations that store your data. For example, to map the current working directory to `/workdir` (the container's working directory):

```
podman run --rm -v .:/workdir ghcr.io/gvtulder/hipjsw:main --help
```

See below for additional command-line arguments.

### From source

Clone this repository, install the requirements, and run the module:

```
git clone https://github.com/gvtulder/hipjsw
cd hipjsw
python3 -m virtualenv venv && . venv/bin/activate
pip install -r requirements.txt
python3 src/hipjsw --help
```

See below for additional command-line arguments.


## Usage

### File input

In its simplest form, provide a hip X-ray image in DICOM or JPEG format to compute the joint space measurements:

```
hipjsw hip-x-ray.jpg
```

Similarly, run on multiple images at once or point to a directory:

```
hipjsw hip-x-ray-a.dcm hip-x-ray-b.dcm
hipjsw images/
```

### CSV input

For more detailed input, provide a list of images in CSV format:

```
input_image,scan_id
hip-x-ray-a.dcm,Patient A
hip-x-ray-b.dcm,Patient B
```

and run

```
hipjsw input.csv
```

Only the `input_image` column is required. Some image-specific parameters can be specified as well:

| Name                  | Description                                                                |
|-----------------------|----------------------------------------------------------------------------|
| `input_image`         | Required. Location of the image.                                           |
| `input_points`        | Location of the BoneFinder points file.                                    |
| `input_pixel_spacing` | Pixel spacing in mm/pixel.                                                 |
| `center_x`            | The center x coordinate of the femoral head (in pixels).                   |
| `center_y`            | The center y coordinate of the femoral head (in pixels).                   |
| `side`                | The side of the hip (left/right) for the given coordinates or points file. |
| `scan_id`             | Identifier/title of this image for use in visual outputs and filenames.    |

Any additional, unknown fields are copied to the CSV or JSON output.

Other CSV-related options:

``--images-path``: provide a base path for the `input_image` field.

``--points-path``: provide a base path for the `input_points` field.

### Output

Measurements can be saved as CSV, JSON, NumPy objects, or visualizations.

**Global output options:**

``--output-csv``: a filename for CSV output.

``--output-json``: a filename for JSON output.

``--print-json``: print JSON output directly.

**Per-image output options:**

``--output-plots``: filename for visualizations.

``--output-trace``: filename for detailed NumPy savez outputs.

``--plot-types detail``: plot an overview of the measurements (default).

``--plot-types overlay``: plot the segmentation of the input image.

**Variable filenames:**

When using multiple input images, use the variables `{scan_id}`, `{side}`, `{plot}` to interpolate image-specific filenames. For example:

```
--output-plots "outputs/plot-{scan_id}-{side}-{plot}.png"
--output-trace "outputs/plot-{scan_id}-{side}.png"
```


## Notes on left and right

Note that in the standard arrangement of hip radiographs, the left hip is shown on the right of the image, and the right hip on the left. The measurements refer to the anatomical sides.

To simplify the segmentation and analysis, the left hips (right on the image) are flipped horizontally during preprocessing. The output images use the same orientation. To show the left hip in its original orientation, use the `--plot-left-right` option.


## License

Copyright © 2026 by Gijs van Tulder / TU Delft.

The code in this repository is made available under the [GPLv3.0 license](https://www.gnu.org/licenses/gpl-3.0.en.html).

The hip detection model (first stage of preprocessing, see `src/hipjsw/hip_detector.py`) is based on the [YOLOLite](https://github.com/Lillthorin/YoloLite-Official-Repo) project, which is available under the [Apache 2.0 license](https://www.apache.org/licenses/LICENSE-2.0).

