# hipJSW – Measuring joint space width in hip X-ray images

This directory contains the training and preprocessing scripts used to create the models used in the [hipJSW](https://github.com/gvtulder/hipjsw) program for the automated measurement of joint space width in hip radiographs.


## Outline of the training code

Note: This contains the code used to train the original models, provided mostly for reference. You may need to change things to make this work on your system and on your data. If you don’t want to train your own models, look at the [hipJSW](https://github.com/gvtulder/hipjsw) program to hip joint space width measurements.

**[``lists``](lists/)**: Train/validation/test splits used in training.

**[``manual-annotations``](manual-annotations/)**: Some utility code to convert manual annotations made in 3D Slicer to a JSON format that can be used in preprocessing.

**[``preprocessing``](preprocessing/)**: Preprocessing code to load DICOM and JPEG images, along with BoneFinder landmark points, and convert these to cropped hips in HDF5 format.

**[``segmentation``](segmentation/)**: Training and evaluation of the hip segmentation model.

**[``detection``](detection/)**: The hip detection model. This is not used for the paper, but only as a convenience helper in the hipJSW program.


## Authors, validation paper, citation

This code was written by [Gijs van Tulder](https://www.vantulder.net/) at the [Delft University of Technology](https://www.tudelft.nl/ewi/over-de-faculteit/afdelingen/intelligent-systems/pattern-recognition-bioinformatics/pattern-recognition) in 2026 and is made available under the [GPLv3.0 license](https://www.gnu.org/licenses/gpl-3.0.en.html). Models were trained using data from the World COACH consortium.

See the following paper for a description and evaluation of the algorithm:

> *Automated joint space width measurements for the hip joint: an open-source method*
>
> Van den Berg M.A., van Tulder G., Ahedi H., Arden N., Bierma-Zeinstra S.M.A., Boer C.G., van Buuren M.M.A., Cicuttini F.M., Cootes T.F., Felson D.T., Gielis W.P., Heerey J.J., Jones G., Kemp J., Kluzek S., Lane N.E., Lindner C., Lynch J.A., van Meurs J.B.J., Mosler A., Nelson A.E.,  Nevitt M.C., Oei E.H., Riedstra N.S., Runhaar J., Tang J., Weinans H., Krijthe J.H., Boel F., Agricola R. (2026)
>
> ...

If you use this software in your research, please cite the above paper and refer to the software on Zenodo:

> ...

As the license implies, this software is not in any way approved for use in clinical practice. Use at your own risk and always validate the results if you care about their correctness.


## License

Copyright © 2026 Gijs van Tulder / TU Delft.

The code in this repository is made available under the [GPLv3.0 license](https://www.gnu.org/licenses/gpl-3.0.en.html).

Following the [TU Delft Guidelines on Research Software](https://doi.org/10.5281/zenodo.4629634):

> Technische Universiteit Delft hereby disclaims all copyright interest in the program “hipJSW” written by Gijs van Tulder.
>
> Lucas van Vliet, Dean of the Faculty of Electrical Engineering, Mathematics and Computer Science (EEMCS)

The hip detection model (first stage of preprocessing, see [`hip_detector.py`](src/hipjsw/hip_detector.py)) is based on the [YOLOLite](https://github.com/Lillthorin/YoloLite-Official-Repo) project, which is available under the [Apache 2.0 license](https://www.apache.org/licenses/LICENSE-2.0).

