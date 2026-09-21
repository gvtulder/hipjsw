# Evaluation of the hip detection model

_Gijs van Tulder, September 2026_

## About the model

The hipJSW joint space width measurement algorithm expects a cropped input centered on the hip joint. This location can be provided manually or, as in the paper, through landmark points provided by the BoneFinder software.

For convenience, we also provide a hip detection model based on [YOLOLite](https://github.com/Lillthorin/YoloLite-Official-Repo) that automatically detect hips if no exact location is given. The same model can also be used to provide a rough estimate of the pixel spacing if no pixel spacing is available (e.g., for JPEG inputs without known pixel spacing).

This short report describes how this model was trained and evaluates its performance.


## Dataset

The model was trained and evaluated on the data from the World COACH consortium that was also used for the main algorithm. The table shows the number of scans per cohort, divided between training, validation, and test sets.

| Cohort    | Training   | Validation  | Test     |
|-----------|------------|-------------|----------|
| CHECK     |     337    |        61   |     56   |
| Chingford |     486    |        97   |     94   |
| FORCE     |     161    |        31   |     37   |
| JoCo      |    2456    |       574   |    551   |
| MOST      |     581    |       115   |    106   |
| OAI       |    3044    |       667   |    655   |
| RS3       |    1730    |       374   |    371   |
| SF        |    3531    |       709   |    764   |
| **Total** | **12326**  |    **2628** | **2634** |

Most scans show one or two hips in pelvic radiographs. The OST cohort contains full-length leg scans.

The annotations were derived from BoneFinder landmarks, which were used to compute the coordinates of each hip: center x and y of the femoral head, and the femoral head diameter. For this YOLO model, the coordinates were converted to square ROIs centered on the femoral head, with width and height equal to the femoral head diameter. Left and right hips were labeled with the appropriate class label.


## Training

The model trained was the `edge_s` architecture from the YOLOLite repository, with input images resized and letterboxed to fit within 640 by 640 pixels. The model was trained on the training images, with the validation images used for early stopping based on validation loss.


## Detection

On unseen images, the model outputs candidate ROIs for left and right hips sorted by confidence above a minimum threshold. For our purposes, we use the highest-scoring left hip and highest-scoring right hip as the output. The ROI is converted to an estimate for the femoral head center and diameter.


## Evaluation

The model was evaluated on the images from the test set. Results are compared against the BoneFinder annotations.

### Hip detection

The model correctly detected almost all hips in the test set. The femoral head localization is fairly precise. The distance between the YOLOLite-estimated femoral head center and the BoneFinder-estimated center is sufficiently small for a reliable cropping. (For context: the input to the joint space segmentation and measurement model is a 10 x 10 cm area centered on the femoral head.)

Note that this is an optional first step: if necessary, the hip detection can be checked and corrected manually before calculating the joint space width.

| Cohort     | Hips detected | Hips missed | Mean error (mm) | Std (mm) |
|------------|---------------|-------------|-----------------|----------|
| CHECK      |           110 |             |            0.58 |     0.32 |
| Chingford  |           170 |           3 |            0.59 |     0.39 |
| FORCE      |            70 |             |            0.60 |     0.42 |
| JoCo       |          1098 |             |            0.54 |     0.36 |
| MOST       |           209 |             |            1.27 |     0.77 |
| OAI        |          1307 |             |            0.56 |     0.38 |
| RS3        |           732 |             |            0.58 |     0.41 |
| SF         |          1489 |             |            0.54 |     0.35 |

<img style="width:50%; height:auto;" alt="Error distribution of femoral head estimates" title="Error distribution of femoral head estimates" src="images/femoral-head-errors.png?raw=true">


### Pixel space estimation

The hip detection model can also be used to estimate the pixel spacing in scans where this is unknown, such as JPEG images. For this purpose, we estimate the pixel spacing by comparing the diameter of the ROI detected by the hip detection model with a standard femoral head diameter of 55 mm (the rounded average in our dataset).

For most scans in our test set, the estimated pixel spacing is fairly close to the true pixel spacing. However, this remains a fallback option: it is always recommended to provide the true pixel spacing if this is known to obtain the most accurate joint space width measurements.

<img style="width:50%; height:auto;" alt="Estimated vs true pixel spacing" title="Estimated vs true pixel spacing" src="images/pixel-spacing-estimate-errors.png?raw=true">


