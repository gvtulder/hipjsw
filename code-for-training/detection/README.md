# Hip detection model

The hip detection model is based on code from [YOLOLite](https://github.com/Lillthorin/YoloLite-Official-Repo), which is available under the [Apache 2.0 license](https://www.apache.org/licenses/LICENSE-2.0).

## Changes to YOLOLite

* In ``scripts/data/augment.py``, disable horizontal and vertical flipping.
* In ``tools/evaluate.py``, add a ``--test_folder_labels`` option to support alternative path structures.

## Preprocessing

Convert images to JPG and derive the ROIs from BoneFinder points files:

```
./prepare-for-yolo-202606.sh
```

## Training

Train the model on the images and labels:

```
./train_hip.sh
```

## Evaluation

Optionally, evaluate and compute statistics:

```
./evaluate_hip.sh
```

To compare the performance of the hip detection against BoneFinder annotations, run inference on all images, then compute the statistics and generate plots.

```
./infer_hip_onnx.sh
python collect-stats.py
```

## Export trained model to ONNX format

Convert the final model to ONNX format for use in the hipJSW program:

```
./onnx_export_hip.sh
```

Find the ``model_decoded.onnx`` file in ``runs/export/``.
