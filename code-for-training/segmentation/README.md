# Hip segmentation model

The segmentation model is trained in two phases. First, the model is trained on the large set of images with BoneFinder annotations. After that, the model is fine-tuned on a much smaller set of manually annotated images. The final model can then be applied to new images, or exported to ONNX format for use in the hipJSW program.

See the [preprocessing scripts](../preprocessing/) to generate the HDF5 files. Update the example scripts to match the paths on your system.

## Training

First, pretrain on segmentations derived from BoneFinder landmarks:

```
./run_train_bonefinder.sh
```

Then, fine-tune on manual segmentations:

```
./run_train_finetune_manual.sh
```

## Prediction

Use the trained model to generate segmentations for preprocessed images:

```
./predict_example.sh
```

## Export to ONNX

Convert the PyTorch model to ONNX format for use in the hipJSW program. Provide the checkpoint (``.ckpt``) and arguments file (``args.json``) to export, along with the desired output filename:

```
python export_to_onnx.py \
  --checkpoint ${CHECKPOINT} \
  --model-args ${ARGSJSON} \
  --output ${OUTPUT}
```
