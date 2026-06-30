# Manual annotations from 3D Slicer

Utility script to convert annotations from a 3D Slicer scene (``.mrb``) to a more machine-friendly hip annotation in JSON format.

Load an image in 3D Slicer and annotate four curves:

* left femur
* left sourcil
* right femur
* right sourcil

Convert the 3D Slicer scene to JSON:

```
python parse_scene.py \
  --input input-annotation-scene.mrb \
  --image input-image.dcm \
  --output-overlay output-overlay.png \
  --output-json output-curves.json
```
