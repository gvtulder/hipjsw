# right hip is left on image, stored in dcm_L.pts
python loader.py \
  --input-image ../images/OAI-9763898-V00-20051017.dcm \
  --input-points ../images/OAI-9763898-V00-20051017.dcm_L.pts \
  --side right \
  --save-image "test-{side}-from-points.png"
# left hip is right on image, stored in dcm_RasL.pts
python loader.py \
  --input-image ../images/OAI-9763898-V00-20051017.dcm \
  --input-points ../images/OAI-9763898-V00-20051017.dcm_RasL.pts \
  --side left \
  --save-image "test-{side}-from-points.png"

# specify center-x, center-y
python loader.py \
  --input-image ../images/OAI-9763898-V00-20051017.dcm \
  --center-x 810 \
  --center-y 1368 \
  --side right \
  --save-image "test-{side}-from-coord.png"
python loader.py \
  --input-image ../images/OAI-9763898-V00-20051017.dcm \
  --center-x 2132 \
  --center-y 1381 \
  --side left \
  --save-image "test-{side}-from-coord.png"

# specify nothing: use hip detector
python loader.py \
  --input-image ../images/OAI-9763898-V00-20051017.dcm \
  --save-image "test-{side}-from-yolo.png"
