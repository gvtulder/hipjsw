# Hip segmentation model checkpoints

These `torch.save` checkpoints are the models used in the paper.

* The final model fine-tuned on manual annotations:<br>
  ``checkpoint-60421_1-best-val-loss-epoch=144-step=435.ckpt``<br>
  Training parameters:<br>
  ``args-60421_1-best-val-loss-epoch=144-step=435.json``

* The initial model trained on BoneFinder annotations:<br>
  ``checkpoint-60384_1-best-val-loss-epoch=184-step=143930.ckpt``<br>
  Training parameters:<br>
  ``args-60384_1-best-val-loss-epoch=184-step=143930.json``

Alternative models for 0.2mm/pixel resolution:

* The final model fine-tuned on manual annotations:<br>
  ``checkpoint-19160_10-best-val-loss-epoch=227-step=684.ckpt``<br>
  Training parameters:<br>
  ``args-19160_10-best-val-loss-epoch=227-step=684.json``

* The initial model trained on BoneFinder annotations:<br>
  ``checkpoint-18915_1-best-val-loss-epoch=199-step=155600.ckpt``<br>
  Training parameters:<br>
  ``args-18915_1-best-val-loss-epoch=199-step=155600.json``
