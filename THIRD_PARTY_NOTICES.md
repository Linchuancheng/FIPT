# Third-party attribution

The training utilities derive from AdaBrain-Bench and LaBraM. Model components derive from the supplied FIPT modifications of LaBraM and CBraMod. The consolidation retains their computations and parameter names. Original source hashes are recorded in `configs/source_manifest.json`.

- AdaBrain-Bench: Junyu Wang and contributors; the supplied MIT license is in `licenses/AdaBrain-Bench-LICENSE`.
- CBraMod: Jiquan Wang and contributors; the supplied MIT license is in `licenses/CBraMod-LICENSE`.
- LaBraM: Wei-Bang Jiang and contributors, https://github.com/935963004/LaBraM . Its upstream attributions also identify BEiT-v2, timm, DeiT, DINO and BIOT.
- PyTorch, timm, einops, NumPy, SciPy and PyHealth retain their own licenses. Checkpoints and datasets are obtained separately under their providers' terms.

The bundled licenses apply to their respective upstream contributions; this document does not assign authorship of FIPT to those projects.

## Original source attribution headers

```text
# --------------------------------------------------------
# Ada-Brainbench
# Based on LaBraM, EEG_Image_decode, BEiT-v2, timm, DeiT, and DINO code bases
# https://github.com/935963004/LaBraM
# https://github.com/ncclab-sustech/EEG_Image_decode
# https://github.com/microsoft/unilm/tree/master/beitv2
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# https://github.com/facebookresearch/dino
# ---------------------------------------------------------
```

```text
# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# By Wei-Bang Jiang
# Based on BEiT-v2, timm, DeiT, DINO, and BIOT code bases
# https://github.com/microsoft/unilm/tree/master/beitv2
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# https://github.com/facebookresearch/dino
# https://github.com/ycq091044/BIOT
# ---------------------------------------------------------
```

```text
# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# By Wei-Bang Jiang
# Based on BEiT-v2, timm, DeiT, and DINO code bases
# https://github.com/microsoft/unilm/tree/master/beitv2
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# https://github.com/facebookresearch/dino
# ---------------------------------------------------------
```