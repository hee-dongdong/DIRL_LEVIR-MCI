# [소프트웨어융합학과 캡스톤디자인] DIRL 모델 기반의 Distractor-Immune Remote Sensing Change Captioning

## Project Summary
- Goal: generate natural-language change captions from satellite image pairs while remaining robust to distractors such as illumination, seasonal shifts, and viewpoint differences.
- Approach: start from the Distractors-Immune Representation Learning (DIRL) backbone and add an auxiliary segmentation head that consumes LEVIR-MCI change masks. The head uses a U-Net style decoder with skip connections to recover pixel-level change shapes, and its loss (weight 0.05) is added to the captioning objective.
- Data: LEVIR-MCI (10,077 image pairs, 50,385 captions) with binary change masks; masks use background=0, red=(255,0,0) -> class 1, yellow=(255,255,0) -> class 2.
- Training setup: Adam (lr 2e-4), batch size 128, max_iter 13,000 (~241 epochs) on GPU. Checkpoints and logs are written under `./experiments/<exp_name>/`.
- Results (LEVIR-MCI): the auxiliary head improves most caption metrics over the DIRL baseline.

| Metric  | DIRL | DIRL + Aux (Ours) |
|---------|------|-------------------|
| BLEU-4  | 57.17| 58.39 |
| METEOR  | 39.13| 39.19 |
| ROUGE-L | 73.87| 73.93 |
| CIDEr   |133.34|133.39 |
| SPICE   |32.18 |31.41 |

Github (code origin): https://github.com/hee-dongdong/DIRL_LEVIR-MCI.git

## Code instruction

### Environment
- Python 3.8+ with CUDA GPU recommended. Core dependencies include PyTorch 1.8.0+cu111, torchvision 0.9.0, transformers 4.39.3, allennlp 2.3.0, faiss-gpu 1.7.2, pycocotools/pycocoevalcap, h5py, and matplotlib.
- `requirements.txt` is a pip list snapshot; drop the first two header lines (starting with `Package` and dashes) before running `pip install -r requirements.txt`, or manually install the pinned versions listed there.
- Example setup:
  ```bash
  python -m venv .venv
  source .venv/bin/activate
  pip install --upgrade pip
  # edit requirements.txt to remove the header lines, then:
  pip install -r requirements.txt
  ```

### Data preparation
- Target dataset: LEVIR-MCI with change captions and binary segmentation masks.
- Expected layout (as referenced in `configs/dynamic/transformer_levir.yaml`):
  - `data_loader/LEVIR-MCI-dataset/images_flattened/{before,after}` and corresponding `{before_feature,after_feature}` `.npy` feature folders.
  - `data_loader/LEVIR-MCI-dataset/images_flattened/annotations/{transformer_vocab.json, transformer_labels.h5, split.json}`.
  - Auxiliary masks at `data_loader/LEVIR-MCI-dataset/images/{train,val,test}/label_rgb/*.png` using the color mapping noted above.
- The repo currently points to an absolute path (`/data/inseong/skrr/DIRL_Capstone/...`); update the paths inside `configs/dynamic/transformer_levir*.yaml` to match your dataset location or adjust the symlink under `data_loader/`.
- Experiment outputs default to `./experiments`; change `exp_dir` in `configs/config_transformer.py` if you prefer a different location.

### Training
- Single-GPU training example (adjust `gpu_id` in the YAML if needed):
  ```bash
  python train.py --cfg configs/dynamic/transformer_levir.yaml
  ```
- Snapshots: `experiments/<exp_name>/snapshots/<exp_name>_checkpoint_<iter>.pt`.
- Validation captions during training are saved to `experiments/<exp_name>/eval_sents/<exp_name>_sents_<iter>/sc_results.json`.
- Loss composition in `train.py`: `total = speaker_loss + 0.03*dirl_loss + 0.05*ccr_loss + 0.05*aux_loss` with gradient clipping controlled by `train.grad_clip`.

### Inference and evaluation
- Generate captions (and optional mask visualizations) from a saved checkpoint:
  ```bash
  python test.py --cfg configs/dynamic/transformer_levir.yaml --snapshot 13000 --save_masks --num_mask_samples 50
  ```
  Outputs live in `experiments/<exp_name>/test_output_<snapshot>/`, including `captions/sc_results.json`, attention maps, and `predicted_masks/` when `--save_masks` is set.
- To score generated captions against the ground-truth annotations, use `evaluate.py` or `utils.eval_utils.score_generation` with the same annotation files referenced in your config. `evaluate.py` expects a directory that contains the generated `sc_results.json` (and optionally `nsc_results.json`), plus the dataset `anno` and a `type_file` describing change types.

## Demo
- Minimal dry-run using an existing checkpoint:
  1. Prepare the LEVIR-MCI data in the expected layout and update the config paths.
  2. Activate your environment and ensure the required packages are installed.
  3. Run `python test.py --cfg configs/dynamic/transformer_levir.yaml --snapshot <iter> --save_masks --visualize`.
  4. Inspect outputs under `experiments/<exp_name>/test_output_<iter>/` (captions, attention visualizations, and predicted masks colored red/yellow for buildings/roads).

## Conclusion and Future Work
- The auxiliary segmentation head strengthens visual grounding and improves BLEU-4/METEOR/ROUGE-L/CIDEr compared to the DIRL baseline on LEVIR-MCI, demonstrating positive transfer from mask prediction to caption quality.
- Limitations: SPICE lags the baseline, and results do not yet surpass the latest SOTA models such as Chg2Cap. Performance also depends on the quality of pre-extracted features and mask labels.
- Future directions: try stronger backbones or vision transformers for feature extraction, tune auxiliary loss weighting or multi-task schedules, add data augmentation to cover illumination/viewpoint distractors, and explore more detailed mask semantics beyond the current binary change classes.
