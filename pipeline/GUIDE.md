# Brain-to-Text 2025 Pipeline Guide

Quick start for training, evaluation, and test prediction generation.

Prereqs
- Python env with `torch` + CUDA build if using GPU.
- Data available at `/srv/s8l_storage/kaggle/data/brain-to-text-25/hdf5_data_final`.

Install
```bash
cd /scratch2/s8l/kaggle/aleks/repo
pip install -r pipeline/requirements.txt
```

Sanity check GPU
```bash
python - <<'PY'
import torch
print("cuda_available:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())
PY
```

Training (baseline phoneme CTC)
```bash
python pipeline/meta/train.py pipeline/meta/configs/rnn_gru_baseline.yaml
```

Training (char CTC, local WER without Redis)
```bash
python pipeline/meta/train.py pipeline/meta/configs/rnn_gru_char_ctc.yaml
```

Validation metrics
- Baseline CTC reports `val_loss` and `val_per`.
- Char CTC reports `val_loss`, `val_cer`, and `val_wer` (local WER).

Evaluation (val split)
```bash
python pipeline/meta/evaluate.py pipeline/meta/configs/rnn_gru_baseline.yaml --split val
```

Test predictions (no labels)
```bash
python pipeline/meta/evaluate.py pipeline/meta/configs/rnn_gru_baseline.yaml \
  --split test \
  --output runs/rnn_gru_baseline/test_preds.jsonl
```

Submission CSV (Kaggle format)
```bash
python pipeline/meta/generate_submission.py pipeline/meta/configs/rnn_gru_char_ctc.yaml \
  --checkpoint /srv/s8l_storage/kaggle/aleks/runs/rnn_gru_char_ctc/checkpoints/best.pt \
  --output runs/rnn_gru_char_ctc/submission.csv
```

If you already have predictions JSONL:
```bash
python pipeline/meta/generate_submission.py pipeline/meta/configs/rnn_gru_char_ctc.yaml \
  --predictions runs/rnn_gru_char_ctc/test_preds.jsonl \
  --output runs/rnn_gru_char_ctc/submission.csv
```

Checkpoint selection
- By default, `meta/evaluate.py` loads `runs/<experiment>/checkpoints/best.pt`.
- To override:
```bash
python pipeline/meta/evaluate.py pipeline/meta/configs/rnn_gru_baseline.yaml \
  --checkpoint /srv/s8l_storage/kaggle/aleks/runs/rnn_gru_baseline/checkpoints/best.pt \
  --split val
```

Optional: competition WER with Redis LM (baseline task)
1. Start Redis and the LM worker from the NEJM baseline repo.
2. In `pipeline/meta/configs/rnn_gru_baseline.yaml`, set:
   - `competition.wer.enabled: true`
3. Run training or evaluation to get `val_wer`.

Adding a new architecture
1. Drop a model in `pipeline/models/architectures/` (single class).
2. Create a YAML config in `pipeline/meta/configs/` pointing to it:
   - `model.module`, `model.class`, `model.params`
3. Run training:
```bash
python pipeline/meta/train.py pipeline/meta/configs/<your_config>.yaml
```

Notes
- `training.require_cuda: true` enforces GPU usage; set `false` to allow CPU.
- `training.progress_bar: true` enables tqdm progress bars.
- `training.resume_checkpoint` can be set to continue from a checkpoint.
