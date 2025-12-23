Brain-to-Text '25 experiments

Layout
- `meta/train.py`: static training entrypoint (config-driven)
- `meta/evaluate.py`: evaluation/test runner for val/test splits
- `meta/configs/`: YAML configs (one per architecture/run)
- `models/architectures/`: drop-in model architectures
- `b2txt/`: dataset + augmentation utilities
- `b2txt/tasks/`: task definitions (loss/metrics/predictions)

Data
- Dataset root: `/srv/s8l_storage/kaggle/data/brain-to-text-25/hdf5_data_final`
- Sessions are listed in `meta/configs/rnn_gru_baseline.yaml` (remove to auto-discover)

Train (baseline RNN)
- `python meta/train.py meta/configs/rnn_gru_baseline.yaml`

Fine-tune
- Point `training.resume_checkpoint` in the YAML to a previous checkpoint.
- Set `training.resume_optimizer: true` if you want to continue optimizer state.

Evaluate / test
- Validation with metrics: `python meta/evaluate.py meta/configs/rnn_gru_baseline.yaml --split val`
- Test predictions: `python meta/evaluate.py meta/configs/rnn_gru_baseline.yaml --split test --output runs/rnn_gru_baseline/test_preds.jsonl`

Notes
- `meta/evaluate.py` writes per-trial phoneme predictions (JSONL) when `--output` is provided.
- `b2txt/data.py` supports unlabeled test data via `allow_unlabeled`.
- Augmentations are configured under `data.transforms` in the YAML; both train/eval apply them in `b2txt/tasks/ctc_phoneme.py`.
- `task` in the YAML selects the training/eval logic (e.g., CTC phoneme decoding).
- `training.require_cuda: true` enforces GPU usage and will error if CUDA is unavailable.
- For H100, install a CUDA-enabled torch build (e.g. `pip install --index-url https://download.pytorch.org/whl/cu124 torch`).
- Training progress uses a tqdm bar (disable with `training.progress_bar: false`).
- The CTC task reports validation loss and phoneme error rate (PER).
- Competition WER is supported via the Redis LM decoder; set `competition.wer.enabled: true` and run the LM service before training/validation.
- To emit decoded sentences during test/eval JSONL output, set `competition.wer.decode_predictions: true`.
