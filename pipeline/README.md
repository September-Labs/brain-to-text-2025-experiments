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
- Character CTC (no Redis, local WER): `python meta/train.py meta/configs/rnn_gru_char_ctc.yaml`

Fine-tune
- Point `training.resume_checkpoint` in the YAML to a previous checkpoint.
- Set `training.resume_optimizer: true` if you want to continue optimizer state.

Evaluate / test
- Validation with metrics: `python meta/evaluate.py meta/configs/rnn_gru_baseline.yaml --split val`
- Test predictions: `python meta/evaluate.py meta/configs/rnn_gru_baseline.yaml --split test --output runs/rnn_gru_baseline/test_preds.jsonl`
- Submission CSV: `python meta/generate_submission.py meta/configs/rnn_gru_char_ctc.yaml --output runs/rnn_gru_char_ctc/submission.csv`

Notes
- Full walkthrough: `GUIDE.md`
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
- Character CTC computes WER locally (no LM). Ensure `data.n_classes` matches `len(task.params.vocab) + 1` (blank).

WER + LM service quickstart (from the NEJM baseline)
1. Start a redis server (if not already running): `redis-server --port 6379`
2. In the NEJM repo root, launch the LM worker:
   - 1-gram (fast, low RAM):
     `python language_model/language-model-standalone.py --lm_path language_model/pretrained_language_models/openwebtext_1gram_lm_sil --do_opt --nbest 100 --acoustic_scale 0.325 --blank_penalty 90 --alpha 0.55 --redis_ip localhost --gpu_number 0`
   - 3-gram (~60GB RAM):
     `python language_model/language-model-standalone.py --lm_path language_model/pretrained_language_models/openwebtext_3gram_lm_sil --do_opt --nbest 100 --acoustic_scale 0.325 --blank_penalty 90 --alpha 0.55 --redis_ip localhost --gpu_number 0`
   - 5-gram (~300GB RAM):
     `python language_model/language-model-standalone.py --lm_path language_model/pretrained_language_models/openwebtext_5gram_lm_sil --rescore --do_opt --nbest 100 --acoustic_scale 0.325 --blank_penalty 90 --alpha 0.55 --redis_ip localhost --gpu_number 0`
3. Enable WER: set `competition.wer.enabled: true` in the YAML.
