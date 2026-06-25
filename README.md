# Easy-to-Hard German Continued Pretraining for a Small English MLM

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lxldraco/German-MLM-Curriculum-MVP/blob/main/notebooks/german_mlm_curriculum_mvp.ipynb)

A small research MVP testing whether a readability-based easy-to-hard curriculum helps a small English masked language model adapt to German better than mixed-difficulty continued pretraining.

This project was developed for the course **Neural Networks in Practice** at the University of Tübingen. It is intended as a transparent course-project prototype rather than a production-ready German language model.

## Project Overview

The project asks a focused question:

> Does easy-to-hard continued pretraining help a small English masked language model adapt to German more effectively than mixed-difficulty continued pretraining?

The original project idea involved linguistically targeted masking strategies, such as POS-specific or grammar-oriented masking. For this MVP, the scope was reduced to a more feasible variable: the **ordering of German training data by readability level**.

The experiment uses the German4All readability-controlled German corpus and compares two training conditions under the same model, tokenizer, masking probability, sequence length, optimizer settings, and training budget:

1. **Mixed baseline**: easy, medium, and hard German examples are mixed throughout training.
2. **Strict easy-to-hard curriculum**: training is divided into easy, medium, and hard stages.

The main metric is masked-language-modeling validation loss on easy, medium, hard, and overall validation splits.

## Repository Status

This repository currently represents an **MVP / course-project prototype**.

- The notebook is designed for Google Colab.
- The MVP uses `prajjwal1/bert-tiny` with the `bert-base-uncased` tokenizer.
- The experiment uses German4All as the only dataset.
- The current result does **not** show a clear final-loss advantage for the strict easy-to-hard curriculum.
- The project should not be interpreted as a competitive German model or as a general proof of curriculum learning.

## Research Question and Hypotheses

### Main research question

Does easy-to-hard continued pretraining help a small English masked language model adapt to German more effectively than mixed-difficulty continued pretraining?

### H1: Final-loss curriculum advantage

Under the same training budget, an easy-to-hard curriculum will produce lower MLM validation loss than mixed-difficulty training, especially on medium and hard German validation data.

### H2: Phase-specific curriculum effect

Even if final overall loss is not better, the curriculum condition may show stronger local improvement on the validation split corresponding to the current training phase.

### H3: Mixed exposure as a strong alternative

Mixed training may perform similarly or better because it exposes the model to the full German difficulty distribution from the beginning.

## Method

### Dataset

The MVP uses **German4All**, a German readability-controlled paraphrasing dataset. The five German4All readability levels are collapsed into three broad difficulty groups:

| Group | Role in curriculum | Training examples | Validation examples |
|---|---|---:|---:|
| Easy | first curriculum stage | 45,828 | 5,090 |
| Medium | second curriculum stage | 22,914 | 2,545 |
| Hard | final curriculum stage | 45,828 | 5,090 |
| Total | — | 114,570 | 12,725 |

German4All is useful for this MVP because it directly provides readability-related difficulty information. A major limitation is that human readability difficulty may not correspond to MLM difficulty, especially when using an English WordPiece tokenizer on German text.

### Model and tokenizer

The MVP uses:

- Model: `prajjwal1/bert-tiny`
- Tokenizer: `bert-base-uncased`
- Objective: masked language modeling
- MLM probability: `0.15`
- Maximum sequence length: `128`

The English tokenizer is part of the cross-lingual adaptation setup, but it also creates a tokenizer mismatch for German capitalization, compounds, inflection, umlauts, and subword fragmentation.

### Training setup

| Setting | Value |
|---|---:|
| Total training steps | 900 |
| Batch size | 8 |
| Evaluation interval | every 100 steps |
| Checkpoint interval | every 100 steps |
| Learning rate | 5e-5 |
| Weight decay | 0.01 |
| Max evaluation batches per split | 30 |
| Random seed | 42 |

### Training conditions

| Condition | Schedule |
|---|---|
| Mixed | easy / medium / hard examples mixed throughout all 900 steps |
| Curriculum | steps 1–300 easy, 301–600 medium, 601–900 hard |

## Current MVP Results

The first MVP successfully ran the intended comparison. Lower validation loss is better.

| Condition | Easy | Medium | Hard | Overall |
|---|---:|---:|---:|---:|
| Curriculum | 3.8651 | 4.1099 | **4.2101** | 4.0513 |
| Mixed | **3.6923** | **4.0939** | 4.2223 | **3.9983** |
| Curriculum - Mixed | +0.1728 | +0.0160 | -0.0122 | +0.0530 |

The result does **not** support a clear final-loss curriculum advantage. The mixed baseline achieved lower final loss on easy, medium, and overall validation. The curriculum condition achieved only a very small advantage on the hard split.

However, the curriculum schedule did show phase-specific behavior: the model improved more clearly on the validation split corresponding to the current curriculum stage. This suggests that the curriculum affected local learning dynamics, even though the local gains did not produce a better final overall model.

## How to Run

The notebook is designed for Google Colab.

### 1. Open the notebook

Use the Colab badge above, or open:

```text
notebooks/german_mlm_curriculum_mvp.ipynb
```

### 2. Use a GPU runtime

In Google Colab:

```text
Runtime -> Change runtime type -> GPU
```

The completed MVP run used a Tesla T4 GPU in Colab.

### 3. Run all cells

The notebook will:

1. configure the experiment;
2. check required packages;
3. load German4All CSV data directly from Hugging Face Hub with pandas;
4. construct easy, medium, hard, and mixed training data;
5. tokenize the data for masked language modeling;
6. train the mixed baseline and curriculum condition;
7. save logs, checkpoints, prepared data, and plots;
8. generate final validation-loss tables and learning curves.

### 4. Output directory

By default, the notebook writes outputs to local Colab storage:

```text
/content/german_mlm_curriculum_mvp_outputs
```

For safer long-running experiments, set `cfg.use_google_drive = True` in the configuration cell. Then outputs will be saved to:

```text
/content/drive/MyDrive/german_mlm_curriculum_mvp_outputs
```

## Output Files

The notebook may create the following output directories:

```text
german_mlm_curriculum_mvp_outputs/
├── checkpoints/
├── logs/
├── plots/
└── prepared_data/
```

Typical useful files include:

| Path | Purpose | Recommended for GitHub? |
|---|---|---|
| `logs/eval_log.csv` | validation loss by condition, step, phase, and split | Yes |
| `logs/train_log.csv` | training loss by condition and step | Yes |
| `logs/final_validation_loss_table.csv` | final comparison table | Yes |
| `plots/*.png` | learning curves and dataset-size plots | Yes |
| `checkpoints/` | model checkpoints and optimizer states | No |
| `prepared_data/` | derived train/validation CSV files | Usually no |

For a public portfolio repository, the safest approach is to upload only aggregate logs and plots, not model checkpoints or large prepared data files.

## Files in This Repository

| File / directory | Purpose |
|---|---|
| `README.md` | public project introduction and reproduction guide |
| `notebooks/german_mlm_curriculum_mvp.ipynb` | main Google Colab MVP notebook |
| `docs/proposal_draft_curriculum_german_mlm.pdf` | short proposal / project write-up |
| `docs/project_context_MLM.md` | extended project context and iteration plan |
| `requirements.txt` | optional local dependency list |
| `LICENSE` | repository license |

## Limitations

- The MVP uses only one dataset: German4All.
- The experiment uses only one random seed.
- The model is very small and trained for only 900 steps.
- Intermediate validation is batch-limited for speed.
- The mixed baseline and curriculum condition do not fully control for difficulty exposure ratio.
- Human readability difficulty may not match MLM difficulty for an English tokenizer.
- Strict block curriculum is likely too rigid because easier data disappear after the early phase.
- The output was originally generated in Colab local storage, so future runs should use Google Drive saving by default.

## Next Steps

Planned improvements:

1. Save all outputs to Google Drive by default.
2. Add step-0 evaluation before any optimizer update.
3. Run full final validation at the final checkpoint.
4. Compare balanced mixed training with balanced easy-to-hard curriculum training.
5. Add a hard-to-easy control condition.
6. Test a soft replay curriculum, where easier data remain available while harder data are introduced.
7. Add diagnostics for whether German4All readability levels correspond to actual MLM difficulty.
8. Run repeated seeds if compute budget allows.

## Technical Stack

- Python
- Google Colab
- PyTorch
- Hugging Face `transformers`
- Hugging Face Hub
- pandas
- NumPy
- scikit-learn
- matplotlib

## License

This project is released under the MIT License. See the `LICENSE` file for details.
