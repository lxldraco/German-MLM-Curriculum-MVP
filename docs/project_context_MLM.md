# Project Context: Curriculum-Based German Continued Pretraining for a Small English MLM

## 1. Project Identity

This project is designed for the course **Neural Networks in Practice** as a 3 ECTS term project. It studies whether readability-based curriculum input can help a small English masked language model adapt to German through continued pretraining.

The original broader idea involved linguistically targeted masking strategies, including POS-specific and grammar-oriented masking. This scope has now been reduced because POS tagging and the required preprocessing pipeline would make the project too large for the available time. The current project therefore focuses on a more feasible comparison:

> **easy-to-hard German continued pretraining vs. mixed-difficulty German continued pretraining**.

The project does not aim to train a large model from scratch or produce a state-of-the-art German model. It uses continued pretraining of an existing small English encoder model on German text, making the experiment feasible in Google Colab with free GPU resources.

## 2. Background and Motivation

Masked language modeling (MLM), as used in BERT-style models, trains a model to predict masked tokens from context. Standard MLM usually uses randomly mixed corpus data, but curriculum learning suggests that the order or difficulty of training examples may affect learning efficiency.

This project applies curriculum learning at the **input-data level** rather than at the token-masking level. Instead of asking which tokens should be masked, it asks whether German text should be presented in a readability-based order. The second-language-learning motivation is loose but useful: human learners are usually not exposed to random language input, but to material organized from easier to harder levels.

The project also relates to small-model training and continued pretraining. Work such as TinyStories suggests that small models can learn meaningful language patterns when the input distribution is carefully controlled. Domain-adaptive and continued pretraining also motivate adapting an existing model to a new language or domain rather than training from scratch.

## 3. Research Question and Hypotheses

### 3.1 Main Research Question

> Does easy-to-hard continued pretraining help a small English masked language model adapt to German more effectively than mixed-difficulty continued pretraining?

### 3.2 Current Hypothesis Set

The current proposal uses three working hypotheses. To avoid confusion, **H1 is the original primary directional hypothesis**, while **H2 and H3 are secondary working hypotheses added after the first MVP result to make the analysis more explicit**.

#### H1: Final-loss curriculum advantage

Under the same training budget, an easy-to-hard curriculum will produce lower MLM validation loss than mixed-difficulty training, especially on medium and hard German validation data.

This is the main hypothesis inherited from the original project idea. If supported, it would suggest that readability-based curriculum order improves German adaptation under a fixed compute budget.

#### H2: Phase-specific curriculum effect

Even if the final overall loss is not better, the curriculum condition may show stronger local improvement on the validation split corresponding to the current training phase. For example, during the easy phase, easy validation loss may decrease faster; during the medium phase, medium validation loss may improve more; and during the hard phase, hard validation loss may improve more.

This hypothesis separates **local learning dynamics** from **final model quality**. It allows the project to report whether the curriculum schedule affects training behavior even if it does not win on final validation loss.

#### H3: Mixed exposure as a strong alternative

Mixed training may perform similarly or better because it exposes the model to the full German difficulty distribution from the beginning. This broad exposure may help the model learn basic German lexical and syntactic patterns while also adapting early to harder examples.

This hypothesis became important after MVP v1.0, where mixed training outperformed the strict easy-to-hard curriculum on final overall validation loss. It should be treated as a competing explanation rather than as a replacement for H1.

### 3.3 Hypothesis Status

The current project should present the hypotheses transparently:

- **Primary hypothesis:** H1, the original curriculum advantage hypothesis.
- **Secondary diagnostic hypothesis:** H2, added to analyze phase-level learning behavior.
- **Competing hypothesis:** H3, added after the MVP as a plausible explanation for the observed result.

Future reports should clearly distinguish between hypotheses defined before an experiment and interpretations introduced after observing results.

## 4. Dataset and Difficulty Construction

The project uses **German4All**, a German readability-controlled paraphrasing dataset with aligned paragraph-level paraphrases across five readability levels. German4All is suitable for this project because it directly supports construction of easy, medium, and hard German input without building a new annotated corpus.

For the MVP, the five German4All levels were collapsed into three groups:

| Group | Role in curriculum |
|---|---|
| Easy | first curriculum stage |
| Medium | second curriculum stage |
| Hard | final curriculum stage |

In MVP v1.0, the expanded dataset contained 114,570 training examples and 12,725 validation examples:

| Group | Training examples | Validation examples |
|---|---:|---:|
| Easy | 45,828 | 5,090 |
| Medium | 22,914 | 2,545 |
| Hard | 45,828 | 5,090 |
| Total | 114,570 | 12,725 |

The main advantage of German4All is feasibility. It provides difficulty labels and enough data for a small continued-pretraining experiment. Its main limitation is that human readability difficulty may not equal MLM difficulty for an English model and tokenizer. German capitalization, compounds, umlauts, morphology, sentence length, and WordPiece fragmentation may dominate the loss pattern. In future work, a purely human-authored CEFR-style dataset may be used for a more linguistically natural study, but German4All remains the only dataset in the current MVP to keep the scope manageable.

## 5. Model and Training Setup

MVP v1.0 used `prajjwal1/bert-tiny` with the `bert-base-uncased` tokenizer. This was chosen as a fast pilot model suitable for free Google Colab GPU sessions. The English uncased tokenizer is part of the cross-lingual adaptation setting, but it also creates a German tokenizer mismatch, especially for capitalization, compounds, umlauts, and morphology.

The objective is standard masked language modeling. Given an input sequence \(x=(x_1,\ldots,x_n)\) and a set of masked positions \(M\), the model minimizes:

\[
\mathcal{L}_{\mathrm{MLM}}(\theta) = - \frac{1}{|M|} \sum_{i \in M} \log p_{\theta}(x_i \mid x_{\setminus M}).
\]

The MVP used a fixed training budget of 900 optimization steps, batch size 8, maximum sequence length 128, MLM probability 0.15, and evaluation every 100 steps. Each validation split was evaluated with a maximum of 30 batches during the pilot run.

The two MVP conditions were:

1. **Mixed baseline:** easy, medium, and hard examples mixed throughout training.
2. **Strict easy-to-hard curriculum:** steps 1–300 easy, steps 301–600 medium, steps 601–900 hard.

The main evaluation metric was MLM validation loss on easy, medium, hard, and overall validation splits.

## 6. MVP v1.0 Result

MVP v1.0 successfully ran, but the result did not support H1. Final validation losses were:

| Condition | Easy | Medium | Hard | Overall |
|---|---:|---:|---:|---:|
| Curriculum | 3.8651 | 4.1099 | **4.2101** | 4.0513 |
| Mixed | **3.6923** | **4.0939** | 4.2223 | **3.9983** |
| Curriculum - Mixed | +0.1728 | +0.0160 | -0.0122 | +0.0530 |

The mixed baseline achieved lower final loss on easy, medium, and overall validation. The curriculum model achieved only a very small advantage on the hard validation split. Therefore, the MVP result should be treated as a preliminary negative or mixed result rather than as evidence for a curriculum advantage.

However, the curriculum did show phase-specific behavior consistent with H2. During the easy phase, easy validation loss improved more clearly. During the medium and hard phases, the corresponding validation splits also showed phase-related improvements. This means the curriculum schedule affected learning locally, even though these local gains did not produce a better final overall model.

## 7. Interpretation of the MVP Result

The strongest current interpretation is that strict easy-to-hard block training is not automatically better for German MLM adaptation. The mixed baseline may be strong because it exposes the model to the full difficulty distribution from the beginning. This supports H3 as a serious competing explanation.

Several limitations affect the interpretation:

- The curriculum schedule was too strict: easy-only, then medium-only, then hard-only.
- The comparison did not fully isolate order because mixed training followed the natural 40/20/40 data ratio, while curriculum used equal one-third stage lengths.
- German4All readability difficulty may not equal MLM difficulty under an English WordPiece tokenizer.
- The pilot used only one seed, 900 steps, a very small model, and limited validation batches.
- Training results were saved only in local Colab storage, not to Google Drive, so future runs must improve persistence.

## 8. Iteration Plan

The next iteration should focus on experimental control and reproducibility rather than immediately scaling the model.

First, the notebook must save all outputs to Google Drive by default, including logs, checkpoints, plots, final tables, configuration files, and prepared data. Colab local storage is not reliable enough for reportable experiments.

Second, the next version should add true step-0 evaluation before any optimizer update and should run full final validation, even if intermediate evaluations remain batch-limited.

Third, the design should control the exposure-ratio confound. A balanced mixed baseline should sample easy, medium, and hard examples with equal probability, and it should be compared to a balanced easy-to-hard curriculum with the same total exposure.

Fourth, at least one order-control condition should be added, preferably hard-to-easy. This will test whether the direction of the curriculum matters.

Fifth, a soft or replay curriculum should be tested. A possible schedule is 80/15/5 easy-medium-hard in phase 1, 40/40/20 in phase 2, and 20/30/50 in phase 3. This is more realistic than a strict block curriculum because easier material remains available while harder material is introduced.

Finally, a diagnostic analysis should check whether German4All readability levels correspond to actual MLM difficulty. Useful diagnostics include initial MLM loss by readability level, WordPiece fragmentation, sentence length, and frequent subword patterns by level.

## 9. Expected Contribution

Even if H1 remains unsupported, the project is still valuable. It provides a controlled small-scale test of a pedagogically plausible idea and shows that human-readable difficulty does not necessarily translate into better MLM training order. A reportable contribution may therefore be:

> For a small English MLM adapted to German with German4All, strict easy-to-hard continued pretraining did not outperform mixed-difficulty training. Mixed exposure may be stronger because it provides broader distributional coverage from the beginning. Future work should test balanced exposure, replay curricula, and model-based difficulty measures.

A positive result in later iterations would require stronger evidence: full final validation, controlled exposure ratios, repeated seeds if possible, and comparison with hard-to-easy or replay-based schedules.

## 10. Workflow Note for Future Drafts

Hypotheses should not be silently rewritten when moving from project context to proposal draft. Future writing should follow this rule:

1. Keep the **original hypothesis** clearly marked.
2. Mark any new hypothesis as **added**, **secondary**, **diagnostic**, or **post-MVP interpretation**.
3. Do not present a post-hoc explanation as if it had been part of the original design.
4. When a proposal needs a stronger hypothesis structure, first update `project_context.md`, then use that file as the source for the proposal.

This prevents hidden scope drift and keeps the project history auditable.

## References

- Anschütz, M. et al. (2025). *German4All: A Dataset and Model for Readability-Controlled Paraphrasing in German*.
- Devlin, J. et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*.
- Eldan, R., & Li, Y. (2023). *TinyStories: How Small Can Language Models Be and Still Speak Coherent English?*
- Gururangan, S. et al. (2020). *Don’t Stop Pretraining: Adapt Language Models to Domains and Tasks*.
- Lee, M. et al. (2022). *Efficient Pre-training of Masked Language Model via Concept-based Curriculum Masking*.
