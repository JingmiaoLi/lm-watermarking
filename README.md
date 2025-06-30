# 💧 A Watermark for Large Language Models 🔍

Official implementation of the watermarking and detection algorithms presented in the papers:

- **A Watermark for Large Language Models**  
  John Kirchenbauer*, Jonas Geiping*, Yuxin Wen, Jonathan Katz, Ian Miers, Tom Goldstein
- **On the Reliability of Watermarks for Large Language Models**  
  John Kirchenbauer*, Jonas Geiping*, Yuxin Wen, Manli Shu, Khalid Saifullah, Kezhi Kong, Kasun Fernando, Aniruddha Saha, Micah Goldblum, Tom Goldstein

---

## ✨ Extensions and Additional Scripts by Jingmiao Li

This repository has been extended with additional scripts for research on reducing false positives and exploring entropy-based filtering:

- `my_watermark_detection_entropy_step1.py`  
  Compute token entropies and green masks for all input texts.
- `my_watermark_detection_entropy_step2.py`  
  Recalculate z-scores and predictions with multiple entropy thresholds.
- `my_watermark_processor2_entropy.py`  
  Detector supporting entropy filtering and bigram-level analysis.
- `my_watermark_detection_ner.py`  
  Detection using Named Entity Recognition (NER) token weighting.
- `my_watermark_detection_baseline.py`  
  Baseline detection script.
- `my_generate.py`  
  Generation script for creating watermarked text.
- Other utilities for batch processing and experimentation.

These scripts provide:
- **Entropy-based filtering**: skip low-entropy tokens when computing detection statistics.
- **NER-based weighting**: adjust contribution of named entities.
- **Two-pass detection pipelines**.
- **Batch processing** with checkpointing and resumption.

---

## ✨ Example Usage of Entropy-based Detection

**Step 1: Compute entropy and masks**
```bash
python my_watermark_detection_entropy_step1.py   --folder_path ./data   --gamma 0.25   --tokenizer meta-llama/Llama-3.1-8B   --ignore_repeated_bigrams   --max_tokens 200   --return_green_token_mask
```

**Step 2: Recalculate z-scores with multiple thresholds**
```bash
python my_watermark_detection_entropy_step2.py   --input_dir ./step1_outputs   --output_dir ./step2_outputs   --gamma 0.25   --z_threshold 4.0   --entropy_threshold 0.5 1.0 2.0   --mask_type bigram   --num_workers 4
```

---

## ✨ How to Use the Original Watermark Code

Our implementation can be added into any Hugging Face generation pipeline as an additional LogitProcessor. The main classes are:

- `WatermarkLogitsProcessor`
- `WatermarkDetector`

from `extended_watermark_processor.py`.

### Example: Generate Watermarked Text
```python
from extended_watermark_processor import WatermarkLogitsProcessor

watermark_processor = WatermarkLogitsProcessor(
    vocab=list(tokenizer.get_vocab().values()),
    gamma=0.25,
    delta=2.0,
    seeding_scheme="selfhash"
)

tokenized_input = tokenizer(input_text, return_tensors='pt').to(model.device)

output_tokens = model.generate(
    **tokenized_input,
    logits_processor=LogitsProcessorList([watermark_processor])
)

output_tokens = output_tokens[:, tokenized_input["input_ids"].shape[-1]:]
output_text = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0]
```

### Example: Detect Watermark
```python
from extended_watermark_processor import WatermarkDetector

watermark_detector = WatermarkDetector(
    vocab=list(tokenizer.get_vocab().values()),
    gamma=0.25,
    seeding_scheme="selfhash",
    device=model.device,
    tokenizer=tokenizer,
    z_threshold=4.0,
    normalizers=[],
    ignore_repeated_ngrams=True
)

score_dict = watermark_detector.detect(output_text)
```

---

## 📝 Recommended Hyperparameters
- **Gamma**: 0.25
- **Delta**: 2.0
- **Context width h**: 4
- **Seeding scheme**: `"selfhash"`
- For detection: `--ignore-repeated-ngrams=True`

---

## 📁 Repo Contents

- `watermark_processor.py`: Minimal implementation.
- `extended_watermark_processor.py`: Extended implementation (recommended).
- `homoglyphs.py`, `normalizers.py`: Helpers for text normalization and homoglyph detection.
- `demo_watermark.py`: Gradio demo and minimal working example.
- `app.py`: Quickstart Gradio app wrapper.
- `watermark_reliability_release/`: Alternate watermark variants from the robustness paper.
- **Extensions by Jingmiao Li**: see above.

---

## 🏷 License

Apache-2.0 License.

---

## ✨ Contributing

Suggestions and PRs welcome!

---

## 🔗 References

- [Original Paper: A Watermark for Large Language Models](https://arxiv.org/abs/2306.04634)
- [On the Reliability of Watermarks](https://arxiv.org/abs/2308.00113)
