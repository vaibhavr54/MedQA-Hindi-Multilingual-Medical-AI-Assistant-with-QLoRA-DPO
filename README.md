---
title: MedQA Hindi
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---


![CI](https://github.com/vaibhavr54/MedQA-Hindi-Multilingual-Medical-AI-Assistant-with-QLoRA-DPO/actions/workflows/ci.yml/badge.svg)

BERTScore | 0.71 | 0.82 | 0.87 | ↑22% |
| Medical Accuracy | 45% | 72% | 78% | ↑73% |

### Evaluation Methodology

- **ROUGE-L**: Measures longest common subsequence between generated and reference answers
- **BERTScore**: Semantic similarity using multilingual BERT embeddings
- **Medical Accuracy**: Keyword overlap on medical entities (medications, symptoms, body parts, treatments)

---

## 📁 Project Structure

```
medqa-hindi/
├── data/
│   ├── prepare_dataset.py      # Dataset download & formatting
│   ├── translation.py          # IndicTrans2 EN→HI translation
│   ├── create_dpo_pairs.py     # Self-play DPO pair generation
│   └── processed/              # Generated datasets (gitignored)
├── training/
│   ├── qlora_finetune.py       # QLoRA training script
│   ├── dpo_align.py            # DPO alignment script
│   └── configs/
│       ├── qlora.yaml          # QLoRA hyperparameters
│       └── dpo.yaml            # DPO hyperparameters
├── evaluation/
│   ├── evaluate.py             # Metrics computation
│   ├── compare_models.py       # Report generation
│   └── results/                # Evaluation outputs (gitignored)
├── api/
│   ├── main.py                 # FastAPI application
│   ├── inference.py            # Model loading & generation
│   └── models.py               # Pydantic schemas
├── frontend/
│   ├── index.html              # Single-page application
│   ├── styles.css              # Dark medical theme
│   ├── app.js                  # Frontend logic
│   └── charts.js               # Chart.js visualizations
├── requirements.txt
└── README.md
```

---

## 💻 Hardware Requirements

### Training (Kaggle T4)

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| GPU | NVIDIA T4 (16GB) | V100/A100 |
| RAM | 16GB | 32GB |
| Disk | 50GB | 100GB |
| Time | ~5 hours total | ~3 hours |

### Inference

| Resource | Requirement |
|----------|-------------|
| GPU | 4GB+ VRAM (T4 works) |
| RAM | 8GB |
| Disk | 10GB (models) |

---

## ⚠️ License & Disclaimer

### Medical Disclaimer

**This system is for educational purposes only.** It is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.

**यह जानकारी केवल शैक्षिक उद्देश्यों के लिए है।** यह पेशेवर चिकित्सा सलाह का विकल्प नहीं है। किसी भी स्वास्थ्य संबंधी निर्णय से पहले अपने चिकित्सक से परामर्श करें।

### License

MIT License - See [LICENSE](LICENSE) file for details.

The model weights and datasets may have their own licenses:
- **Qwen2.5**: Tongyi Qianwen License
- **MedQuAD**: Public Domain
- **MedMCQA**: Apache 2.0
- **IndicTrans2**: MIT

---

## 🙏 Acknowledgments

- [Qwen Team](https://github.com/QwenLM/Qwen) for the base model
- [HuggingFace](https://huggingface.co/) for datasets and transformers
- [AI4Bharat](https://ai4bharat.iitm.ac.in/) for IndicTrans2
- [Kaggle](https://www.kaggle.com/) for free GPU access

---

## 📬 Contact

For questions or contributions, please open an issue on GitHub.

**Built with ❤️ for better healthcare AI in Indian languages.**
