# ml_server

<!-- code_chunk_output -->

- [KoBERT with Huggingface](#kobert-with-huggingface)
- [Requirements](#requirements)
- [How to install](#how-to-install)
- [Reference](#reference)

<!-- /code_chunk_output -->

---

## KoBERT with Huggingface

Huggingface 기반 KoBERT를 사용하여 텍스트 감정 분석을 진행하였습니다.

## Requirements

- Python >= 3.6
- PyTorch >= 1.8.1
- transformers >= 4.8.2
- sentencepiece >= 0.1.91

## How to install

### in colab

```python
!pip install "git+https://github.com/SKTBrain/KoBERT.git#egg=kobert_tokenizer&subdirectory=kobert_hf"
```

### in vscode

```sh
python -m venv .venv
pip install --upgrade pip
pip install "git+https://github.com/SKTBrain/KoBERT.git#egg=kobert_tokenizer&subdirectory=kobert_hf"
pip install -r requirements.txt
```

---

## Reference

- [KoBERT](https://github.com/SKTBrain/KoBERT)
- [Huggingface Transformers](https://github.com/huggingface/transformers)
