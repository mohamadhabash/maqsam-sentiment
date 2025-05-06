# Maqsam Sentiment Analysis Service

> **Part 1 Submission**: A clean, stateless REST API for inferring sentiment (Positive / Neutral / Negative) from customer-service call summaries in both English and Arabic, leveraging open-source, self-hosted LLMs.

---

## 🎯 Assignment Objectives
1. **Stateless, Synchronous REST API**  
   - Each API endpoint executes synchronously, handling each input in a single operation without preserving any session or context between requests.
2. **Self-Hosted LLM Inference**  
   - No external LLM APIs; uses Hugging Face transformers to load and run open-source models locally.  
3. **Clean, Modular Code**  
   - Well-organized into `app/`, `config/`, `scripts/`, and `tests/` for maintainability and readability.

---

## 🛠️ Implementation Details

### Technology Stack & Frameworks
- **FastAPI** for the REST interface (synchronous endpoints)  
- **Uvicorn** as the ASGI server  
- **PyTorch** + **bitsandbytes** for LLM inference (supports 4-bit quantization)  
- **PyYAML** for configuration management  
- **PyTest** & **TestClient** for unit and integration tests

### Code Structure
```
maqsam_sentiment/
├── config/                  # Configuration files
│   └── config.yaml          # Model, prompt, and task settings
├── app/                     # Core application logic (Python package)
│   ├── __init__.py
│   ├── api.py               # FastAPI router
│   ├── main.py              # Uvicorn launcher
│   ├── model_loader.py      # Reads config & loads tokenizer/model
│   ├── predictor.py         # Inference logic
│   └── utils.py             # Shared utilities
├── scripts/                 # CLI scripts
│   ├── benchmark_inference.py
│   └── evaluate_sentiment.py
├── tests/                   # Test suite
│   ├── test_api.py
│   ├── test_predictor.py
│   └── test_benchmark.py
├── data/                    # Arabic & English customer-service call summaries test-sets
│   ├── test_set_en.csv
│   └── test_set_ar.csv
├── requirements.txt         # Python dependencies
├── Dockerfile               # Container build recipe
├── run.sh                   # Quick API start script
└── README.md                # Project documentation
```

---

## 🖥️ Hardware & Environment Assumptions
- **GPU**: We assume access to an NVIDIA T4 (16 GB) on Google Colab.  
  - **Rationale**: Colab provides free T4 instances. All model choices and quantization strategies optimize to fit within 16 GB.  
- **CPU Fallback**: The API can also run on CPU only, with reduced throughput.  
- **Docker**: For production-like isolation and reproducibility.

---

## 📊 Model Comparison & Selection

| Model Name                      | Architecture | #Params | Precision            | GPU RAM (weights only) | Peak GPU RAM (generation) | Inference Time | Accuracy (Ar / En) | F1-Score (Ar / En) |
|---------------------------------|--------------|--------:|----------------------|------------------------|---------------------------|---------------:|--------------------:|-------------------:|
| multilingual-sentiment-analysis | BERT-Based   |  134 M  | FP32                 | 0.504 GB               | 0.535 GB                  |    12.5 ms     |    0.38 / 0.52      |   0.46 / 0.58      |
| jais-family-590m-chat           | GPT3-Based   |  590 M  | FP32                 | 2.482 GB               | 2.650 GB                  |   178.6 ms     |    0.65 / 0.68      |   0.54 / 0.57      |
| jais-family-1p3b-chat           | GPT3-Based   | 1.3 B   | FP32                 | 5.259 GB               | 5.503 GB                  |   292.4 ms     |    0.43 / 0.45      |   0.33 / 0.37      |
| jais-family-2p7b-chat           | GPT3-Based   | 2.7 B   | FP32                 | 10.382 GB              | 10.797 GB                 |   549.2 ms     |    0.85 / 0.85      |   0.84 / 0.85      |
| **jais-family-6p7b-chat**       | GPT3-Based   | 6.7 B   | **INT4 (4-bit quant)** | **7.541 GB**           | **8.134 GB**              |  1664.3 ms     | **0.95 / 0.98**     | **0.94 / 0.98**    |

**Test Set**: I curated **120 customer-service call summaries** (60 English, 60 Arabic), and balanced equally across the three sentiment labels (Positive, Neutral, Negative). Summaries **mimic real-world customer interactions**, covering various problem resolutions, factual inquiries, and complaints, ensuring diverse vocabulary and structure for robust evaluation.

The JAIS-Family models—developed by InceptionAI—are instruction-tuned, bilingual (English & Arabic) causal language models available in multiple sizes (590 M, 1.3 B, 2.7 B, 6.7 B, ..., 30 B parameters). They deliver strong multilingual reasoning performance across benchmarks like MMLU, EXAMS, and situational QA, and support.

**Justification for Model Selection**: On a T4 GPU with 16 GB memory (as available in Google Colab), the 6.7 B parameter JAIS model in 4-bit quantization (jais-family-6p7b-chat) provides an optimal balance between model capacity and resource constraints. Full-precision (FP32/FP16) variants of this size require ~20–22 GB of RAM, exceeding T4 limits—whereas 4-bit quantization compresses the model weights to ~7.5 GB and requires only ~8.1 GB peak during inference. This configuration achieves the highest accuracy (95% Arabic, 98% English) and F1-scores (0.94/0.98) on our balanced 120-summary test set, making it the ideal choice for high-quality, resource-efficient sentiment analysis.

Additionally, the lightweight BERT-based multilingual-sentiment-analysis model can be fine-tuned on domain-specific call summaries to achieve robust performance with minimal resource overhead, and smaller JAIS variants (e.g., 590 M or 1.3 B) can be adapted via LoRA (Low-Rank Adaptation) to produce compact, high-accuracy sentiment models that fit within tighter GPU memory.


---

## ⚙️ Configuration (`config/config.yaml`)

```yaml
model:
  name: inceptionai/jais-family-6p7b-chat
  device: auto
  temperature: 0.3
  top_p: 0.9
  repetition_penalty: 1.2
  max_new_tokens: 20
  min_length_buffer: 4
  load_in_4bit: true

prompt:
  base: |-
    ### Instruction: Your name is 'Jais' ...
    ### Input: [|Human|] {Question}
    ### Response :
task:
  sentiment: |-
    ### Task
    Classify the following customer-service summary into **one** of: Positive, Neutral, or Negative.

    ### Labels
    - **Positive**: expressions of gratitude, praise, or clear problem resolution.
    - **Neutral**: strictly factual or informational statements with **no** emotional or evaluative content.
    - **Negative**: complaints, expressions of frustration, or unresolved issues.

    ### Examples
    1. Text: "I’m grateful for the quick support and resolution provided today."  
      Sentiment: Positive  
    2. Text: "I checked my account balance for today and noted the amount available."  
      Sentiment: Neutral  
    3. Text: "The agent confirmed my delivery date and provided a tracking number."  
      Sentiment: Neutral  
    4. Text: "I waited on hold for an hour and my issue remains unresolved."  
      Sentiment: Negative  
    5. Text: "The staff handled my request efficiently and ensured everything was updated."  
      Sentiment: Positive  

    ### Reasoning
    Please think step by step about whether this text contains **emotional content** or is **purely factual**, then answer with **only** the sentiment label.
```

---

## 🚀 Running the Service

### Local Execution
```bash
pip install -r requirements.txt
huggingface-cli login --token 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
bash run.sh
# or
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Docker
```bash
docker build -t maqsam-sentiment .
docker run --rm -e HUGGINGFACE_TOKEN=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 -p 8000:8000 maqsam-sentiment
```

### Google Colab + ngrok
```python
!pip install pyngrok
from pyngrok import ngrok
!ngrok config add-authtoken 2rcc49Z2DSeCRvkAiadZwjbXHcE_3A8bqYvg8UiwR5MtxXCY8
!huggingface-cli login --token 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
tunnel = ngrok.connect(8000, "http")
print("Public URL:", tunnel.public_url)
get_ipython().system_raw('PYTHONPATH="$PWD" uvicorn app.main:app --host 0.0.0.0 --port 8000 &')
```

---

## 📞 Example API Calls

#### cURL
```bash
curl -X POST http://localhost:8000/api/predict      -H "Content-Type: application/json"      -d '{"summary":"I’m frustrated that my order arrived late."}'
# => {"sentiment":"Negative"}
```

#### Python `requests`
```python
import requests

url = "http://localhost:8000/api/predict"
payload = {"summary": "Thank you for the quick resolution!"}
resp = requests.post(url, json=payload)
print(resp.json())  # {'sentiment':'Positive'}
```

---

## 📈 Benchmarking & Evaluation

- **Benchmark**: `python scripts/benchmark_inference.py`
  Measures GPU RAM (weights & peak) and latency on a sample prompt.
- **Evaluation**: `python scripts/evaluate_sentiment.py`
  Computes Accuracy & F1-Score on the balanced English set (120 samples).

---

## 🔮 Future Work & Fine-Tuning

- **LoRA Adaptation**: Efficiently fine-tune smaller JAIS models (590 M / 1.3 B) for competitive accuracy in ~2–3 GB of GPU RAM.
- **BERT Fine-Tuning**: Standard fine-tuning of the multilingual BERT baseline for sub-50 ms CPU inference in edge environments.
- **Part 2 & 3**: Design docs for scaling, system architecture, and summarization pipelines to follow.

---

*Prepared by Mohammad for the Maqsam Senior ML Engineer assignment*
