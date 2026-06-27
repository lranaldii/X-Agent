# X-Agent

Oversight Reasoning framework that audits LLM conversations, runs an
auditor/expert debate, and corrects sycophantic answers. Supports OpenAI and
Google Gemini backends.

## Usage

OpenAI:
```bash
pip install -e ".[openai]"
export OPENAI_API_KEY=sk-...
python examples/demo_openai.py
```

Google Gemini:
```bash
pip install -e ".[google]"
export GOOGLE_API_KEY=AQ.A...
python examples/demo_openai.py --backend google
```
