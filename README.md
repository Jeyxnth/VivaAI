# VivaAI — LLM Module

Live demo of the question generation + answer evaluation pipeline using **Llama 3 via Ollama**.

## Setup (one-time)

### 1. Install Ollama
```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows → download from https://ollama.com/download
```

### 2. Pull Llama 3
```bash
ollama pull llama3
```

### 3. Install Python dependencies
```bash
pip install fastapi uvicorn httpx python-multipart
```

## Run

Open **two terminals** in this folder:

**Terminal 1 — Ollama**
```bash
ollama serve
```

**Terminal 2 — VivaAI server**
```bash
uvicorn server:app --reload --port 8000
```

Then open `http://localhost:8000` in your browser.

---

## What you'll see

| Panel | Description |
|-------|-------------|
| **Left** | Module config — edit the system prompt, question count, time per question. Quick presets for 3 modules. Token stream shows raw LLM output token-by-token. |
| **Center** | Live viva session — AI-generated question streams in character-by-character. Timer counts down. Submit a text answer to get AI evaluation. |
| **Right** | Session history — every Q with score and justification. |

## How it works

```
System prompt + history
        ↓
  POST /generate-question/stream   →  SSE token stream  →  question bubble
        ↓
  Student answers
        ↓
  POST /evaluate-answer            →  JSON {score, justification, strengths, gaps}
        ↓
  Saved to history  →  next question generated with full context
```

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET  | `/health` | Check Ollama connectivity + model status |
| GET  | `/default-modules` | Return 3 sample module configs |
| POST | `/generate-question/stream` | Stream next question (SSE) |
| POST | `/evaluate-answer` | Evaluate answer, return JSON score |

## Swapping text for audio

When you're ready to wire in Whisper STT, replace the `<textarea>` answer input in `index.html` with an audio recorder, then POST the `.webm` file to a `/transcribe` endpoint (see `server.py` comments). The `submitAnswer()` function accepts any string — just replace the source.

## File structure

```
vivaai-llm/
├── server.py      # FastAPI backend — all LLM logic here
├── index.html     # Frontend UI — served by FastAPI at /
└── README.md
```
