# Lingoflow API - Fast & Accurate Multilingual Translation Service

A lightweight **FastAPI**-based translation service powered by **Groq** and **Llama 3.3 70B**, designed for real-time, accurate, and format-preserving language translation.

---

## Features

- **Auto Language Detection** – No need to specify source language.
- **ISO-639-1 Standard Codes** – Full support for `en`, `es`, `fr`, `zh`, `ja`, `ar`, `hi`, `pt`, `ru`, `ko`, etc.
- **Preserves Formatting** – Line breaks, bullet points, code blocks, emojis, and punctuation are kept intact.
- **JSON-Only Output** – Clean, predictable responses with no extra text.
- **Same-Language Handling** – Returns original text if source = target.
- **Fallback Logic** – Defaults to English if detection fails.

---

## API Endpoint

### `POST /translate`

Translates input text to the specified target language.

#### Request Body (JSON)

```
curl -X POST http://127.0.0.1:8000/a2a/agent/translate \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": "trans-001",
    "method": "message/send",
    "params": {
      "message": {
        "kind": "message",
        "role": "user",
        "parts": [
          { "kind": "text", "text": "Life is beautiful to italian" }
        ],
        "messageId": "msg-001",
        "taskId": "task-001"
      },
      "configuration": { "blocking": true }
    }
  }'
```
#### Response(JSON)

```
{
  "jsonrpc": "2.0",
  "id": "trans-001",
  "result": {
    "id": "008567bd-bc6f-4633-b55b-9d156bf7e4e0",
    "contextId": "aa10970b-9a4e-4842-b5ae-50581ebda9f8",
    "status": {
      "state": "completed",
      "timestamp": "2025-11-02T21:31:13.612252",
      "message": {
        "messageId": "81126502-b0e4-494a-96ce-7c41ab7cca18",
        "role": "agent",
        "parts": [
          {
            "kind": "text",
            "text": "la vita è bella"
          }
        ],
        "kind": "message",
        "taskId": "008567bd-bc6f-4633-b55b-9d156bf7e4e0"
      }
    },
    "artifacts": [
      {
        "artifactId": "acac455c-ee68-4e83-9bb3-ecbea2fc969b",
        "name": "translation",
        "parts": [
          {
            "kind": "text",
            "text": "la vita è bella"
          }
        ]
      }
    ],
    "history": [
      {
        "messageId": "8cfac237-b570-4fd9-8119-da58e0fbffaf",
        "role": "agent",
        "parts": [
          {
            "kind": "text",
            "text": "la vita è bella"
          }
        ],
        "kind": "message",
        "taskId": "008567bd-bc6f-4633-b55b-9d156bf7e4e0"
      }
    ],
    "kind": "task"
  }
}
```

## How to Run Locally
1. Clone the repository:
   ```bash
   git clone hhttps://github.com/iganya/hng13-task-0.git
   cd your-repo
   ```
2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
3. Install dependencies:
   ```bash
   pip install -e .
   ```
4. Set up .env file 
   ```GROQ_API_KEY=your_groq_api_key_here```
   Get your API key from https://console.groq.com
5. Run the FastAPI application:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```
6. Access the API at `http://127.0.0.1:8000/`

