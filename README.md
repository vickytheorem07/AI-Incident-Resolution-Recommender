# AI Incident Resolution Recommender

A FastAPI service that retrieves similar historical support tickets from Chroma and uses a Groq chat model to recommend an incident resolution.

## Features

- Upload a CSV of support tickets to the `/load_doc` endpoint.
- Validate the required CSV columns: `ticket_id`, `issue_description`, `category`, and `resolution_notes`.
- Convert each CSV row into one LangChain document. The document ID is the row's `ticket_id`; its `category` is stored as Chroma metadata.
- Insert documents into Chroma in batches of 1,000.
- Retrieve recommendations using semantic similarity with an optional category metadata filter.
- Validate recommendation requests and responses with Pydantic models.
- Log request IDs, processing latency, and LLM input/output token counts when the provider returns usage data.

## Requirements

- Python 3.11 or newer
- [uv](https://docs.astral.sh/uv/)
- A Groq API key

## Setup

Install the locked project dependencies:

```bash
uv sync
```

Create a `.env` file in the project root. The app settings require both `groq_llm_api_key` and `LANGSMITH_API_KEY` at startup:

```dotenv
groq_llm_api_key=your-groq-api-key
LANGSMITH_API_KEY=your-langsmith-api-key

# Optional settings
MAX_UPLOAD_SIZE_MB=500
llm_model_name=openai/gpt-oss-20b
temperature=0.9
huggingface_embedding_model_name=sentence-transformers/all-MiniLM-L6-v2
persist_dir=chroma_db
collection_name=documents
top_k=3
```

`huggingface_api` is optional for the default public embedding model. The first embedding-model use may download model files.

## Run the API

```bash
uv run --with "uvicorn[standard]" uvicorn app.main:app --reload
```

The API is available at `http://127.0.0.1:8000`. Interactive API docs are at `http://127.0.0.1:8000/docs`; the health check is `GET /health`.

## Upload ticket data

Send a UTF-8 CSV as multipart form data. The file must include all four required columns. `ticket_id` values must be nonempty and unique within the upload.

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/load_doc" \
  -F "file=@train.csv"
```

The configured upload limit defaults to 500 MiB. A successful response looks like:

```json
{
  "status": "success",
  "filename": "train.csv",
  "message": "CSV tickets loaded and stored in vector db successfully.",
  "chunks": 1000
}
```

The `chunks` value is the number of ticket documents inserted. Chroma uses each `ticket_id` as the document ID, so uploading the same ticket IDs again updates those records.

## Request a recommendation

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/recommendation" \
  -H "Content-Type: application/json" \
  -d '{
    "incident_id": "INC-1042",
    "category": "Account Access",
    "issue_description": "Users cannot sign in after a password reset."
  }'
```

The request requires a nonblank `incident_id`, `category`, and `issue_description`. The response is validated and includes the incident ID:

```json
{
  "incident_id": "INC-1042",
  "response": "Confirm the password reset completed, then clear the affected users’ active sessions and have them sign in again.",
  "reason": "Similar account-access tickets were resolved by refreshing sessions after a password reset."
}
```

If no supported resolution can be found, the model is instructed to return `Solution not found`. Retrieval is filtered by the request category; historical records need the `category` metadata to use that filter, so re-upload older records created before category metadata was added.

## Logs and data

Application logs are written to the console and to `logs/app.log` by default. You can set `LOG_DIR`, `LOG_FILE`, and `LOG_LEVEL` in the environment. Logs include request IDs, timing, counts, and token usage when available; ticket descriptions and resolution text are not logged.

Chroma persists data under `chroma_db` by default. Keep the database directory available across restarts if you want to retain the indexed tickets.
