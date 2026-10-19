# Clutch IQ v4 — Vercel deployment

The static UI (`index.html`, `assets/`) is served by Vercel; the FastAPI backend
runs as a Python function at `api/index.py` (which imports `backend_app.app`).
All API routes live under `/api/*`, so the UI's default backend URL `/api` works
same-origin.

```bash
vercel dev      # local: http://localhost:3000 and http://localhost:3000/api/health
vercel --prod   # production
```

Notes
- `vercel.json` sets `maxDuration: 60` for the API function (the comparison arena
  and long training runs take a few seconds).
- Uploaded datasets, tournaments and logs are kept in memory, as in v3: a cold
  start forgets them.
- The production Q-table (`q_table.json` + `q_table_meta.json`) is read-only on
  Vercel; "save as production policy" works only on a local/writable server.
- If the API is unreachable the UI falls back to the in-browser engine
  automatically (badge in the top bar).

Direct FastAPI dev (also serves the UI): `uvicorn backend_app:app --reload --port 8000`
→ http://localhost:8000/
