# Free Deployment Guide

The app is split into two pieces that deploy independently:

- **`backend/`** — the FastAPI service with the AI models (FinBERT, BiLSTM,
  ChromaDB, LangGraph agent). This is the heavy piece (~4-5GB Docker image
  once torch/transformers/chromadb are installed), so it needs a host with
  generous free container limits → **Hugging Face Spaces (Docker SDK)**,
  which gives 16GB RAM on the free CPU tier.
- **`frontend/`** — plain static HTML/CSS/JS. Deploys instantly to any free
  static host → **Vercel** or **Netlify**.

Models are loaded lazily (only on first actual use, not at startup) and
results are cached to disk (`backend/cache_utils.py`), so the backend stays
responsive even on a free, shared CPU instance.

---

## 1. Deploy the backend to Hugging Face Spaces

1. Go to https://huggingface.co/new-space
2. Name it, choose **Docker** as the Space SDK, pick the free CPU tier, create it.
3. Clone the new Space's git repo locally, then copy everything from this
   project's `backend/` folder into the root of that Space repo (the
   `Dockerfile` must sit at the Space repo's root):
   ```bash
   git clone https://huggingface.co/spaces/<your-username>/<space-name>
   cp -r backend/* <space-name>/
   cd <space-name>
   ```
4. In the Space's **Settings → Variables and secrets**, add:
   - `GROQ_API_KEY`
   - `NEWS_API_KEY`
   (Never commit these — the repo's `.gitignore` already excludes `backend/.env`.)
5. Commit and push:
   ```bash
   git add .
   git commit -m "Deploy backend"
   git push
   ```
6. The Space will build the Docker image (first build takes a while — torch +
   transformers + chromadb). Once it's live, your API is at:
   `https://<your-username>-<space-name>.hf.space`
7. Verify it: open `https://<your-username>-<space-name>.hf.space/docs` —
   you should see the FastAPI Swagger UI.

Notes:
- HF Spaces' free tier can sleep after inactivity; the first request after
  a sleep will be slow (cold start + lazy model load). Subsequent requests
  use the disk cache in `backend/.cache/` and `backend/chroma_db/` as long
  as the Space container stays warm.
- If you'd rather not copy files into a second git repo, you can instead
  add the HF Space as a second git remote for this repo and push the
  `backend/` subtree — either approach works.

---

## 2. Deploy the frontend to Vercel (or Netlify)

1. Edit `frontend/config.js` and replace the placeholder with your real
   backend URL from step 1:
   ```js
   : "https://<your-username>-<space-name>.hf.space";
   ```
2. **Vercel:**
   - https://vercel.com/new → import this GitHub repo
   - Set **Root Directory** to `frontend`
   - Framework preset: "Other" (no build step needed) → Deploy
3. **Netlify (alternative):**
   - https://app.netlify.com/start → import this repo
   - Base directory: `frontend`, Build command: (none), Publish directory: `frontend`
4. Once deployed, open the Vercel/Netlify URL — the landing page should load,
   and searching a ticker should successfully call your Hugging Face backend
   (check the browser's Network tab if something doesn't load).

---

## 3. Quick local smoke test before deploying

```bash
# Terminal 1 — backend
cd backend
pip install -r requirements.txt
python main.py                 # http://localhost:8000

# Terminal 2 — frontend
cd frontend
python -m http.server 5500     # http://localhost:5500/index.html
```

`frontend/config.js` auto-points `localhost` at `http://localhost:8000`, so
this mirrors the split-deployment setup before you push anywhere.
