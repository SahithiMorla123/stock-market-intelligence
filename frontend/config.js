// Backend API URL. Loaded before script.js.
//
// Local dev (frontend served separately from backend): defaults to localhost:8000.
// Deployed: set this to your deployed backend's URL
// (e.g. a Hugging Face Space: "https://<your-username>-<space-name>.hf.space").
window.__API_BASE__ = (window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1")
  ? "http://localhost:8000"
  : "https://REPLACE-WITH-YOUR-BACKEND-URL.hf.space";
