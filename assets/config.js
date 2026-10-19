/* Deployment setting. Leave empty to use the same-origin /api (all-in-one hosting).
 * For a split deploy (UI on Vercel, API on Render) set the full API URL, e.g.
 *   window.CLUTCHIQ_API_BASE = "https://clutchiq-api.onrender.com/api";
 * The page tries this first and falls back to /api, then to the in-browser engine. */
window.CLUTCHIQ_API_BASE = "https://clutchiq-api.onrender.com/api";
