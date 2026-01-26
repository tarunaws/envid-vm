const baseUrl = process.env.LT_BASE_URL || 'http://localhost:8080';
const token = (process.env.LT_AUTH_TOKEN || '').trim();

const payload = { q: 'Hello', source: 'en', target: 'hi' };
const headers = { 'Content-Type': 'application/json' };
if (token) headers.Authorization = `Bearer ${token}`;

fetch(`${baseUrl}/translate`, {
  method: 'POST',
  headers,
  body: JSON.stringify(payload),
})
  .then((res) => res.text())
  .then((text) => console.log(text))
  .catch((err) => console.error(err));
