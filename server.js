import express from 'express';
import cors from 'cors';
import fetch from 'node-fetch';
import dotenv from 'dotenv';
import path from 'path';
import { fileURLToPath } from 'url';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
app.use(cors());
app.use(express.static(path.join(__dirname, 'public')));

const GITHUB_REPO = process.env.GITHUB_REPO || 'codedspaces/demo-2';
const POLL_MS = Number(process.env.POLL_MS || 15000);

let lastStarCount = null;
let lastStarsUrl = `https://github.com/${GITHUB_REPO}/stargazers`;
let lastRepoUrl = `https://github.com/${GITHUB_REPO}`;
let sseClients = new Set();

async function fetchStars() {
  const url = `https://api.github.com/repos/${GITHUB_REPO}`;
  const headers = { 'User-Agent': 'live-star-tracker' };
  const token = process.env.GITHUB_TOKEN;
  if (token) headers['Authorization'] = `Bearer ${token}`;
  const res = await fetch(url, { headers });
  if (!res.ok) throw new Error(`GitHub API error: ${res.status}`);
  const data = await res.json();
  return {
    stargazers_count: data.stargazers_count,
    html_url: data.html_url,
    name: data.full_name,
  };
}

function broadcast(data) {
  const payload = `data: ${JSON.stringify(data)}\n\n`;
  for (const res of sseClients) {
    try { res.write(payload); } catch { /* ignore */ }
  }
}

app.get('/events', (req, res) => {
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  res.flushHeaders?.();

  sseClients.add(res);

  // Send initial state
  if (lastStarCount !== null) {
    res.write(`data: ${JSON.stringify({
      stars: lastStarCount,
      starsUrl: lastStarsUrl,
      repoUrl: lastRepoUrl,
      repo: GITHUB_REPO,
    })}\n\n`);
  }

  req.on('close', () => {
    sseClients.delete(res);
  });
});

app.get('/api/stars', async (req, res) => {
  try {
    const data = await fetchStars();
    const response = {
      stars: data.stargazers_count,
      starsUrl: `${data.html_url}/stargazers`,
      repoUrl: data.html_url,
      repo: data.name,
    };
    res.json(response);
  } catch (e) {
    res.status(500).json({ error: String(e) });
  }
});

// Poll GitHub and broadcast via SSE
async function pollLoop() {
  try {
    const data = await fetchStars();
    const stars = data.stargazers_count;
    lastStarsUrl = `${data.html_url}/stargazers`;
    lastRepoUrl = data.html_url;
    if (lastStarCount === null || stars !== lastStarCount) {
      lastStarCount = stars;
      broadcast({
        stars,
        starsUrl: lastStarsUrl,
        repoUrl: lastRepoUrl,
        repo: data.name,
      });
    }
  } catch (e) {
    // optionally log
  } finally {
    setTimeout(pollLoop, POLL_MS);
  }
}

pollLoop();

app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Server listening on http://localhost:${PORT}`);
  console.log(`Tracking repo: ${GITHUB_REPO}`);
});