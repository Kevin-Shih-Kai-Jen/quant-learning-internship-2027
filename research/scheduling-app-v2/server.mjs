import http from 'node:http';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
const port = Number(process.env.PORT || 4173);
const allowed = new Set(['index.html', 'styles.css', 'app.mjs', 'domain.mjs', 'storage.mjs', 'solver.mjs', 'solver-worker.mjs', 'manifest.webmanifest', 'icon.svg', 'sw.js']);
const types = { html: 'text/html; charset=utf-8', css: 'text/css; charset=utf-8', mjs: 'text/javascript; charset=utf-8', js: 'text/javascript; charset=utf-8', svg: 'image/svg+xml', webmanifest: 'application/manifest+json' };
http.createServer(async (req, res) => {
  const path = new URL(req.url, 'http://localhost').pathname;
  const name = path === '/' ? 'index.html' : path.slice(1);
  if (!['GET', 'HEAD'].includes(req.method) || !allowed.has(name)) { res.writeHead(404); res.end('Not found'); return; }
  try {
    const body = await readFile(fileURLToPath(new URL(name, import.meta.url)));
    res.writeHead(200, { 'Content-Type': types[name.split('.').at(-1)], 'Cache-Control': 'no-cache', 'X-Content-Type-Options': 'nosniff', 'Referrer-Policy': 'no-referrer', 'Content-Security-Policy': "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; worker-src 'self'; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'" });
    res.end(req.method === 'HEAD' ? undefined : body);
  } catch { res.writeHead(500); res.end('Could not load asset'); }
}).listen(port, '127.0.0.1', () => console.log('班伴：http://127.0.0.1:' + port));
