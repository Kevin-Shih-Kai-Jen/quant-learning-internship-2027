const CACHE = 'banban-shell-v2-20260925-1';
const ASSETS = ['./', './index.html', './styles.css', './app.mjs', './domain.mjs', './storage.mjs', './solver.mjs', './solver-worker.mjs', './manifest.webmanifest', './icon.svg'];
self.addEventListener('install', event => event.waitUntil(caches.open(CACHE).then(cache => cache.addAll(ASSETS))));
self.addEventListener('activate', event => event.waitUntil(caches.keys().then(keys => Promise.all(keys.filter(key => key.startsWith('banban-shell-') && key !== CACHE).map(key => caches.delete(key))))));
self.addEventListener('fetch', event => {
  if (event.request.method !== 'GET' || new URL(event.request.url).origin !== self.location.origin) return;
  event.respondWith(fetch(event.request).catch(() => caches.match(event.request).then(cached => cached || new Response('Offline', { status: 503 }))));
});
