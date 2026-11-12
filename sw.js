/* Scan the World test app — app shell cache. Results are always fetched
   fresh (network only), so a newly published result appears without an
   app update. */
const CACHE = "stw-shell-v12"; // v3.7: fresh cache name — a new build must never ride the old shell (iOS stale-cache lesson)
const SHELL = ["./", "index.html", "style.css", "app.js", "pcm-worklet.js", "manifest.webmanifest",
               "icons/icon-192.png", "icons/icon-512.png", "icons/icon-180.png"];
self.addEventListener("install", e => {
  e.waitUntil(caches.open(CACHE).then(c => c.addAll(SHELL)).then(() => self.skipWaiting()));
});
self.addEventListener("activate", e => {
  e.waitUntil(caches.keys().then(keys => Promise.all(
    keys.filter(k => k !== CACHE).map(k => caches.delete(k)))).then(() => self.clients.claim()));
});
self.addEventListener("fetch", e => {
  const url = new URL(e.request.url);
  if (url.origin !== self.location.origin) return; // e.g. the 3D viewer library — let the browser handle it
  if (url.pathname.includes("/results/")) return; // network only (evidence + trophy model)
  e.respondWith(caches.match(e.request).then(hit => hit || fetch(e.request)));
});
