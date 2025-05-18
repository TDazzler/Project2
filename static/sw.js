self.addEventListener('install', event => {
  event.waitUntil(
    caches.open('v1').then(cache => cache.addAll([
      '/',
      '/static/manifest.json',
      '/static/icons/icon-192.png',
      '/static/icons/icon-512.png'
    ]))
  );
});

self.addEventListener('install', e => {
  console.log('[ServiceWorker] Installed');
});

self.addEventListener('activate', e => {
  console.log('[ServiceWorker] Activated');
});

self.addEventListener('fetch', event => {
  event.respondWith(
    caches.match(event.request).then(response => response || fetch(event.request))
  );
});
