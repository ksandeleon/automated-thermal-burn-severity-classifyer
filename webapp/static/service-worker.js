// Service Worker for Burn Classifier PWA
const CACHE_NAME = 'burn-classifier-v2';
const urlsToCache = [
  '/',
  '/static/css/style.css',
  '/static/css/index-style.css',
  '/static/css/result-style.css',
  '/static/js/main.js',
  '/static/manifest.json',
  // Bootstrap & External Libraries (cached from CDN)
  'https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css',
  'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css',
  'https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap'
];

// Install Event - Cache resources
self.addEventListener('install', (event) => {
  console.log('[Service Worker] Installing...');
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then((cache) => {
        console.log('[Service Worker] Caching app shell');
        return cache.addAll(urlsToCache);
      })
      .then(() => {
        console.log('[Service Worker] Installed successfully');
        return self.skipWaiting(); // Activate worker immediately
      })
      .catch((error) => {
        console.error('[Service Worker] Installation failed:', error);
      })
  );
});

// Activate Event - Clean up old caches
self.addEventListener('activate', (event) => {
  console.log('[Service Worker] Activating...');
  event.waitUntil(
    caches.keys()
      .then((cacheNames) => {
        return Promise.all(
          cacheNames.map((cacheName) => {
            if (cacheName !== CACHE_NAME) {
              console.log('[Service Worker] Deleting old cache:', cacheName);
              return caches.delete(cacheName);
            }
          })
        );
      })
      .then(() => {
        console.log('[Service Worker] Activated successfully');
        return self.clients.claim(); // Take control of all pages
      })
  );
});

// Fetch Event - Serve from cache, fallback to network
self.addEventListener('fetch', (event) => {
  const { request } = event;

  // Skip caching for API requests (upload, analyze, progress)
  if (request.url.includes('/upload') ||
      request.url.includes('/analyze') ||
      request.url.includes('/progress') ||
      request.method !== 'GET') {
    return; // Let the browser handle it normally
  }

  event.respondWith(
    caches.match(request)
      .then((cachedResponse) => {
        if (cachedResponse) {
          console.log('[Service Worker] Serving from cache:', request.url);
          return cachedResponse;
        }

        console.log('[Service Worker] Fetching from network:', request.url);
        return fetch(request)
          .then((response) => {
            // Don't cache if not a valid response
            if (!response || response.status !== 200 || response.type === 'error') {
              return response;
            }

            // Clone the response
            const responseToCache = response.clone();

            // Cache static resources only
            if (request.url.includes('/static/') ||
                request.url.includes('cdn.jsdelivr.net') ||
                request.url.includes('cdnjs.cloudflare.com') ||
                request.url.includes('fonts.googleapis.com')) {
              caches.open(CACHE_NAME)
                .then((cache) => {
                  cache.put(request, responseToCache);
                });
            }

            return response;
          })
          .catch((error) => {
            console.error('[Service Worker] Fetch failed:', error);
            // You can return a custom offline page here
            return caches.match('/');
          });
      })
  );
});

// Handle messages from the main thread
self.addEventListener('message', (event) => {
  if (event.data && event.data.type === 'SKIP_WAITING') {
    self.skipWaiting();
  }

  if (event.data && event.data.type === 'CACHE_URLS') {
    event.waitUntil(
      caches.open(CACHE_NAME)
        .then((cache) => cache.addAll(event.data.payload))
    );
  }
});
