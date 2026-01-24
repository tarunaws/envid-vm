// Proxy setup for local Flask backends.
//
// This repo's active metadata stack is:
// - React frontend: http://localhost:3000
// - Envid Metadata (Multimodal): http://localhost:5016
//
// Older demo services (5011–5013) are intentionally disabled/archived.

const { createProxyMiddleware } = require('http-proxy-middleware');

module.exports = function(app) {
  // Envid Metadata (Multimodal)
  app.use(
    '/backend',
    createProxyMiddleware({
      target: 'http://localhost:5016',
      changeOrigin: true,
      pathRewrite: { '^/backend': '' },
      logLevel: 'warn',
    })
  );
};
