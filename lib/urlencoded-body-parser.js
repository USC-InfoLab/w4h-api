import express from 'express';

// File uploads stream the request body, even when curl labels it as form data.
export const urlencodedBodyParser = express.urlencoded({
  limit: '10mb',
  extended: true,
  type: (req) => {
    const uploadFilePut = req.method === 'PUT'
      && /^\/(?:contributions|packages)\/uploads\/[^/]+\/files\/[^/]+\/?$/.test(req.path);
    return !uploadFilePut && Boolean(req.is('application/x-www-form-urlencoded'));
  },
});
