import jwt from 'jsonwebtoken';
import { readUserApiKeyFromRequest, loadUserFromApiKeyRaw } from './userApiKeyAuth.js';

function authenticateJwtOrUserApiKey(req, res, next) {
  const rawKey = readUserApiKeyFromRequest(req);
  if (rawKey) {
    return loadUserFromApiKeyRaw(rawKey)
      .then((user) => {
        if (!user) {
          return res.sendStatus(401);
        }
        req.user = user;
        req.authViaUserApiKey = true;
        next();
      })
      .catch((err) => {
        console.error('User API key auth error:', err);
        res.status(500).json({ error: err.message || 'Authentication failed' });
      });
  }

  const authorization = req.headers.authorization;
  if (!authorization) {
    return res.sendStatus(401);
  }

  jwt.verify(authorization, process.env.TOKEN_SECRET, (err, decoded) => {
    if (err) {
      return res.sendStatus(403);
    }
    req.user = decoded;
    next();
  });
}

export { authenticateJwtOrUserApiKey };
