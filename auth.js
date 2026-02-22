import jwt from 'jsonwebtoken';

const authenticate = function(req, res, next) {
  const authorization = req.headers.authorization;
  if (authorization) {
    jwt.verify(authorization, process.env.TOKEN_SECRET, (err, decoded) => {
      if (err) {
        return res.sendStatus(403);
      }

      req.user = decoded;
      next();
    });
  } else {
    res.sendStatus(401);
  }
};

const sign = function(user) {
  return jwt.sign(user, process.env.TOKEN_SECRET, { expiresIn: 86400 });  //expires in 24h
};

const requireAdmin = function(req, res, next) {
  if (!req.user || req.user.role !== 'admin') {
    return res.status(403).json({ 
      success: false, 
      error: 'Access denied. Admin privileges required.' 
    });
  }
  next();
};

export { authenticate, sign, requireAdmin };
