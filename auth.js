const jwt = require('jsonwebtoken');

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

module.exports = {
  authenticate: authenticate,
  sign: sign
};
