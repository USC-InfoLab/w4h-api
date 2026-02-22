import express from 'express';
const router = express.Router();
import bcrypt from 'bcrypt';
import { query } from '../lib/utils/db.js';
import * as auth from '../auth.js';
import jwt from 'jsonwebtoken';
import { validatePassword } from '../lib/utils/password-validator.js';

const SCHEMA = 'w4h';
const TABLE = 'users';

router.post('/login', login);
router.post('/token-valid', tokenValid);
router.get('/', auth.authenticate, list);
router.get('/user/:username', auth.authenticate, getByUsername);
router.post('/add', auth.authenticate, addUser);
router.post('/set', auth.authenticate, setUser);
router.delete('/:username', auth.authenticate, del);
router.get('/roles', auth.authenticate, roles);

function roles(req, res) {
  if (req.user.role !== 'admin') {
    return res.json(null);
  }
  res.json(['clinician', 'researcher', 'admin']);
}

function tokenValid(req, res) {
  try {
    const { token } = req.body;
    if (!token) {
      return res.json({ tokenExpired: true });
    }
    
    // Verify the token (will throw if expired or invalid)
    jwt.verify(token, process.env.TOKEN_SECRET);
    res.json({ tokenExpired: false });
  } catch (error) {
    res.json({ tokenExpired: true });
  }
}

async function del(req, res) {
  if (req.user.role !== 'admin') {
    return res.json(null);
  }
  try {
    await query(`DELETE FROM ${SCHEMA}.${TABLE} WHERE username = $1`, [req.params.username]);
    res.json({ success: true, message: 'User deleted' });
  } catch (error) {
    console.error('Delete user error:', error);
    res.status(500).json({ error: error.message });
  }
}

async function setUser(req, res) {
  if (req.user.role !== 'admin') {
    return res.json(null);
  }
  try {
    const { user } = req.body;
    
    // Validate password if provided
    if (user.password) {
      const passwordValidation = validatePassword(user.password);
      if (!passwordValidation.valid) {
        return res.status(400).json({ error: passwordValidation.message });
      }
    }
    
    let updateQuery = `UPDATE ${SCHEMA}.${TABLE} SET name = $1, email = $2, role = $3`;
    const params = [user.name, user.email, user.role, user.username];

    if (user.password) {
      const hashedPassword = bcrypt.hashSync(user.password, 8);
      updateQuery += `, password = $4`;
      updateQuery += ` WHERE username = $5`;
      params.push(hashedPassword);
      params[3] = user.username;
    } else {
      updateQuery += ` WHERE username = $4`;
    }

    const result = await query(updateQuery, params);
    if (result.rowCount > 0) {
      res.json({ success: true, message: 'User updated' });
    } else {
      res.json({ error: 'User not found' });
    }
  } catch (error) {
    console.error('Set user error:', error);
    res.status(500).json({ error: error.message });
  }
}

async function addUser(req, res) {
  if (req.user.role !== 'admin') {
    return res.json(null);
  }
  try {
    const { user } = req.body;
    
    // Validate password if provided
    if (user.password) {
      const passwordValidation = validatePassword(user.password);
      if (!passwordValidation.valid) {
        return res.status(400).json({ error: passwordValidation.message });
      }
    }
    
    const hashedPassword = bcrypt.hashSync(user.password, 8);

    await query(
      `INSERT INTO ${SCHEMA}.${TABLE} (username, name, email, password, role) VALUES ($1, $2, $3, $4, $5)`,
      [user.username, user.name, user.email, hashedPassword, user.role]
    );
    res.json({ success: true, message: 'User created' });
  } catch (error) {
    console.error('Add user error:', error);
    res.status(500).json({ error: error.message });
  }
}

async function getByUsername(req, res) {
  if (req.user.role !== 'admin') {
    return res.json(null);
  }
  try {
    const result = await query(
      `SELECT id, username, name, email, role, created_at FROM ${SCHEMA}.${TABLE} WHERE username = $1`,
      [req.params.username]
    );
    if (result.rows.length > 0) {
      res.json(result.rows[0]);
    } else {
      res.json(null);
    }
  } catch (error) {
    console.error('Get user error:', error);
    res.status(500).json({ error: error.message });
  }
}

async function list(req, res) {
  if (req.user.role !== 'admin') {
    return res.json(null);
  }
  try {
    const result = await query(
      `SELECT id, username, name, email, role, created_at FROM ${SCHEMA}.${TABLE} ORDER BY username ASC`
    );
    res.json(result.rows);
  } catch (error) {
    console.error('List users error:', error);
    res.status(500).json({ error: error.message });
  }
}

async function login(req, res) {
  try {
    const { username, password } = req.body;
    
    if (!username || !password) {
      return res.json({ error: { message: 'Username and password are required' } });
    }

    const result = await query(
      `SELECT id, username, name, email, password, role FROM ${SCHEMA}.${TABLE} WHERE username = $1`,
      [username]
    );

    if (result.rows.length === 0) {
      return res.json({ error: { message: 'Invalid Username or Password!' } });
    }

    const user = result.rows[0];
    
    if (!bcrypt.compareSync(password, user.password)) {
      return res.json({ error: { message: 'Invalid Username or Password!' } });
    }

    const userInfo = {
      id: user.id,
      username: user.username,
      name: user.name,
      email: user.email,
      role: user.role
    };

    const token = auth.sign(userInfo);
    userInfo.token = token;

    res.json(userInfo);
  } catch (error) {
    console.error('Login error:', error);
    res.status(500).json({ error: error.message });
  }
}

export default router;