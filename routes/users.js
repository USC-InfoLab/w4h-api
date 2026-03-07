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
router.get('/user/:email', auth.authenticate, getByEmail);
router.post('/add', auth.authenticate, addUser);
router.post('/set', auth.authenticate, setUser);
router.delete('/:email', auth.authenticate, del);
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
    await query(`DELETE FROM ${SCHEMA}.${TABLE} WHERE email = $1`, [req.params.email]);
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
    const params = [user.name, user.email, user.role, user.email];

    if (user.password) {
      const hashedPassword = bcrypt.hashSync(user.password, 8);
      updateQuery += `, password = $4`;
      updateQuery += ` WHERE email = $5`;
      params.push(hashedPassword);
      params[3] = user.email;
    } else {
      updateQuery += ` WHERE email = $4`;
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

    const email = (user.email || '').trim();
    const name = (user.name || email || '').trim();
    const role = user.role;
    const password = user.password;

    if (!email || !role || !password) {
      return res.status(400).json({ error: 'email, role, and password are required' });
    }

    const passwordValidation = validatePassword(password);
    if (!passwordValidation.valid) {
      return res.status(400).json({ error: passwordValidation.message });
    }

    const hashedPassword = bcrypt.hashSync(password, 8);

    await query(
      `INSERT INTO ${SCHEMA}.${TABLE} (name, email, password, role) VALUES ($1, $2, $3, $4)`,
      [name, email, hashedPassword, role]
    );
    res.status(201).json({ success: true, message: 'User created' });
  } catch (error) {
    if (error.code === '23505') {
      return res.status(409).json({ error: 'Email already exists' });
    }

    console.error('Add user error:', error);
    res.status(500).json({ error: error.message });
  }
}

async function getByEmail(req, res) {
  if (req.user.role !== 'admin') {
    return res.json(null);
  }
  try {
    const result = await query(
      `SELECT id, name, email, role, created_at FROM ${SCHEMA}.${TABLE} WHERE email = $1`,
      [req.params.email]
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
      `SELECT
         u.id,
         u.name,
         u.email,
         u.role,
         u.created_at,
         COALESCE(ARRAY_AGG(DISTINCT us.study_id) FILTER (WHERE us.study_id IS NOT NULL), '{}') AS study_ids,
         COALESCE(STRING_AGG(DISTINCT s.name, ', ') FILTER (WHERE s.name IS NOT NULL), '') AS studies
       FROM ${SCHEMA}.${TABLE} u
       LEFT JOIN ${SCHEMA}.user_studies us ON us.user_id = u.id
       LEFT JOIN ${SCHEMA}.studies s ON s.id = us.study_id
       GROUP BY u.id, u.name, u.email, u.role, u.created_at
       ORDER BY u.email ASC`
    );
    res.json(result.rows);
  } catch (error) {
    console.error('List users error:', error);
    res.status(500).json({ error: error.message });
  }
}

async function login(req, res) {
  try {
    const email = (req.body.email || req.body.username || '').trim();
    const { password } = req.body;
    
    if (!email || !password) {
      return res.json({ error: { message: 'Email and password are required' } });
    }

    const result = await query(
      `SELECT id, name, email, password, role FROM ${SCHEMA}.${TABLE} WHERE email = $1`,
      [email]
    );

    if (result.rows.length === 0) {
      return res.json({ error: { message: 'Invalid Email or Password!' } });
    }

    const user = result.rows[0];
    
    if (!bcrypt.compareSync(password, user.password)) {
      return res.json({ error: { message: 'Invalid Email or Password!' } });
    }

    const userInfo = {
      id: user.id,
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