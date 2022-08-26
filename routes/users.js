const express = require('express');
const router = express.Router();
const fs = require('fs-extra');

const auth = require('../auth');

const jwt = require('jsonwebtoken')
router.post('/login', login);
router.get('/', auth.authenticate, list);
router.get('/user:name', auth.authenticate, getByName);
router.post('/add', auth.authenticate, addUser);
router.post('/set', auth.authenticate, setUser);
router.delete('/:name', auth.authenticate, del);
router.get('/roles', auth.authenticate, roles);

function roles(req, res) {
  if (req.user.role !== 'admin') {
    return res.json(null);
  }
  res.json(['editor', 'reporter', 'admin']);
}

function del(req, res) {
  if (req.user.role !== 'admin') {
    return res.json(null);
  }
  let users = fs.readJSONSync(FC.users_file);
  users = users.filter(d => d.name !== req.params.name);
  fs.writeJSONSync(FC.users_file, users);
  res.json('OK');
}

function setUser(req, res) {
  if (req.user.role !== 'admin') {
    return res.json(null);
  }
  let users = fs.readJSONSync(FC.users_file);
  let user = users.find(d => d.name === req.body.user.name);
  if (user) {
    user.username = req.body.user.username;
    user.name = req.body.user.name;
    user.email = req.body.user.email;
    user.role = req.body.user.role;
    if (req.body.user.password) {
      const bcrypt = require('bcrypt');  
      user.password = bcrypt.hashSync(req.body.user.password, 8);
    }
    fs.writeJSONSync(FC.users_file, users);
    res.json('OK');
  } else {
    res.json(null);
  }
}

function addUser(req, res) {
  if (req.user.role !== 'admin') {
    return res.json(null);
  }
  let users = fs.readJSONSync(FC.users_file);
  let newUser = req.body.user;
  const bcrypt = require('bcrypt');  
  newUser.password = bcrypt.hashSync(newUser.password, 8);
  users.push(newUser);
  fs.writeJSONSync(FC.users_file, users);
  res.json('OK');
}

function getByName(req, res) {
  if (req.user.role !== 'admin') {
    return res.json(null);
  }

  let result = [];
  const users = fs.readJSONSync(FC.users_file);
  const user = users.find(d => d.name === req.params.name);
  delete user.password;
  res.json(user);
}

function list(req, res) {
  if (req.user.role !== 'admin') {
    return res.json(null);
  }

  let result = [];
  const users = fs.readJSONSync(FC.users_file);
  users.forEach(d => {
    result.push({ 
      username: d.username, 
      name: d.name, 
      email: d.email, 
      role: d.role
    });
  });
  res.json(result);
}

function login(req, res) {
  const users = fs.readJSONSync(FC.users_file);
  const user = users.find(d => d.username == req.body.username)
  if (!user) {
    return res.json({ error: { message: "Invalid Username or Password!" } });
  }
  const bcrypt = require('bcrypt');
  //const passwordIsValid = bcrypt.compareSync(req.body.password, bcrypt.hashSync('SHS3b527knbdRgg', 8));
  if (!bcrypt.compareSync(req.body.password, user.password)) {
    return res.json({ error: { message: "Invalid Username or Password!" } });
  }

  const userInfo = {
    username: user.username, 
    name: user.name, 
    email: user.email, 
    role: user.role
  };

  const auth = require("../auth");
  const token = auth.sign(userInfo);
  userInfo.token = token;
  
  res.json(userInfo);
}

module.exports = router;