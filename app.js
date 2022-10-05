const path = require('path');
require('dotenv').config({ path: path.join(__dirname, '.env') });

if (process.env.NODE_ENV == 'development') {
  console.log('process.env');
  console.dir(process.env, {showHidden: true, depth: Infinity, colors: true, maxArrayLength: Infinity});
}

///////////////////////////////////////////////////////////////////////

const express = require('express');
const cors = require('cors');
const app = express();

app.use(cors());

const bodyParser = require('body-parser');
const { query } = require('./db');
// app.use(express.urlencoded({extended: true})); 
app.use(bodyParser.json()); // support json encoded bodies
app.use(bodyParser.urlencoded({ extended: true })); // support encoded bodies
app.use(express.static(path.join(__dirname, 'public')));

app.on('uncaughtException', err => {
  console.log('uncaught exception: ', err);
  process.send({ finish: false });
});

app.listen(process.env.PORT, () => {
  console.log('App running at: http://localhost:' + process.env.PORT);
});

//////////////////////////////////////////////////////////////////////

app.get('/users', function(req, res) {
  const sql = `SELECT * FROM public.geomts_users ORDER BY user_id ASC`;
  const vals = query(sql).then(result => {
    console.log(result);
    res.json(result.rows);
  })
});


app.get('/user/calories/:id', function(req, res) {
  const id = req.params.id;
  console.log(id);
  const sql = `SELECT * FROM public.calories WHERE user_id = '` + id.toString() +  `' ORDER BY user_id ASC`;
  const vals = query(sql).then(result => {
    console.log(result);
    res.json(result.rows);
  })
});

app.get('/user/heartrate/:id', function(req, res) {
  const id = req.params.id;
  console.log(id);
  const sql = `SELECT * FROM public.heart_rates WHERE user_id = '` + id.toString() +  `' ORDER BY user_id ASC`;
  const vals = query(sql).then(result => {
    console.log(result);
    res.json(result.rows);
  })
});

app.get('/calories', function(req, res) {
  const sql = `SELECT * FROM public.calories ORDER BY user_id ASC`;
  const vals = query(sql).then(result => {
    console.log(result);
    res.json(result.rows);
  })
});

app.get('/heartrate', function(req, res) {
  const sql = `SELECT * FROM public.heart_rates ORDER BY user_id ASC`;
  const vals = query(sql).then(result => {
    console.log(result);
    res.json(result.rows);
  })
});

app.use('/data', require('./routes/data'));
app.use('/users', require('./routes/users'));

app.get('/', function (req, res) {
  res.send('W4H API');
});
