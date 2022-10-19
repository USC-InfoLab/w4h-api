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
    var res_arr = []
    result.rows.forEach(element => {
      res_arr.push([Date.parse(element.timestamp), element.value])
    });
    res.json(res_arr);
  })
});

app.get('/user/heartrate/:id', function(req, res) {
  const id = req.params.id;
  console.log(id);
  const sql = `SELECT * FROM public.heart_rates WHERE user_id = '` + id.toString() +  `' ORDER BY timestamp ASC`;
  const vals = query(sql).then(result => {
    var res_arr = []
    result.rows.forEach(element => {
      res_arr.push([Date.parse(element.timestamp), element.value])
    });
    res.json(res_arr);
  })
});

app.get('/calories', function(req, res) {
  const sql = `SELECT * FROM public.calories ORDER BY timestamp ASC`;
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

app.get('/user/activity/:id', function(req, res) {
  const id = req.params.id;
  console.log(id);
  const sql = `SELECT * FROM public.heart_rates WHERE user_id = '` + id.toString() +  `' ORDER BY timestamp ASC`;
  const vals = query(sql).then(result => {
    console.log(result);
    var res_low = []
    var res_mod = []
    var res_high = []

    result.rows.forEach(element => {
      if (element.value < 60) {
        res_low.push([Date.parse(element.timestamp), element.value])
      } else if (element.value < 100 && element.value >= 60) {
        res_mod.push([Date.parse(element.timestamp), element.value])
      } else {
        res_high.push([Date.parse(element.timestamp), element.value])
      }
    });
    var ans = {}
    ans['low'] = (res_low);
    ans['moderate'] = (res_mod);
    ans['high'] = (res_high);
    res.json(ans);
  })
});

app.use('/data', require('./routes/data'));
app.use('/users', require('./routes/users'));

app.get('/', function (req, res) {
  res.send('W4H API');
});
