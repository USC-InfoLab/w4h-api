const express = require('express');
const router = express.Router();

const auth = require('../auth');
const { query } = require('../db');

router.get('/data', getData);
router.get('/streaming-data', getStreamingData);
// router.get('/streaming-data', auth.authenticate, getStreamingData);  //with auth

async function getStreamingData(req, res) {
  //auth
  // if (req.user.role !== 'admin' && req.user.role !== 'editor') {
  //   return res.json(new Error('invalid auth'));
  // }

  const sql = `SELECT * FROM crime.all_cities_matview_neighborhoods WHERE neighborhood ='downtown' LIMIT 10 `;
  const result = await query(sql);

  res.json(result);
}

function getData(req, res) {
  //auth
  // if (req.user.role !== 'admin' && req.user.role !== 'editor') {
  //   return res.json(new Error('invalid auth'));
  // }

  const sql = `SELECT * FROM crime.all_cities_matview_neighborhoods WHERE neighborhood ='downtown' LIMIT 10 `;
  const result = query(sql).then(result => {
    console.log(result);
    res.json(result.rows);
  });
}

function fetchUsersGarmin(req, res) {
  const sql = `SELECT * FROM public.geomts_users ORDER BY user_id ASC`;
  const vals = query(sql).then(result => {
    console.log(result);
    res.json(result.rows);
  });
}

module.exports = router;