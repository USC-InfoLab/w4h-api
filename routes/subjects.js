import express from 'express';
const router = express.Router();

//const auth = require('../auth');
import db from '../db.js';

router.get('/users', getUsers);
router.get('/user/location/:id', getLocation);

//---

// app.get('/users', function (req, res) {
//   const sql = `SELECT * FROM public.geomts_users ORDER BY user_id ASC`;
//   const vals = db.query(sql).then(result => {
//     var res_arr = []
//     result.rows.forEach(element => {
//       res_arr.push(element.user_id);
//     })
//     res.json(res_arr);
//   })
// });

function getUsers(req, res) {
  const sql = `SELECT user_id FROM public.geomts_users ORDER BY user_id ASC`;
  db.query(sql).then(result => {
    var res_arr = [];
    result.rows.forEach(element => {
      res_arr.push(element.user_id);
    })
    res.json(res_arr);
  });
}

//---

// app.get('/user/location/:id', function (req, res) {
//   const id = req.params.id;
//   const sql = `SELECT ST_AsText(coordinates) FROM public.locations WHERE user_id = '` + id.toString() + `'`;
//   const vals = query(sql).then(result => {
//     var res_arr = []
//     result.rows.forEach(element => {
//       var coord = element['st_astext'].replace('POINT', '')
//       coord = coord.replaceAll('(', '[')
//       coord = coord.replaceAll(' ', ',')
//       coord = coord.replaceAll(')', ']')
//       coord = coord.replace(/\[|\]/g,'').split(',')
//       coord[0] = parseFloat(coord[0])
//       coord[1] = parseFloat(coord[1])
//       res_arr.push(coord)
//     });
//     //{"lat": 0984320843, "lon": 098r40384}
//     //["lat": 0984320843, "lon": 098r40384]
//     res.json(res_arr);
//   })
// });

function getLocation(req, res) { //seems to return a path, why not a timestamp or at least a duration?
  const sql = `SELECT ST_AsText(coordinates) FROM public.locations WHERE user_id = '` + req.params.id + `'`;
  db.query(sql).then(result => {
    var res_arr = []; //why do all this reformatting for lat, long?
    result.rows.forEach(d => { //we array.map new = old.map(d => {...})
      var coord = d['st_astext'].replace('POINT', '')
      coord = coord.replaceAll('(', '[')
      coord = coord.replaceAll(' ', ',')
      coord = coord.replaceAll(')', ']')
      coord = coord.replace(/\[|\]/g,'').split(',')
      coord[0] = parseFloat(coord[0])
      coord[1] = parseFloat(coord[1])
      res_arr.push(coord)
    });
    //should return something understandable
    //
    //[{"lat": 0984320843, "lon": 098r40384, "duration": },{"lat": 0984320843, "lon": 098r40384},...]
    res.json(res_arr);
  });
}


export default router;