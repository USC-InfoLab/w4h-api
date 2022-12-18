const express = require('express');
const router = express.Router();

//const auth = require('../auth');
const { query } = require('../db');

router.get('/users', getUsers);
router.get('/user/location/:id', getLocation);

//---

// app.get('/users', function (req, res) {
//   const sql = `SELECT * FROM public.geomts_users ORDER BY user_id ASC`;
//   const vals = query(sql).then(result => {
//     var res_arr = []
//     result.rows.forEach(element => {
//       res_arr.push(element.user_id);
//     })
//     res.json(res_arr);
//   })
// });

function getUsers(req, res) {
  const sql = `SELECT user_id FROM public.geomts_users ORDER BY user_id ASC`;
  query(sql).then(result => {
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
  query(sql).then(result => {
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

//---

/*
app.get('/user/pastdata/:id/:startdate/:enddate', function (req, res) {
  const id = req.params.id;
  const startdate = req.params.startdate;
  const startdatelastweek = new Date(Date.parse(startdate) - 7 * 24 * 60 * 60 * 1000)
  var pstDateStart = startdatelastweek.toISOString()

  const enddate = req.params.enddate;
  const enddatelastweek = new Date(Date.parse(enddate) - 7 * 24 * 60 * 60 * 100)
  var pstDateEnd = enddatelastweek.toISOString()

  const sqlCalCurr = `SELECT * FROM public.calories WHERE user_id = '` + id.toString() + `' AND timestamp BETWEEN '` + startdate.toString() + `' AND '` + enddate.toString() + `' order by user_id ASC;`;
  const sqlCalLast = `SELECT * FROM public.calories WHERE user_id = '` + id.toString() + `' AND timestamp BETWEEN '` + pstDateStart.toString() + `' AND '` + pstDateEnd.toString() + `' order by user_id ASC;`;
  const sqlHRCurr = `SELECT * FROM public.heart_rates WHERE user_id = '` + id.toString() + `' AND timestamp BETWEEN '` + startdate.toString() + `' AND '` + enddate.toString() + `' order by user_id ASC;`;
  const sqlHRLast = `SELECT * FROM public.heart_rates WHERE user_id = '` + id.toString() + `' AND timestamp BETWEEN '` + pstDateStart.toString() + `' AND '` + pstDateEnd.toString() + `' order by user_id ASC;`;

  var res1 = {
    'current_week_cal': 0,
    'last_week_cal': 0,
    'current_week_hr': 0,
    'last_week_hr': 0,
    'currCalAvg': 0,
    'prevCaleAvg': 0,
    'changeInCal': 0,
    'currHeartRateAvg': 0,
    'prevHeartRateAvg': 0,
    'changeInHR': 0
  }
  const vals = query(sqlCalCurr).then(result => {
    result.rows.forEach(element => {
      res1['current_week_cal'] += element.value
    });
    res1['currCalAvg'] = res1['current_week_cal']/7
    const vals2 = query(sqlCalLast).then(result2 => {
      result2.rows.forEach(element => {
        res1['last_week_cal'] += element.value
      });
      res1['prevCalAvg'] = res1['last_week_cal']/7
      res1['changeInCal'] = ((res1['currCalAvg'] - res1['prevCalAvg']) / res1['prevCalAvg']) * 100
      const vals3 = query(sqlHRCurr).then(result3 => {
        result3.rows.forEach(element => {
          res1['current_week_hr'] += element.value
        });
        res1['currHeartRateAvg'] = res1['current_week_hr'] / 7
        const vals4 = query(sqlHRLast).then(result4 => {
          result4.rows.forEach(element => {
            res1['last_week_hr'] += element.value
          });
          res1['prevHeartRateAvg'] = res1['last_week_hr'] / 7
          res1['changeInHR'] = ((res1['currHeartRateAvg'] - res1['prevHeartRateAvg']) / res1['prevHeartRateAvg']) * 100
          res.json(res1);
        })
      })
    })
  })
});

app.get('/user/calories/:id', function (req, res) {
  const id = req.params.id;
  console.log(id);
  const sql = `SELECT * FROM public.calories WHERE user_id = '` + id.toString() + `' ORDER BY user_id ASC`;
  const vals = query(sql).then(result => {
    var res_arr = []
    result.rows.forEach(element => {
      res_arr.push([Date.parse(element.timestamp), element.value])
    });
    res.json(res_arr);
  })
});

app.get('/user/heartrate/:id', function (req, res) {
  const id = req.params.id;
  console.log(id);
  const sql = `SELECT * FROM public.heart_rates WHERE user_id = '` + id.toString() + `' ORDER BY timestamp ASC`;
  const vals = query(sql).then(result => {
    var res_arr = []
    result.rows.forEach(element => {
      res_arr.push([Date.parse(element.timestamp), element.value])
    });
    res.json(res_arr);
  })
});

app.get('/user/activity/:id', function (req, res) {
  const id = req.params.id;
  console.log(id);
  const sql = `SELECT * FROM public.heart_rates WHERE user_id = '` + id.toString() + `' ORDER BY timestamp ASC`;
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
    ans['low'] = (res_low); //XXX: never use array['key'] use array.key {lat: o9i4e3}
    ans['moderate'] = (res_mod);
    ans['high'] = (res_high);
    res.json(ans);
  })
});
*/

module.exports = router;