const {Pool} = require('pg');;

const pool = new Pool({
  host: process.env.DB_HOST,
  port: process.env.DB_PORT,
  database: process.env.DB_DATABASE,
  user: process.env.DB_USER,
  password: process.env.DB_PASSWORD
});

pool.on('error', (err) => {
  reject('Unexpected error on idle client', err);
});

module.exports = {
  async query(text, params) {
    let result;
    try {
      result = await pool.query(text, params);
    } catch (err) {
      console.error(err);
    }
    return result;
  }
};