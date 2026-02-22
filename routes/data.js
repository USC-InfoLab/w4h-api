import express from 'express';
const router = express.Router();

import * as auth from '../auth.js';
import db from '../db.js';

// router.get('/data', getData);
// router.get('/streaming-data', auth.authenticate, getStreamingData);  //with auth

export default router;