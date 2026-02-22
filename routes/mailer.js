import express from 'express';
const router = express.Router();
import { authenticate } from '../auth.js';
import fs from 'fs-extra';
import dayjs from 'dayjs';
import { join } from 'path';

router.post('/send', authenticate, send);

import {execa} from 'execa';

async function send(req, res) {
  // Check user role
  if (req.user.role !== 'admin') {
    return res.status(403).json({ error: 'Unauthorized' });
  }

  const command = `${process.env.XNN_CLI} send --exec-send -p ${req.body.partitions.join(',')} -a ${req.body.emails.join(',')} ${req.body.print_id}`;
  const { stdout, stderr } = await execa`${process.env.XNN_CLI} send --exec-send -p ${req.body.partitions.join(',')} -a ${req.body.emails.join(',')} ${req.body.print_id}`;

  // Filter stderr for relevant errors in case we're debugging!
  const filteredStderr = stderr
    .split('\n')
    .filter((line) => !line.includes('Debugger attached') && !line.includes('Waiting for the debugger to disconnect'))
    .join('\n');

  if (filteredStderr) {
    return res.json({
        "status": "error",
        "message": `Failed to execute ${process.env.XNN_CLI} send.`,
        "details": {
          "stderr": stderr,
          "stdout": stdout,
          "command": command
        }
      });
  }

  // If no error, return success message with output
  return res.json({
    "status": "success",
    message: `Emails sent to ${req.body.emails.join(', ')}` 
  });
}

export default router;