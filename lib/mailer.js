import nodemailer from 'nodemailer';

function parseBoolean(value, fallback = false) {
  if (value === undefined || value === null || value === '') {
    return fallback;
  }
  return String(value).toLowerCase() === 'true';
}

function getMailerConfig() {
  return {
    host: process.env.SMTP_HOST || '',
    port: process.env.SMTP_PORT || '587',
    secure: parseBoolean(process.env.SMTP_SECURE, false),
    user: process.env.SMTP_USER || '',
    pass: process.env.SMTP_PASS || '',
    from: process.env.SMTP_FROM || '',
    replyTo: process.env.SMTP_REPLY_TO || ''
  };
}

function isMailerConfigured() {
  const cfg = getMailerConfig();
  return !!(cfg.host && cfg.port && cfg.from);
}

function createTransporter() {
  const cfg = getMailerConfig();

  if (!isMailerConfigured()) {
    throw new Error('SMTP is not configured (host, port, and from are required)');
  }

  const transport = {
    host: cfg.host,
    port: Number(cfg.port),
    secure: cfg.secure
  };

  if (cfg.user || cfg.pass) {
    transport.auth = {
      user: cfg.user,
      pass: cfg.pass
    };
  }

  return nodemailer.createTransport(transport);
}

async function sendMail({ to, subject, text, html }) {
  const cfg = getMailerConfig();
  const transporter = createTransporter();

  return transporter.sendMail({
    from: cfg.from,
    to,
    subject,
    text,
    html,
    ...(cfg.replyTo ? { replyTo: cfg.replyTo } : {})
  });
}

export { getMailerConfig, isMailerConfigured, sendMail };
