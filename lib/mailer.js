import nodemailer from 'nodemailer';

function parseBoolean(value, fallback = false) {
  if (value === undefined || value === null || value === '') {
    return fallback;
  }
  return String(value).toLowerCase() === 'true';
}

function getMailerConfig(overrides = null) {
  const fromEnv = {
    host: process.env.SMTP_HOST || '',
    port: process.env.SMTP_PORT || '587',
    secure: parseBoolean(process.env.SMTP_SECURE, false),
    user: process.env.SMTP_USER || '',
    pass: process.env.SMTP_PASS || '',
    from: process.env.SMTP_FROM || '',
    replyTo: process.env.SMTP_REPLY_TO || ''
  };

  if (!overrides || typeof overrides !== 'object') {
    return fromEnv;
  }

  const pass =
    overrides.password !== undefined && overrides.password !== null && overrides.password !== ''
      ? String(overrides.password)
      : (overrides.pass !== undefined && overrides.pass !== null && overrides.pass !== ''
        ? String(overrides.pass)
        : fromEnv.pass);

  return {
    host: overrides.host != null && String(overrides.host).trim() !== ''
      ? String(overrides.host).trim()
      : fromEnv.host,
    port: overrides.port != null && String(overrides.port).trim() !== ''
      ? String(overrides.port).trim()
      : fromEnv.port,
    secure: overrides.secure !== undefined && overrides.secure !== null
      ? !!overrides.secure
      : fromEnv.secure,
    user: overrides.user != null
      ? String(overrides.user).trim()
      : fromEnv.user,
    pass,
    from: overrides.from != null && String(overrides.from).trim() !== ''
      ? String(overrides.from).trim()
      : fromEnv.from,
    replyTo: overrides.replyTo != null
      ? String(overrides.replyTo).trim()
      : fromEnv.replyTo
  };
}

function isMailerConfigured(cfg = null) {
  const c = cfg || getMailerConfig();
  return !!(c.host && c.port && c.from);
}

function createTransporter(overrides = null) {
  const cfg = getMailerConfig(overrides);

  if (!isMailerConfigured(cfg)) {
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

async function sendMail({ to, subject, text, html, smtp = null }) {
  const cfg = getMailerConfig(smtp);
  const transporter = createTransporter(smtp);

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
