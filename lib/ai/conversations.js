import { query } from '../../db.js';

function titleFromMessages(messages) {
  const first = (messages || []).find((m) => m.role === 'user' && String(m.content || '').trim());
  if (!first) return null;
  let text = String(first.content).trim().replace(/\s+/g, ' ');
  const cut = text.search(/[.!?\n]/);
  if (cut > 12 && cut < 56) text = text.slice(0, cut);
  if (text.length > 36) text = `${text.slice(0, 36).trim()}…`;
  if (!text) return null;
  return text.charAt(0).toUpperCase() + text.slice(1);
}

export async function listConversations(userId) {
  const res = await query(
    `SELECT id, title, pinned, title_locked, created_at, updated_at
     FROM w4h.ai_conversations
     WHERE user_id = $1
     ORDER BY pinned DESC, updated_at DESC`,
    [userId]
  );
  return res.rows;
}

export async function getConversation(userId, id) {
  const conv = await query(
    `SELECT id, title, pinned, title_locked, created_at, updated_at
     FROM w4h.ai_conversations
     WHERE id = $1::uuid AND user_id = $2`,
    [id, userId]
  );
  if (!conv.rowCount) return null;
  const msgs = await query(
    `SELECT role, content, created_at
     FROM w4h.ai_messages
     WHERE conversation_id = $1::uuid
     ORDER BY created_at ASC`,
    [id]
  );
  return { ...conv.rows[0], messages: msgs.rows.map((m) => ({ role: m.role, content: m.content })) };
}

export async function createConversation(userId, { title } = {}) {
  const existing = await listConversations(userId);
  let max = 0;
  for (const c of existing) {
    const m = String(c.title || '').match(/^Chat\s+(\d+)$/i);
    if (m) max = Math.max(max, Number(m[1]));
  }
  const name = String(title || '').trim() || `Chat ${max + 1}`;
  const res = await query(
    `INSERT INTO w4h.ai_conversations (user_id, title)
     VALUES ($1, $2)
     RETURNING id, title, pinned, title_locked, created_at, updated_at`,
    [userId, name]
  );
  return { ...res.rows[0], messages: [] };
}

export async function updateConversation(userId, id, patch) {
  const sets = ['updated_at = NOW()'];
  const values = [id, userId];
  let p = 3;
  if (patch.title != null) {
    const title = String(patch.title).trim();
    if (title) {
      sets.push(`title = $${p++}`);
      values.push(title.length > 48 ? `${title.slice(0, 48)}…` : title);
      sets.push('title_locked = true');
    }
  }
  if (patch.pinned != null) {
    sets.push(`pinned = $${p++}`);
    values.push(!!patch.pinned);
  }
  const res = await query(
    `UPDATE w4h.ai_conversations SET ${sets.join(', ')}
     WHERE id = $1::uuid AND user_id = $2
     RETURNING id, title, pinned, title_locked, created_at, updated_at`,
    values
  );
  return res.rows[0] || null;
}

export async function deleteConversation(userId, id) {
  const res = await query(
    `DELETE FROM w4h.ai_conversations WHERE id = $1::uuid AND user_id = $2 RETURNING id`,
    [id, userId]
  );
  return res.rowCount > 0;
}

export async function appendTurn(userId, conversationId, { userContent, assistantContent, messages }) {
  let convId = conversationId;
  if (convId) {
    const owned = await query(
      `SELECT id, title_locked FROM w4h.ai_conversations WHERE id = $1::uuid AND user_id = $2`,
      [convId, userId]
    );
    if (!owned.rowCount) {
      const created = await createConversation(userId);
      convId = created.id;
    }
  } else {
    const created = await createConversation(userId);
    convId = created.id;
  }

  await query(
    `INSERT INTO w4h.ai_messages (conversation_id, role, content)
     VALUES ($1::uuid, 'user', $2), ($1::uuid, 'assistant', $3)`,
    [convId, userContent, assistantContent]
  );

  const owned = await query(
    `SELECT title_locked FROM w4h.ai_conversations WHERE id = $1::uuid`,
    [convId]
  );
  const autoTitle = !owned.rows[0]?.title_locked ? titleFromMessages(messages) : null;
  if (autoTitle) {
    await query(
      `UPDATE w4h.ai_conversations
       SET title = $2, updated_at = NOW()
       WHERE id = $1::uuid`,
      [convId, autoTitle]
    );
  } else {
    await query(
      `UPDATE w4h.ai_conversations SET updated_at = NOW() WHERE id = $1::uuid`,
      [convId]
    );
  }

  return getConversation(userId, convId);
}
