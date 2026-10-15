import { test } from 'node:test';
import assert from 'node:assert/strict';
import { validateIdent, quoteIdent } from './sql-ident.js';

test('quoteIdent preserves case and wraps in double quotes', () => {
  assert.equal(quoteIdent('HR'), '"HR"');
  assert.equal(quoteIdent('Subject ID'), '"Subject ID"');
});

test('quoteIdent escapes embedded double quotes', () => {
  assert.equal(quoteIdent('a"b'), '"a""b"');
});

test('validateIdent rejects empty and NUL', () => {
  assert.throws(() => validateIdent(''));
  assert.throws(() => validateIdent('a\0b'));
});

test('validateIdent accepts/rejects at the 63-byte boundary', () => {
  const ok62 = 'a'.repeat(62);
  const ok63 = 'a'.repeat(63);
  const bad64 = 'a'.repeat(64);
  assert.equal(validateIdent(ok62), ok62);
  assert.equal(validateIdent(ok63), ok63);
  assert.throws(() => validateIdent(bad64));
});
