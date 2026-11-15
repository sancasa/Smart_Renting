/**
 * Money and proposal actions carry an idempotency key. The key stays the same across
 * retries of one user intent, so a retry after an ambiguous failure cannot create a
 * second action. A new key is only minted for a new intent.
 */
export function newIdempotencyKey(): string {
  if (typeof crypto !== 'undefined' && 'randomUUID' in crypto) return crypto.randomUUID();
  return `key-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
}
