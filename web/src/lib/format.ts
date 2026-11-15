/** Accepts "YYYY-MM-DD" (shown as that calendar day) or a full ISO timestamp. */
export function formatDate(iso: string): string {
  const dateOnly = /^(\d{4})-(\d{2})-(\d{2})$/.exec(iso);
  const date = dateOnly ? new Date(Number(dateOnly[1]), Number(dateOnly[2]) - 1, Number(dateOnly[3])) : new Date(iso);
  return new Intl.DateTimeFormat('en-US', { dateStyle: 'medium' }).format(date);
}

export function formatDateTime(iso: string): string {
  return new Intl.DateTimeFormat('en-US', { dateStyle: 'medium', timeStyle: 'short' }).format(new Date(iso));
}

export function pluralize(count: number, singular: string, plural = `${singular}s`): string {
  return `${count} ${count === 1 ? singular : plural}`;
}

const firstCharacter = (word: string): string => Array.from(word)[0] ?? '';

/** Up to two initials for an avatar: first and last name, or the first two letters of a single name. */
export function initialsOf(name: string): string {
  const words = name.trim().split(/[\s-]+/).filter(Boolean);
  if (words.length === 0) return '?';
  const first = firstCharacter(words[0]);
  const second = words.length > 1 ? firstCharacter(words[words.length - 1]) : (Array.from(words[0])[1] ?? '');
  return (first + second).toUpperCase();
}

export function roleLabel(role: 'tenant' | 'landlord'): string {
  return role === 'tenant' ? 'Tenant' : 'Landlord';
}
