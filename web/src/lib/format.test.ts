import { describe, expect, it } from 'vitest';
import { initialsOf, roleLabel } from './format';

describe('initialsOf', () => {
  it('uses the first and last name', () => {
    expect(initialsOf('Alex Rivera')).toBe('AR');
    expect(initialsOf('jordan lee')).toBe('JL');
    expect(initialsOf('Mary-Jane Smith')).toBe('MS');
  });
  it('uses the first two letters of a single name', () => {
    expect(initialsOf('Madonna')).toBe('MA');
    expect(initialsOf('Q')).toBe('Q');
  });
  it('copes with spacing, accents and empty names', () => {
    expect(initialsOf('  Zoë   Åberg ')).toBe('ZÅ');
    expect(initialsOf('   ')).toBe('?');
  });
});

describe('roleLabel', () => {
  it('capitalizes the role', () => {
    expect(roleLabel('tenant')).toBe('Tenant');
    expect(roleLabel('landlord')).toBe('Landlord');
  });
});
