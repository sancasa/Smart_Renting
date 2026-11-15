import { describe, expect, it } from 'vitest';
import type { Participant } from '../services/types';
import { counterpartyOf } from './participants';

const tenant: Participant = { id: 't', role: 'tenant', displayName: 'Alex Rivera' };
const landlord: Participant = { id: 'l', role: 'landlord', displayName: 'Jordan Lee' };

describe('counterpartyOf', () => {
  it('gives each person the other party, never themselves', () => {
    expect(counterpartyOf([tenant, landlord], 't')).toBe(landlord);
    expect(counterpartyOf([tenant, landlord], 'l')).toBe(tenant);
  });

  it('has nobody to show when the viewer is alone on the lease', () => {
    expect(counterpartyOf([tenant], 't')).toBeUndefined();
  });
});
