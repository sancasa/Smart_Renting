import { describe, expect, it } from 'vitest';
import { stepStates } from './lifecycle';

const states = (stage: Parameters<typeof stepStates>[0], complete: boolean) => stepStates(stage, complete).map((s) => s.state);

describe('stepStates', () => {
  it('marks earlier steps done, the current step in progress, and later steps upcoming', () => {
    expect(states('active', false)).toEqual(['done', 'done', 'current', 'upcoming', 'upcoming']);
    expect(states('move_out', false)).toEqual(['done', 'done', 'done', 'current', 'upcoming']);
  });

  it('keeps settlement in progress while funds are unresolved', () => {
    expect(states('settlement', false)).toEqual(['done', 'done', 'done', 'done', 'current']);
  });

  it('shows every step done only when the lease is complete', () => {
    expect(states('settlement', true)).toEqual(['done', 'done', 'done', 'done', 'done']);
  });

  it('keeps the five milestones in order', () => {
    expect(stepStates('setup', false).map((s) => s.label)).toEqual(['Setup', 'Move-in', 'Active', 'Move-out', 'Settlement']);
  });
});
