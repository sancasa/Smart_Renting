import { describe, expect, it } from 'vitest';
import {
  buildScenario,
  saveDraft,
  selectLease,
  selectProposal,
  submitProposal,
  type ScenarioId,
} from '../../services/sample/engine';
import type { Role } from '../../services/types';
import { summarizeDeposit } from './summarizeDeposit';

const now = new Date('2026-11-14T12:00:00.000Z');

function summaryFor(scenario: ScenarioId, role: Role) {
  const state = buildScenario(scenario, now);
  return summarizeDeposit(selectLease(state, role), selectProposal(state, role))!;
}

const labels = (s: { facts: { label: string }[] }) => s.facts.map((f) => f.label);

describe('summarizeDeposit', () => {
  it('makes the amount returned to the tenant the dominant message once settled', () => {
    const s = summaryFor('settled', 'tenant');
    expect(s.figure).toBe('$880.00');
    expect(s.label).toBe('returned to you');
    expect(s.tone).toBe('outcome');
    // The headline already says what was returned, so the sentence does not repeat it.
    expect(s.detail).toBe('Settlement is complete. Nothing is held.');
  });

  it('shows the original deposit and the deductions as supporting facts, without repeating the headline', () => {
    const s = summaryFor('settled', 'tenant');
    expect(s.facts.map((f) => [f.label, f.value])).toEqual([
      ['Original deposit', '$1,000.00'],
      ['Deductions', '$120.00'],
    ]);
    expect(s.facts.find((f) => f.note)?.note).toBe('Paid to Jordan Lee');
    expect(s.facts.some((f) => f.value === '$880.00')).toBe(false);
  });

  it('puts the $120.00 that needs review first while the refund is still processing', () => {
    const s = summaryFor('refund_pending', 'tenant');
    expect(s.figure).toBe('$120.00');
    expect(s.label).toBe('needs your review');
    expect(s.tone).toBe('attention');
    // Processing money is its own row; it is never listed as returned.
    expect(labels(s)).toEqual(['Original deposit', 'Refund to you', 'Still held']);
    expect(s.facts.find((f) => f.id === 'refund-processing')).toMatchObject({
      value: '$880.00',
      processing: true,
    });
    expect(s.facts.find((f) => f.id === 'held')?.value).toBe('$120.00');
    expect(s.hasProcessing).toBe(true);
  });

  it('lists the refund as returned only after it is confirmed', () => {
    const s = summaryFor('refund_confirmed', 'tenant');
    expect(s.figure).toBe('$120.00');
    expect(labels(s)).toEqual(['Original deposit', 'Returned to you', 'Still held']);
    expect(s.facts.find((f) => f.id === 'returned')).toMatchObject({ value: '$880.00', note: 'Confirmed' });
    expect(s.hasProcessing).toBe(false);
  });

  it('tells the landlord whom they are waiting on instead of asking them to review', () => {
    const s = summaryFor('refund_confirmed', 'landlord');
    expect(s.label).toBe("awaiting Alex Rivera's response");
    expect(s.tone).toBe('neutral');
    expect(labels(s)).toContain('Returned to Alex Rivera');
  });

  it('keeps the whole remaining balance held during a dispute', () => {
    const s = summaryFor('disputed', 'tenant');
    expect(s.figure).toBe('$120.00');
    expect(s.label).toBe('held while the dispute is reviewed');
  });

  it('shows a full refund as processing, not returned, for a zero-deduction proposal', () => {
    let state = buildScenario('move_out', now);
    state = saveDraft(
      state,
      [
        { itemId: 'finding-wall', included: false },
        { itemId: 'finding-carpet', included: false },
      ],
      now,
    );
    state = submitProposal(state, state.proposal.version, now);
    const s = summarizeDeposit(selectLease(state, 'tenant'), selectProposal(state, 'tenant'))!;
    expect(s.figure).toBe('$1,000.00');
    expect(s.label).toBe('full refund processing');
    expect(s.facts.some((f) => f.label.startsWith('Returned'))).toBe(false);
  });

  it('gives the landlord the amount paid to them as the settled outcome', () => {
    const s = summaryFor('settled', 'landlord');
    expect(s.figure).toBe('$120.00');
    expect(s.label).toBe('paid to you');
    expect(s.facts.find((f) => f.id === 'returned')?.value).toBe('$880.00');
  });

  it('shows only the held deposit while nothing needs doing', () => {
    const s = summaryFor('active', 'tenant');
    expect(s.figure).toBe('$1,000.00');
    expect(s.label).toBe('deposit held');
    expect(s.facts).toEqual([]);
  });

  it('refuses to summarize funds that do not add up', () => {
    const state = buildScenario('refund_pending', now);
    const lease = selectLease(state, 'tenant');
    const broken = { ...lease, funds: { ...lease.funds, returnedConfirmed: lease.funds.returnedPending } };
    expect(summarizeDeposit(broken, selectProposal(state, 'tenant'))).toBeNull();
  });
});
