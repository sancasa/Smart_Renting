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
    expect(s.label).toBe('Returned to you');
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

  it('puts the $120.00 under review first while the refund is still processing', () => {
    const s = summaryFor('refund_pending', 'tenant');
    expect(s.figure).toBe('$120.00');
    expect(s.label).toBe('Deductions under review');
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
    // Same label for both roles: it names the amount, while the sentence says who acts.
    expect(s.label).toBe('Deductions under review');
    expect(s.tone).toBe('neutral');
    expect(labels(s)).toContain('Returned to Alex Rivera');
  });

  it('keeps the whole remaining balance held during a dispute', () => {
    const s = summaryFor('disputed', 'tenant');
    expect(s.figure).toBe('$120.00');
    // Disputing does not release money, so the label says it stays held until resolution.
    expect(s.label).toBe('Held until the dispute is resolved');
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
    expect(s.label).toBe('Full refund processing');
    expect(s.facts.some((f) => f.label.startsWith('Returned'))).toBe(false);
  });

  it('gives the landlord the amount paid to them as the settled outcome', () => {
    const s = summaryFor('settled', 'landlord');
    expect(s.figure).toBe('$120.00');
    expect(s.label).toBe('Paid to you');
    expect(s.facts.find((f) => f.id === 'returned')?.value).toBe('$880.00');
  });

  it('shows only the held deposit while nothing needs doing', () => {
    const s = summaryFor('active', 'tenant');
    expect(s.figure).toBe('$1,000.00');
    expect(s.label).toBe('Deposit held');
    expect(s.facts).toEqual([]);
  });

  it('refuses to summarize funds that do not add up', () => {
    const state = buildScenario('refund_pending', now);
    const lease = selectLease(state, 'tenant');
    const broken = { ...lease, funds: { ...lease.funds, returnedConfirmed: lease.funds.returnedPending } };
    expect(summarizeDeposit(broken, selectProposal(state, 'tenant'))).toBeNull();
  });

  describe('adapts to how much there is to show', () => {
    it('keeps the full breakdown when several values are meaningful', () => {
      const pending = summaryFor('refund_pending', 'tenant');
      expect(pending.quiet).toBeUndefined();
      expect(pending.facts.map((f) => f.label)).toEqual(['Original deposit', 'Refund to you', 'Still held']);
      const settled = summaryFor('settled', 'tenant');
      expect(settled.quiet).toBeUndefined();
      expect(settled.facts.map((f) => f.label)).toEqual(['Original deposit', 'Deductions']);
    });

    it('shows the original deposit as a quiet line when it is the only supporting value', () => {
      const s = summaryFor('move_out', 'landlord');
      expect(s.facts).toEqual([]);
      expect(s.quiet).toMatchObject({ label: 'Original deposit', value: '$1,000.00' });
    });

    it('has nothing to add at all while the deposit is simply held', () => {
      const s = summaryFor('active', 'tenant');
      expect(s.facts).toEqual([]);
      expect(s.quiet).toBeUndefined();
    });

    it('does not give a $0.00 deductions row its own column', () => {
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
      const processing = summarizeDeposit(selectLease(state, 'tenant'), selectProposal(state, 'tenant'))!;
      expect(processing.facts).toEqual([]);
      expect(processing.quiet).toMatchObject({ label: 'Original deposit', value: '$1,000.00' });
      expect(processing.label).toBe('Full refund processing');
    });
  });
});
