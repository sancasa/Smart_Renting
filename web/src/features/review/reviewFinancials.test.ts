import { describe, expect, it } from 'vitest';
import {
  acceptProposal,
  buildScenario,
  saveDraft,
  selectLease,
  selectProposal,
  submitProposal,
  type ScenarioId,
} from '../../services/sample/engine';
import type { Role } from '../../services/types';
import { refundStatusSentence, reviewFinancials } from './reviewFinancials';

const now = new Date('2026-11-14T12:00:00.000Z');

function financialsFor(scenario: ScenarioId, role: Role) {
  const state = buildScenario(scenario, now);
  return reviewFinancials(selectLease(state, role), selectProposal(state, role)!)!;
}
const labels = (f: { items: { label: string }[] }) => f.items.map((i) => i.label);

describe('reviewFinancials', () => {
  it('orders the tenant deduction review as deposit, minus deductions, refund', () => {
    const f = financialsFor('refund_pending', 'tenant');
    expect(f.items.map((i) => [i.label, i.value])).toEqual([
      ['Original deposit', '$1,000.00'],
      ['Proposed deductions', '\u2212$120.00'],
      ['Refund to you', '$880.00'],
    ]);
    // The deductions are the amount under decision, so they lead.
    expect(f.items.find((i) => i.lead)?.id).toBe('deductions');
    expect(f.items.find((i) => i.id === 'deductions')?.note).toBe('Held pending resolution');
  });

  it('drops the duplicate "Still held" column in the tenant review', () => {
    for (const scenario of ['refund_pending', 'refund_confirmed'] as const) {
      const f = financialsFor(scenario, 'tenant');
      expect(f.items.some((i) => i.label === 'Still held')).toBe(false);
      expect(f.items).toHaveLength(3);
      expect(f.note).toBeUndefined();
    }
  });

  it('never presents a processing refund as returned', () => {
    const f = financialsFor('refund_pending', 'tenant');
    const refund = f.items.find((i) => i.id === 'refund')!;
    expect(refund).toMatchObject({ label: 'Refund to you', value: '$880.00', note: 'Processing, not yet returned', processing: true });
    expect(f.items.some((i) => /^Returned/.test(i.label) || /^Returned/.test(i.note ?? ''))).toBe(false);
  });

  it('says the refund was returned only once the transfer is confirmed', () => {
    const f = financialsFor('refund_confirmed', 'tenant');
    expect(f.items.find((i) => i.id === 'refund')).toMatchObject({ label: 'Refund to you', note: 'Returned, confirmed' });
    expect(f.items.some((i) => i.processing)).toBe(false);
  });

  it('reports the actual state of a refund transfer that failed or is unknown', () => {
    const state = buildScenario('refund_pending', now);
    for (const [status, note] of [
      ['failed', 'Transfer failed'],
      ['unknown', 'Status not yet confirmed'],
    ] as const) {
      const broken = { ...state, operations: state.operations.map((o) => (o.kind === 'refund' ? { ...o, status } : o)) };
      // funds only count confirmed and processing transfers, so a failed one leaves the money held
      const lease = selectLease(broken, 'tenant');
      const f = reviewFinancials(lease, selectProposal(broken, 'tenant')!)!;
      expect(f.items.find((i) => i.id === 'refund')?.note).toBe(note);
      expect(f.items.find((i) => i.id === 'refund')?.processing).toBe(false);
    }
  });

  it('describes the separate refund accurately beside the decision', () => {
    const sentence = (scenario: ScenarioId) => {
      const state = buildScenario(scenario, now);
      return refundStatusSentence(selectLease(state, 'tenant'), selectProposal(state, 'tenant')!);
    };
    expect(sentence('refund_pending')).toBe('Your $880.00 refund is separate. It is still processing and has not been returned yet.');
    expect(sentence('refund_confirmed')).toBe('Your $880.00 refund is separate and has already been returned.');
  });

  it('keeps the landlord view on its own order, with the hold rule', () => {
    const landlord = financialsFor('refund_confirmed', 'landlord');
    expect(landlord.items[0]).toMatchObject({ label: 'Deductions under review', value: '$120.00' });
    expect(landlord.note).toBe('$120.00 stays held until Alex Rivera accepts or a dispute is resolved.');
    expect(landlord.note).not.toMatch(/accept or dispute/i);
  });

  it('leads with the confirmed refund once settled, with the deposit and deductions as support', () => {
    const f = financialsFor('settled', 'tenant');
    expect(f.items[0]).toMatchObject({ label: 'Returned to you', value: '$880.00', note: 'Confirmed', outcome: true });
    expect(f.items.map((i) => [i.label, i.value])).toEqual([
      ['Returned to you', '$880.00'],
      ['Original deposit', '$1,000.00'],
      ['Deductions', '$120.00'],
      ['Still held', '$0.00'],
    ]);
    expect(f.items.find((i) => i.id === 'deductions')?.note).toBe('Paid to Jordan Lee');
  });

  it('leads with what the landlord received once settled', () => {
    const f = financialsFor('settled', 'landlord');
    expect(f.items[0]).toMatchObject({ label: 'Paid to you', value: '$120.00', outcome: true });
    expect(labels(f)).toContain('Returned to Alex Rivera');
  });

  it('keeps the whole remaining balance held during a dispute', () => {
    const f = financialsFor('disputed', 'tenant');
    expect(f.items[0]).toMatchObject({ label: 'Held until the dispute is resolved', value: '$120.00' });
  });

  it('shows processing payout separately after the deductions are accepted', () => {
    let state = buildScenario('refund_confirmed', now);
    state = acceptProposal(state, state.proposal.version, now);
    const f = reviewFinancials(selectLease(state, 'tenant'), selectProposal(state, 'tenant')!)!;
    expect(f.items[0]).toMatchObject({ label: 'Payment to Jordan Lee', value: '$120.00', processing: true });
    expect(f.items.find((i) => i.id === 'held')?.value).toBe('$0.00');
  });

  it('shows what the landlord is about to send before anything has moved', () => {
    const f = financialsFor('move_out', 'landlord');
    expect(f.items[0]).toMatchObject({ label: 'Proposed deductions', value: '$120.00' });
    expect(f.items.find((i) => i.id === 'refund')).toMatchObject({ label: 'Refund to Alex Rivera', value: '$880.00', note: 'Starts when you submit' });
  });

  it('shows a full refund as processing, with no deductions, for a zero-deduction proposal', () => {
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
    const f = reviewFinancials(selectLease(state, 'tenant'), selectProposal(state, 'tenant')!)!;
    expect(f.items[0]).toMatchObject({ label: 'Refund to you', value: '$1,000.00', processing: true });
    expect(f.note).toBe('No deductions, so nothing stays held.');
  });

  it('refuses to summarize funds that do not add up', () => {
    const state = buildScenario('refund_pending', now);
    const lease = selectLease(state, 'tenant');
    const broken = { ...lease, funds: { ...lease.funds, returnedConfirmed: lease.funds.returnedPending } };
    expect(reviewFinancials(broken, selectProposal(state, 'tenant')!)).toBeNull();
  });
});
