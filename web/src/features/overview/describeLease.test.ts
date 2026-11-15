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
import { describeLease } from './describeLease';

const now = new Date('2026-11-14T12:00:00.000Z');

function describeFor(scenario: ScenarioId, role: Role) {
  const state = buildScenario(scenario, now);
  return describeLease(selectLease(state, role), selectProposal(state, role));
}

describe('describeLease', () => {
  it('shows nothing urgent while the deposit is simply held, and offers move-out to either participant', () => {
    for (const role of ['tenant', 'landlord'] as const) {
      const result = describeFor('active', role);
      expect(result.headline).toContain('$1,000.00');
      expect(result.detail).toContain('Nothing needs doing');
      expect(result.primary).toMatchObject({ kind: 'request_move_out', label: 'Request move-out' });
    }
  });

  it('hides the move-out action when allowedActions does not include it, and names who can request it', () => {
    const state = buildScenario('active', now);
    const lease = { ...selectLease(state, 'tenant'), allowedActions: [], moveOutAuthorizedRoles: ['landlord' as const] };
    const result = describeLease(lease, null);
    expect(result.primary).toBeUndefined();
    expect(result.detail).toContain('Jordan Lee can request move-out');
  });

  it('gives the landlord one review action and tells the tenant who they are waiting on', () => {
    expect(describeFor('move_out', 'landlord').primary).toMatchObject({ label: 'Review proposed charges' });
    const tenant = describeFor('move_out', 'tenant');
    expect(tenant.primary).toBeUndefined();
    expect(tenant.waiting).toContain('Waiting on Jordan Lee');
  });

  it('offers one review action and says the amount stays held until accepted or a dispute is resolved', () => {
    const tenant = describeFor('refund_pending', 'tenant');
    expect(tenant.headline).toContain('$120.00 in deductions');
    expect(tenant.detail).toBe('This amount stays held until you accept the deductions or a dispute is resolved.');
    expect(tenant.primary).toMatchObject({ label: 'Review deductions', variant: 'primary' });
  });

  it('does not repeat refund amounts the breakdown already shows', () => {
    // Processing versus confirmed is carried by the breakdown rows, not by a second sentence.
    for (const scenario of ['refund_pending', 'refund_confirmed'] as const) {
      expect(describeFor(scenario, 'tenant').detail).not.toContain('$880.00');
    }
  });

  it('tells the landlord the tenant has to respond, without offering an action that is not theirs', () => {
    const landlord = describeFor('refund_confirmed', 'landlord');
    expect(landlord.waiting).toContain('Waiting on Alex Rivera');
    expect(landlord.primary).toMatchObject({ variant: 'secondary' });
    // The actor is named once. The sentence above it only states the hold rule.
    expect(landlord.detail).not.toContain('Alex Rivera');
    expect(landlord.detail).toBe('This amount stays held until the deductions are accepted or a dispute is resolved.');
  });

  it('shows refund processing, and no review task, for a zero-deduction proposal', () => {
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
    for (const role of ['tenant', 'landlord'] as const) {
      const result = describeLease(selectLease(state, role), selectProposal(state, role));
      expect(result.headline).toContain('$1,000.00');
      expect(result.headline).toContain('processing');
      expect(result.primary).toBeUndefined();
      expect(result.waiting).toBe('Waiting for the transfer to confirm.');
    }
  });

  it('shows awaiting review after a dispute, with one waiting line and no duplicate hold sentence', () => {
    const result = describeFor('disputed', 'tenant');
    expect(result.headline).toBe('Awaiting review');
    expect(result.detail).toBe('You disputed 1 deduction.');
    expect(result.waiting).toBe('Waiting for a ruling on the disputed deductions.');
  });

  it('says the payout is waiting for confirmation once the deductions are accepted', () => {
    let state = buildScenario('refund_confirmed', now);
    state = acceptProposal(state, state.proposal.version, now);
    const tenant = describeLease(selectLease(state, 'tenant'), selectProposal(state, 'tenant'));
    const landlord = describeLease(selectLease(state, 'landlord'), selectProposal(state, 'landlord'));
    expect(tenant.detail).toBe('You accepted the deductions.');
    expect(landlord.detail).toBe('Alex Rivera accepted the deductions.');
    expect(tenant.waiting).toBe('Waiting for the transfer to confirm.');
  });

  it('never says disputing releases money, in any state for either role', () => {
    const misleading = /accept or dispute|until they respond|until you respond|accepts or disputes/i;
    const scenarios: ScenarioId[] = ['active', 'move_out', 'refund_pending', 'refund_confirmed', 'disputed', 'settled'];
    for (const scenario of scenarios) {
      for (const role of ['tenant', 'landlord'] as const) {
        const { detail, waiting } = describeFor(scenario, role);
        // "Waiting on X to accept or dispute" names who acts; it does not describe the hold.
        expect(detail).not.toMatch(misleading);
        if (waiting && !/^Waiting on .+ to accept or dispute the deductions\.$/.test(waiting)) {
          expect(waiting).not.toMatch(misleading);
        }
      }
    }
  });

  it('summarizes settlement from confirmed amounts only', () => {
    const tenant = describeFor('settled', 'tenant');
    expect(tenant.headline).toBe('Settlement complete.');
    expect(tenant.detail).toContain('$880.00');
    expect(tenant.detail).toContain('$120.00 was paid to Jordan Lee');
    expect(tenant.detail).toContain('Nothing is held');
  });
});
