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

  it('says the refund is processing and the remainder is held, without calling it returned', () => {
    const tenant = describeFor('refund_pending', 'tenant');
    expect(tenant.headline).toContain('$120.00 in deductions');
    expect(tenant.detail).toContain('$880.00 is processing');
    expect(tenant.detail).not.toContain('confirmed');
    expect(tenant.detail).toContain('$120.00 stays held');
    expect(tenant.primary).toMatchObject({ label: 'Review deductions', variant: 'primary' });
  });

  it('says the refund is confirmed once verified', () => {
    expect(describeFor('refund_confirmed', 'tenant').detail).toContain('$880.00 is confirmed');
  });

  it('tells the landlord the tenant has to respond, without offering an action that is not theirs', () => {
    const landlord = describeFor('refund_confirmed', 'landlord');
    expect(landlord.waiting).toContain('Waiting on Alex Rivera');
    expect(landlord.primary).toMatchObject({ variant: 'secondary' });
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

  it('shows awaiting review with the held amount after a dispute', () => {
    const result = describeFor('disputed', 'tenant');
    expect(result.headline).toBe('Awaiting review');
    expect(result.detail).toContain('$120.00 stays held');
  });

  it('summarizes settlement from confirmed amounts only', () => {
    const tenant = describeFor('settled', 'tenant');
    expect(tenant.headline).toBe('Settlement complete.');
    expect(tenant.detail).toContain('$880.00');
    expect(tenant.detail).toContain('$120.00 was paid to Jordan Lee');
    expect(tenant.detail).toContain('Nothing is held');
  });
});
