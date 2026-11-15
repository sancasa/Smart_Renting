import { describe, expect, it } from 'vitest';
import { reconcileFunds } from '../../lib/funds';
import { ServiceError } from '../errors';
import { SAMPLE_LEASE_ID } from './engine';
import { createSampleService } from './index';

const fixedNow = () => new Date('2026-11-14T12:00:00.000Z');

function setup() {
  return createSampleService({ latencyMs: 0, storage: null, now: fixedNow });
}

async function failure(promise: Promise<unknown>): Promise<ServiceError> {
  try {
    await promise;
  } catch (error) {
    return error as ServiceError;
  }
  throw new Error('Expected the call to fail');
}

describe('sample lease, full-deduction path', () => {
  it('starts with a $1,000 deposit all held', async () => {
    const { service } = setup();
    const lease = await service.getLease(SAMPLE_LEASE_ID);
    expect(lease.funds.original).toBe('1000000000');
    expect(lease.funds.held).toBe('1000000000');
    expect(reconcileFunds(lease.funds)).toEqual({ ok: true });
  });

  it('keeps pending, confirmed and held amounts in separate buckets at every step', async () => {
    const { service, controls } = setup();
    controls.loadScenario('move_out');
    controls.setViewerRole('landlord');
    const draft = (await service.getProposal(SAMPLE_LEASE_ID))!;
    expect(draft.deductionTotal).toBe('120000000');
    expect(draft.refundAmount).toBe('880000000');

    await service.submitProposal(SAMPLE_LEASE_ID, draft.version, 'k-submit');
    let lease = await service.getLease(SAMPLE_LEASE_ID);
    expect(lease.funds.returnedPending).toBe('880000000');
    expect(lease.funds.returnedConfirmed).toBe('0');
    expect(lease.funds.held).toBe('120000000');
    expect(reconcileFunds(lease.funds)).toEqual({ ok: true });

    controls.confirmPendingTransfers();
    lease = await service.getLease(SAMPLE_LEASE_ID);
    expect(lease.funds.returnedConfirmed).toBe('880000000');
    expect(lease.funds.returnedPending).toBe('0');
    expect(lease.funds.held).toBe('120000000');

    controls.setViewerRole('tenant');
    await service.acceptProposal(SAMPLE_LEASE_ID, draft.version, 'k-accept');
    lease = await service.getLease(SAMPLE_LEASE_ID);
    expect(lease.phase).toBe('accepted');
    expect(lease.funds.paidToLandlordPending).toBe('120000000');
    expect(lease.funds.held).toBe('0');
    expect(reconcileFunds(lease.funds)).toEqual({ ok: true });

    controls.confirmPendingTransfers();
    lease = await service.getLease(SAMPLE_LEASE_ID);
    expect(lease.phase).toBe('settled');
    expect(lease.funds.paidToLandlordConfirmed).toBe('120000000');
    expect(lease.funds.returnedConfirmed).toBe('880000000');
    expect(lease.funds.held).toBe('0');
  });
});

describe('zero-deduction path', () => {
  it('skips acceptance but shows the refund as processing before settlement', async () => {
    const { service, controls } = setup();
    controls.loadScenario('move_out');
    controls.setViewerRole('landlord');
    await service.saveProposalDraft(SAMPLE_LEASE_ID, [
      { itemId: 'finding-wall', included: false },
      { itemId: 'finding-carpet', included: false },
    ]);
    const draft = (await service.getProposal(SAMPLE_LEASE_ID))!;
    expect(draft.deductionTotal).toBe('0');
    expect(draft.refundAmount).toBe('1000000000');

    await service.submitProposal(SAMPLE_LEASE_ID, draft.version, 'k-zero');
    let lease = await service.getLease(SAMPLE_LEASE_ID);
    expect(lease.phase).toBe('proposal_submitted');
    expect(lease.funds.returnedPending).toBe('1000000000');
    expect(lease.funds.returnedConfirmed).toBe('0');
    expect(lease.waitingOn).toEqual({ actor: 'system', awaiting: 'transfer_confirmation' });

    controls.setViewerRole('tenant');
    lease = await service.getLease(SAMPLE_LEASE_ID);
    expect(lease.allowedActions).toEqual([]); // no accept or dispute task
    const error = await failure(service.acceptProposal(SAMPLE_LEASE_ID, draft.version, 'k-no'));
    expect(error.code).toBe('unauthorized');

    controls.confirmPendingTransfers();
    lease = await service.getLease(SAMPLE_LEASE_ID);
    expect(lease.phase).toBe('settled');
    expect(lease.funds.returnedConfirmed).toBe('1000000000');
    expect(lease.funds.held).toBe('0');
  });
});

describe('dispute path', () => {
  it('keeps the whole remaining balance held and moves to awaiting review', async () => {
    const { service, controls } = setup();
    controls.loadScenario('refund_confirmed');
    controls.setViewerRole('tenant');
    const proposal = (await service.getProposal(SAMPLE_LEASE_ID))!;
    await service.disputeProposal(
      SAMPLE_LEASE_ID,
      proposal.version,
      { itemIds: ['finding-wall'], reason: 'Already there at move-in.' },
      'k-dispute',
    );
    const lease = await service.getLease(SAMPLE_LEASE_ID);
    expect(lease.phase).toBe('disputed');
    expect(lease.funds.held).toBe('120000000'); // no partial release of undisputed deductions
    expect(lease.funds.paidToLandlordConfirmed).toBe('0');
    expect(lease.waitingOn?.awaiting).toBe('dispute_ruling');
  });

  it('validates the selection and explanation', async () => {
    const { service, controls } = setup();
    controls.loadScenario('refund_confirmed');
    const proposal = (await service.getProposal(SAMPLE_LEASE_ID))!;
    expect((await failure(service.disputeProposal(SAMPLE_LEASE_ID, proposal.version, { itemIds: [], reason: 'x' }, 'a'))).code).toBe('validation');
    expect((await failure(service.disputeProposal(SAMPLE_LEASE_ID, proposal.version, { itemIds: ['finding-wall'], reason: '   ' }, 'b'))).code).toBe('validation');
  });
});

describe('stale proposals and duplicates', () => {
  it('rejects accepting an outdated version', async () => {
    const { service, controls } = setup();
    controls.loadScenario('refund_confirmed');
    const v1 = (await service.getProposal(SAMPLE_LEASE_ID))!.version;
    controls.markProposalOutdated();
    const error = await failure(service.acceptProposal(SAMPLE_LEASE_ID, v1, 'k-stale'));
    expect(error.code).toBe('stale_proposal');
    expect((await service.getLease(SAMPLE_LEASE_ID)).phase).toBe('proposal_submitted');
  });

  it('applies a repeated idempotency key only once', async () => {
    const { service, controls } = setup();
    controls.loadScenario('refund_confirmed');
    const version = (await service.getProposal(SAMPLE_LEASE_ID))!.version;
    await service.acceptProposal(SAMPLE_LEASE_ID, version, 'same-key');
    await service.acceptProposal(SAMPLE_LEASE_ID, version, 'same-key'); // retry
    const lease = await service.getLease(SAMPLE_LEASE_ID);
    expect(lease.operations.filter((o) => o.kind === 'landlord_payout')).toHaveLength(1);
  });

  it('allows retrying a failed action with the same key', async () => {
    const { service, controls } = setup();
    controls.loadScenario('refund_confirmed');
    const version = (await service.getProposal(SAMPLE_LEASE_ID))!.version;
    await failure(service.acceptProposal(SAMPLE_LEASE_ID, version + 1, 'retry-key'));
    await service.acceptProposal(SAMPLE_LEASE_ID, version, 'retry-key');
    expect((await service.getLease(SAMPLE_LEASE_ID)).phase).toBe('accepted');
  });
});

describe('simulated authorization', () => {
  it('lets either role request move-out when allowedActions includes it', async () => {
    for (const role of ['tenant', 'landlord'] as const) {
      const { service, controls } = setup();
      controls.setViewerRole(role);
      const lease = await service.getLease(SAMPLE_LEASE_ID);
      expect(lease.allowedActions).toContain('request_move_out');
      await service.requestMoveOut(SAMPLE_LEASE_ID, `k-${role}`);
      const after = await service.getLease(SAMPLE_LEASE_ID);
      expect(after.phase).toBe('move_out');
      expect(after.moveOutRequestedOn).toBe('2026-11-14');
    }
  });

  it("refuses actions that are not in the viewer's allowed list, regardless of the UI", async () => {
    const { service, controls } = setup();
    controls.loadScenario('move_out');
    controls.setViewerRole('tenant');
    expect((await failure(service.saveProposalDraft(SAMPLE_LEASE_ID, [{ itemId: 'finding-wall', included: false }]))).code).toBe('unauthorized');
    expect((await failure(service.submitProposal(SAMPLE_LEASE_ID, 1, 'k'))).code).toBe('unauthorized');
  });

  it('never shows the landlord draft to the tenant', async () => {
    const { service, controls } = setup();
    controls.loadScenario('move_out');
    controls.setViewerRole('tenant');
    expect(await service.getProposal(SAMPLE_LEASE_ID)).toBeNull();
    controls.setViewerRole('landlord');
    expect((await service.getProposal(SAMPLE_LEASE_ID))?.status).toBe('draft');
  });

  it('rejects a lease the viewer does not belong to', async () => {
    const { service } = setup();
    expect((await failure(service.getLease('someone-elses-lease'))).code).toBe('unauthorized');
  });
});

describe('landlord draft editing', () => {
  it('validates totals and preserves the original draft apart from revisions', async () => {
    const { service, controls } = setup();
    controls.loadScenario('move_out');
    controls.setViewerRole('landlord');
    const tooMuch = await failure(service.saveProposalDraft(SAMPLE_LEASE_ID, [{ itemId: 'finding-wall', amount: '1000000001' }]));
    expect(tooMuch.code).toBe('validation');

    const updated = await service.saveProposalDraft(SAMPLE_LEASE_ID, [{ itemId: 'finding-wall', amount: '50000000', rationale: 'Touch-up only.' }]);
    expect(updated.deductionTotal).toBe('100000000');
    expect(updated.refundAmount).toBe('900000000');
    expect(updated.history.original.items.find((i) => i.id === 'finding-wall')?.amount).toBe('70000000');
    expect(updated.history.revisions).toHaveLength(2);
  });

  it('refuses to submit a charge without a reason', async () => {
    const { service, controls } = setup();
    controls.loadScenario('move_out');
    controls.setViewerRole('landlord');
    await service.saveProposalDraft(SAMPLE_LEASE_ID, [{ itemId: 'finding-wall', rationale: '' }]);
    const error = await failure(service.submitProposal(SAMPLE_LEASE_ID, 1, 'k'));
    expect(error.code).toBe('validation');
  });
});

describe('persistence', () => {
  it('rehydrates saved sample progress', async () => {
    const store = new Map<string, string>();
    const storage = {
      getItem: (k: string) => store.get(k) ?? null,
      setItem: (k: string, v: string) => void store.set(k, v),
      removeItem: (k: string) => void store.delete(k),
    };
    const first = createSampleService({ latencyMs: 0, storage, now: fixedNow });
    first.controls.loadScenario('refund_pending');
    const second = createSampleService({ latencyMs: 0, storage, now: fixedNow });
    const lease = await second.service.getLease(SAMPLE_LEASE_ID);
    expect(lease.funds.returnedPending).toBe('880000000');
  });
});
