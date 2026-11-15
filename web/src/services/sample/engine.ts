/**
 * Sample lease engine. SIMULATED: nothing here is a real payment, wallet action or
 * blockchain transaction, and no AI ran. It exists so the screens can be evaluated
 * before the backend and contract capabilities exist.
 *
 * The engine plays the role the server will play later: it decides allowed actions,
 * validates totals, checks proposal versions and keeps funds in exactly one bucket.
 */
import { compareUnits, formatUsd, isPositive, isZero, subUnits, sumUnits, toBig, type BaseUnits } from '../../lib/money';
import type { Funds } from '../../lib/funds';
import { ServiceError } from '../errors';
import type {
  ActivityEvent,
  AllowedAction,
  DisputeInput,
  DraftEdit,
  EvidenceEntry,
  Finding,
  Lease,
  Operation,
  Participant,
  Phase,
  PhotoPair,
  Proposal,
  Role,
  Stage,
  WaitingOn,
} from '../types';

export const SAMPLE_LEASE_ID = 'sample-lease';
/** $1,000.00 in 6-decimal base units. */
export const ORIGINAL_DEPOSIT: BaseUnits = '1000000000';

export const TENANT: Participant = { id: 'sample-tenant', role: 'tenant', displayName: 'Alex Rivera' };
export const LANDLORD: Participant = { id: 'sample-landlord', role: 'landlord', displayName: 'Jordan Lee' };

export interface SampleState {
  phase: Phase;
  moveOutRequestedOn?: string;
  proposal: Proposal;
  operations: Operation[];
  events: ActivityEvent[];
  counter: number;
}

function placeholderPair(area: string): PhotoPair {
  return {
    moveIn: {
      kind: 'placeholder',
      alt: `Placeholder for the move-in photo of ${area}. No real photo is attached.`,
      caption: 'Placeholder, no photo attached',
    },
    moveOut: {
      kind: 'placeholder',
      alt: `Placeholder for the move-out photo of ${area}. No real photo is attached.`,
      caption: 'Placeholder, no photo attached',
    },
  };
}

function initialFindings(): Finding[] {
  return [
    {
      id: 'finding-wall',
      label: 'Living room, north wall',
      observation: 'Scuff marks beside the light switch that are not visible in the move-in photo.',
      amount: '70000000',
      rationale: 'Repaint the marked section of the wall.',
      included: true,
      photoPair: placeholderPair('the living room north wall'),
    },
    {
      id: 'finding-carpet',
      label: 'Bedroom, carpet',
      observation: 'A stain near the window that is not visible in the move-in photo.',
      amount: '50000000',
      rationale: 'Professional cleaning of the stained area.',
      included: true,
      photoPair: placeholderPair('the bedroom carpet'),
    },
  ];
}

export function initialState(): SampleState {
  const items = initialFindings();
  return {
    phase: 'active',
    proposal: {
      id: 'sample-proposal',
      version: 1,
      status: 'draft',
      items,
      deductionTotal: sumUnits(items.map((i) => i.amount)),
      refundAmount: subUnits(ORIGINAL_DEPOSIT, sumUnits(items.map((i) => i.amount))),
      history: {
        original: {
          source: 'sample_fixture',
          items: items.map((i) => ({ id: i.id, label: i.label, amount: i.amount, rationale: i.rationale })),
        },
        revisions: [],
      },
    },
    operations: [],
    events: [
      { id: 'event-funded', at: '2026-01-05T10:00:00.000Z', text: 'Deposit funded' },
      { id: 'event-movein', at: '2026-01-06T10:00:00.000Z', text: 'Move-in photos saved' },
    ],
    counter: 0,
  };
}

function nextId(s: SampleState, prefix: string): { id: string; counter: number } {
  const counter = s.counter + 1;
  return { id: `${prefix}-${counter}`, counter };
}

function withEvent(s: SampleState, text: string, now: Date): SampleState {
  const { id, counter } = nextId(s, 'event');
  return { ...s, counter, events: [...s.events, { id, at: now.toISOString(), text }] };
}

function requirePhase(s: SampleState, phase: Phase): void {
  if (s.phase !== phase) {
    throw new ServiceError('invalid_state', 'This step is no longer available. Reload to see the current status.');
  }
}

function includedTotal(items: Finding[]): BaseUnits {
  return sumUnits(items.filter((i) => i.included).map((i) => i.amount));
}

export function computeFunds(operations: Operation[]): Funds {
  const total = (kind: Operation['kind'], status: Operation['status']) =>
    sumUnits(operations.filter((o) => o.kind === kind && o.status === status).map((o) => o.amount));
  const returnedConfirmed = total('refund', 'succeeded');
  const returnedPending = total('refund', 'pending');
  const paidToLandlordConfirmed = total('landlord_payout', 'succeeded');
  const paidToLandlordPending = total('landlord_payout', 'pending');
  const held = subUnits(
    ORIGINAL_DEPOSIT,
    sumUnits([returnedConfirmed, returnedPending, paidToLandlordConfirmed, paidToLandlordPending]),
  );
  return {
    original: ORIGINAL_DEPOSIT,
    returnedConfirmed,
    returnedPending,
    paidToLandlordConfirmed,
    paidToLandlordPending,
    held,
  };
}

// ---- Actions (each returns a new state or throws a ServiceError) ----

export function requestMoveOut(s: SampleState, now: Date): SampleState {
  requirePhase(s, 'active');
  const date = [now.getFullYear(), String(now.getMonth() + 1).padStart(2, '0'), String(now.getDate()).padStart(2, '0')].join('-');
  return withEvent({ ...s, phase: 'move_out', moveOutRequestedOn: date }, 'Move-out requested', now);
}

export function saveDraft(s: SampleState, edits: DraftEdit[], now: Date): SampleState {
  requirePhase(s, 'move_out');
  if (s.proposal.status !== 'draft') {
    throw new ServiceError('invalid_state', 'The proposal was already submitted.');
  }
  const items = s.proposal.items.map((i) => ({ ...i }));
  const revisions = [...s.proposal.history.revisions];
  let counter = s.counter;
  const record = (summary: string) => {
    counter += 1;
    revisions.push({ id: `revision-${counter}`, at: now.toISOString(), by: LANDLORD.displayName, summary });
  };

  for (const edit of edits) {
    const item = items.find((i) => i.id === edit.itemId);
    if (!item) throw new ServiceError('validation', 'That charge no longer exists. Reload and try again.');
    if (edit.included !== undefined && edit.included !== item.included) {
      item.included = edit.included;
      record(`${edit.included ? 'Restored' : 'Removed'} the charge for ${item.label}.`);
    }
    if (edit.amount !== undefined && edit.amount !== item.amount) {
      if (!/^\d+$/.test(edit.amount)) throw new ServiceError('validation', 'Enter a valid amount.');
      record(`Changed the amount for ${item.label} from ${formatUsd(item.amount)} to ${formatUsd(edit.amount)}.`);
      item.amount = edit.amount;
    }
    if (edit.rationale !== undefined && edit.rationale !== item.rationale) {
      if (edit.rationale.length > 500) throw new ServiceError('validation', 'Keep the reason under 500 characters.');
      item.rationale = edit.rationale;
      record(`Edited the reason for ${item.label}.`);
    }
  }

  const deductionTotal = includedTotal(items);
  if (compareUnits(deductionTotal, ORIGINAL_DEPOSIT) > 0) {
    throw new ServiceError(
      'validation',
      `Charges can't add up to more than the ${formatUsd(ORIGINAL_DEPOSIT)} deposit.`,
    );
  }
  return {
    ...s,
    counter,
    proposal: {
      ...s.proposal,
      items,
      deductionTotal,
      refundAmount: subUnits(ORIGINAL_DEPOSIT, deductionTotal),
      history: { ...s.proposal.history, revisions },
    },
  };
}

export function submitProposal(s: SampleState, version: number, now: Date): SampleState {
  requirePhase(s, 'move_out');
  if (s.proposal.status !== 'draft') throw new ServiceError('invalid_state', 'The proposal was already submitted.');
  if (version !== s.proposal.version) {
    throw new ServiceError('stale_proposal', 'The proposal changed. Reload and review it again.');
  }
  const included = s.proposal.items.filter((i) => i.included);
  if (included.some((i) => i.rationale.trim() === '')) {
    throw new ServiceError('validation', 'Add a reason for every charge before submitting.');
  }
  const deductionTotal = includedTotal(s.proposal.items);
  const refundAmount = subUnits(ORIGINAL_DEPOSIT, deductionTotal);
  let next: SampleState = {
    ...s,
    phase: 'proposal_submitted',
    proposal: {
      ...s.proposal,
      status: 'submitted',
      items: included,
      deductionTotal,
      refundAmount,
      submittedAt: now.toISOString(),
      submittedBy: LANDLORD.displayName,
    },
  };
  // ASSUMPTION (pending contract owner): submitting starts the undisputed refund
  // right away, without waiting for the tenant. Posting and refunding may be atomic
  // or separate; sample mode treats them as one step.
  if (isPositive(refundAmount)) {
    const { id, counter } = nextId(next, 'operation');
    next = {
      ...next,
      counter,
      operations: [...next.operations, { id, kind: 'refund', amount: refundAmount, status: 'pending' }],
    };
  }
  return withEvent(next, 'Proposal submitted', now);
}

export function acceptProposal(s: SampleState, version: number, now: Date): SampleState {
  requirePhase(s, 'proposal_submitted');
  if (s.proposal.status !== 'submitted' || isZero(s.proposal.deductionTotal)) {
    throw new ServiceError('invalid_state', 'There are no deductions to accept.');
  }
  if (version !== s.proposal.version) {
    throw new ServiceError('stale_proposal', 'The proposal changed. Reload and review the new version.');
  }
  const { id, counter } = nextId(s, 'operation');
  const next: SampleState = {
    ...s,
    counter,
    phase: 'accepted',
    operations: [
      ...s.operations,
      { id, kind: 'landlord_payout', amount: s.proposal.deductionTotal, status: 'pending' },
    ],
  };
  return withEvent(next, 'Deductions accepted', now);
}

export function disputeProposal(s: SampleState, version: number, input: DisputeInput, now: Date): SampleState {
  requirePhase(s, 'proposal_submitted');
  if (version !== s.proposal.version) {
    throw new ServiceError('stale_proposal', 'The proposal changed. Reload and review the new version.');
  }
  if (input.itemIds.length === 0 || input.itemIds.some((id) => !s.proposal.items.some((i) => i.id === id))) {
    throw new ServiceError('validation', 'Choose at least one deduction to dispute.');
  }
  const reason = input.reason.trim();
  if (reason === '') throw new ServiceError('validation', 'Explain why you are disputing.');
  if (reason.length > 1000) throw new ServiceError('validation', 'Keep the explanation under 1,000 characters.');
  const { id, counter } = nextId(s, 'dispute');
  const next: SampleState = {
    ...s,
    counter,
    phase: 'disputed',
    proposal: {
      ...s.proposal,
      dispute: {
        id,
        proposalVersion: version,
        itemIds: [...input.itemIds],
        reason,
        submittedAt: now.toISOString(),
        status: 'awaiting_review',
      },
    },
  };
  return withEvent(next, 'Deduction dispute submitted', now);
}

/** SAMPLE-ONLY control: marks every pending transfer as confirmed. */
export function confirmPendingTransfers(s: SampleState, now: Date): SampleState {
  if (!s.operations.some((o) => o.status === 'pending')) return s;
  let next: SampleState = s;
  const operations = s.operations.map((o) => {
    if (o.status !== 'pending') return o;
    next = withEvent(
      next,
      `${o.kind === 'refund' ? 'Refund' : 'Payment to landlord'} of ${formatUsd(o.amount)} confirmed (simulated)`,
      now,
    );
    return { ...o, status: 'succeeded' as const };
  });
  next = { ...next, operations };
  const allDone = operations.every((o) => o.status === 'succeeded');
  const zeroDeductionRefundDone =
    next.phase === 'proposal_submitted' && isZero(next.proposal.deductionTotal) && allDone;
  const acceptedPayoutDone = next.phase === 'accepted' && allDone;
  if (zeroDeductionRefundDone || acceptedPayoutDone) {
    next = withEvent({ ...next, phase: 'settled' }, 'Settlement complete', now);
  }
  return next;
}

/** SAMPLE-ONLY control: publishes a newer proposal version so an old accept becomes stale. */
export function markProposalOutdated(s: SampleState, now: Date): SampleState {
  if (s.phase !== 'proposal_submitted') {
    throw new ServiceError('invalid_state', 'There is no submitted proposal to outdate.');
  }
  const { id, counter } = nextId(s, 'revision');
  return {
    ...s,
    counter,
    proposal: {
      ...s.proposal,
      version: s.proposal.version + 1,
      history: {
        ...s.proposal.history,
        revisions: [
          ...s.proposal.history.revisions,
          { id, at: now.toISOString(), by: LANDLORD.displayName, summary: 'Published a newer proposal version (simulated).' },
        ],
      },
    },
  };
}

// ---- Derived views for a given viewer ----

export function allowedActionsFor(s: SampleState, role: Role): AllowedAction[] {
  switch (s.phase) {
    // ASSUMPTION (pending team agreement): either participant may request move-out.
    case 'active':
      return ['request_move_out'];
    case 'move_out':
      return role === 'landlord' ? ['edit_proposal', 'submit_proposal'] : [];
    case 'proposal_submitted':
      return role === 'tenant' && isPositive(s.proposal.deductionTotal) ? ['accept_proposal', 'dispute_proposal'] : [];
    default:
      return [];
  }
}

function waitingOnFor(s: SampleState, role: Role): WaitingOn | undefined {
  switch (s.phase) {
    case 'move_out':
      return role === 'tenant' ? { actor: 'landlord', awaiting: 'move_out_proposal' } : undefined;
    case 'proposal_submitted':
      if (isZero(s.proposal.deductionTotal)) return { actor: 'system', awaiting: 'transfer_confirmation' };
      return role === 'landlord' ? { actor: 'tenant', awaiting: 'tenant_response' } : undefined;
    case 'accepted':
      return { actor: 'system', awaiting: 'transfer_confirmation' };
    case 'disputed':
      return { actor: 'reviewer', awaiting: 'dispute_ruling' };
    default:
      return undefined;
  }
}

function stageFor(phase: Phase): Stage {
  if (phase === 'active') return 'active';
  if (phase === 'move_out') return 'move_out';
  return 'settlement';
}

function evidenceFor(s: SampleState): EvidenceEntry[] {
  const entries: EvidenceEntry[] = [
    {
      id: 'evidence-movein',
      label: 'Move-in inspection',
      detail: 'Sample record',
    },
  ];
  if (s.phase !== 'active') {
    entries.push({
      id: 'evidence-moveout',
      label: 'Move-out inspection',
      detail: 'Sample record',
    });
  }
  if (s.proposal.status === 'submitted' && s.proposal.submittedAt) {
    entries.push({
      id: 'evidence-proposal',
      label: `Proposal version ${s.proposal.version}`,
      detail: `Submitted by ${s.proposal.submittedBy ?? LANDLORD.displayName}.`,
    });
  }
  return entries;
}

export function selectLease(s: SampleState, viewerRole: Role): Lease {
  const viewer = viewerRole === 'tenant' ? TENANT : LANDLORD;
  return {
    id: SAMPLE_LEASE_ID,
    property: { label: 'Maple Court, Unit 2B' },
    participants: [TENANT, LANDLORD],
    viewer: { userId: viewer.id, role: viewerRole },
    asset: { symbol: 'USDC', displayName: 'test USDC' },
    stage: stageFor(s.phase),
    phase: s.phase,
    moveOutRequestedOn: s.moveOutRequestedOn,
    allowedActions: allowedActionsFor(s, viewerRole),
    // ASSUMPTION (pending team agreement): both participants may request move-out.
    moveOutAuthorizedRoles: ['tenant', 'landlord'],
    waitingOn: waitingOnFor(s, viewerRole),
    funds: computeFunds(s.operations),
    operations: s.operations,
    deadlines: [],
    events: s.events,
    evidence: evidenceFor(s),
  };
}

/** The tenant never receives a landlord draft, and nobody has a proposal before move-out. */
export function selectProposal(s: SampleState, viewerRole: Role): Proposal | null {
  if (s.phase === 'active') return null;
  if (s.proposal.status === 'draft' && viewerRole === 'tenant') return null;
  return s.proposal;
}

export type ScenarioId = 'active' | 'move_out' | 'refund_pending' | 'refund_confirmed' | 'disputed' | 'settled';

export const SCENARIOS: { id: ScenarioId; label: string }[] = [
  { id: 'active', label: 'Active: deposit held' },
  { id: 'move_out', label: 'Move-out: landlord reviewing charges' },
  { id: 'refund_pending', label: 'Proposal submitted: refund processing' },
  { id: 'refund_confirmed', label: 'Proposal submitted: refund confirmed' },
  { id: 'disputed', label: 'Disputed: awaiting review' },
  { id: 'settled', label: 'Settled' },
];

export function buildScenario(id: ScenarioId, now: Date): SampleState {
  let s = initialState();
  if (id === 'active') return s;
  s = requestMoveOut(s, now);
  if (id === 'move_out') return s;
  s = submitProposal(s, s.proposal.version, now);
  if (id === 'refund_pending') return s;
  s = confirmPendingTransfers(s, now);
  if (id === 'refund_confirmed') return s;
  if (id === 'disputed') {
    return disputeProposal(
      s,
      s.proposal.version,
      { itemIds: ['finding-wall'], reason: 'Sample reason: I believe this mark was already there at move-in.' },
      now,
    );
  }
  s = acceptProposal(s, s.proposal.version, now);
  return confirmPendingTransfers(s, now);
}

/** Exposed so tests can assert nothing leaks outside the base-unit domain. */
export function totalHeldPlusMoved(s: SampleState): bigint {
  const f = computeFunds(s.operations);
  return (
    toBig(f.held) +
    toBig(f.returnedConfirmed) +
    toBig(f.returnedPending) +
    toBig(f.paidToLandlordConfirmed) +
    toBig(f.paidToLandlordPending)
  );
}
