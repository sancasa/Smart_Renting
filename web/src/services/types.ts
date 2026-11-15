/**
 * PROPOSED service contract, owned by the frontend lane until the team agrees on it.
 * Nothing here is an implemented backend or contract claim. Do not copy into
 * docs/API.md or shared/ without a team decision (/decide).
 *
 * Authority rule: the server decides roles, allowed actions, totals and transfer
 * status. The frontend renders what it is given and never invents policy.
 */
import type { BaseUnits } from '../lib/money';
import type { Funds } from '../lib/funds';

export type Role = 'tenant' | 'landlord';

/** The five milestones shown to people. Display only; it never lets anyone skip a step. */
export type Stage = 'setup' | 'move_in' | 'active' | 'move_out' | 'settlement';

/**
 * Finer lifecycle code (PROPOSED, pending agreement with the backend and contract owners).
 * Kept separate from payment/job status, which lives on Operation.
 */
export type Phase = 'active' | 'move_out' | 'proposal_submitted' | 'accepted' | 'disputed' | 'settled';

export type AllowedAction =
  | 'request_move_out'
  | 'edit_proposal'
  | 'submit_proposal'
  | 'accept_proposal'
  | 'dispute_proposal';

/** What the lease is waiting for. The frontend maps these codes to plain-language copy. */
export type WaitingFor = 'move_out_proposal' | 'tenant_response' | 'transfer_confirmation' | 'dispute_ruling';

export interface WaitingOn {
  actor: Role | 'system' | 'reviewer';
  awaiting: WaitingFor;
}

export interface Participant {
  id: string;
  role: Role;
  displayName: string;
}

export type OperationStatus = 'pending' | 'succeeded' | 'failed' | 'unknown';

export interface Operation {
  id: string;
  kind: 'refund' | 'landlord_payout';
  amount: BaseUnits;
  status: OperationStatus;
  /** Present only when a real transaction exists. Sample mode never sets it. */
  txHash?: string;
}

export interface ActivityEvent {
  id: string;
  at: string;
  text: string;
}

export interface EvidenceEntry {
  id: string;
  label: string;
  detail: string;
}

export interface Deadline {
  label: string;
  at: string;
}

export interface Lease {
  id: string;
  property: { label: string };
  participants: Participant[];
  viewer: { userId: string; role: Role };
  asset: { symbol: string; displayName: string };
  stage: Stage;
  phase: Phase;
  moveOutRequestedOn?: string;
  /** What THIS viewer may do now. The UI shows actions from this list only. */
  allowedActions: AllowedAction[];
  /** Roles authorized to request move-out, so the overview can say who can start it. */
  moveOutAuthorizedRoles: Role[];
  waitingOn?: WaitingOn;
  funds: Funds;
  operations: Operation[];
  /** Only deadlines supplied by agreed policy. Empty means show none. */
  deadlines: Deadline[];
  events: ActivityEvent[];
  evidence: EvidenceEntry[];
}

export interface Photo {
  kind: 'placeholder';
  alt: string;
  caption: string;
}

export interface PhotoPair {
  moveIn: Photo;
  moveOut: Photo;
  /** Highlight regions are shown only when the backend supplies valid coordinates. */
}

export interface Finding {
  id: string;
  label: string;
  observation: string;
  amount: BaseUnits;
  rationale: string;
  included: boolean;
  photoPair: PhotoPair;
}

export interface OriginalFinding {
  id: string;
  label: string;
  amount: BaseUnits;
  rationale: string;
}

export interface Revision {
  id: string;
  at: string;
  by: string;
  summary: string;
}

export interface ProposalHistory {
  /** Kept separate from human edits. In sample mode this is a fixture, not AI output. */
  original: { source: 'sample_fixture' | 'analysis'; items: OriginalFinding[] };
  revisions: Revision[];
}

export interface Dispute {
  id: string;
  proposalVersion: number;
  itemIds: string[];
  reason: string;
  submittedAt: string;
  status: 'awaiting_review';
  ruling?: string;
}

export interface Proposal {
  id: string;
  version: number;
  status: 'draft' | 'submitted';
  submittedAt?: string;
  submittedBy?: string;
  items: Finding[];
  deductionTotal: BaseUnits;
  refundAmount: BaseUnits;
  history: ProposalHistory;
  dispute?: Dispute;
}

export interface DraftEdit {
  itemId: string;
  included?: boolean;
  amount?: BaseUnits;
  rationale?: string;
}

export interface DisputeInput {
  itemIds: string[];
  reason: string;
}

export interface DepositService {
  readonly mode: 'sample' | 'live';
  getLease(leaseId: string): Promise<Lease>;
  /** Null when the viewer has no proposal to see (the tenant never sees a landlord draft). */
  getProposal(leaseId: string): Promise<Proposal | null>;
  requestMoveOut(leaseId: string, idempotencyKey: string): Promise<void>;
  saveProposalDraft(leaseId: string, edits: DraftEdit[]): Promise<Proposal>;
  submitProposal(leaseId: string, version: number, idempotencyKey: string): Promise<void>;
  acceptProposal(leaseId: string, version: number, idempotencyKey: string): Promise<void>;
  disputeProposal(leaseId: string, version: number, input: DisputeInput, idempotencyKey: string): Promise<void>;
  /** Optional change notification (sample adapter now; live adapter may poll later). */
  subscribe?(listener: () => void): () => void;
}
