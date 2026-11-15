import { ServiceError } from '../errors';
import type { AllowedAction, DepositService, DisputeInput, DraftEdit, Lease, Proposal, Role } from '../types';
import {
  SAMPLE_LEASE_ID,
  SCENARIOS,
  acceptProposal,
  allowedActionsFor,
  buildScenario,
  confirmPendingTransfers,
  disputeProposal,
  initialState,
  markProposalOutdated,
  requestMoveOut,
  saveDraft,
  selectLease,
  selectProposal,
  submitProposal,
  type SampleState,
  type ScenarioId,
} from './engine';

const STORAGE_KEY = 'deposit-keeper.sample.v1';

interface Persisted {
  schema: 1;
  viewerRole: Role;
  state: SampleState;
  seenKeys: string[];
}

export interface SampleControls {
  readonly scenarios: { id: ScenarioId; label: string }[];
  /** Plain-language list of what sample mode assumes, shown in the demo controls. */
  readonly assumptions: string[];
  getViewerRole(): Role;
  setViewerRole(role: Role): void;
  loadScenario(id: ScenarioId): void;
  hasPendingTransfers(): boolean;
  confirmPendingTransfers(): void;
  canMarkProposalOutdated(): boolean;
  markProposalOutdated(): void;
  reset(): void;
  subscribe(listener: () => void): () => void;
}

export interface SampleOptions {
  latencyMs?: number;
  /** Pass null to disable persistence (tests). Defaults to sessionStorage. */
  storage?: Pick<Storage, 'getItem' | 'setItem' | 'removeItem'> | null;
  now?: () => Date;
}

export const SAMPLE_ASSUMPTIONS = [
  'Inspections are treated as complete. Real inspection rules are pending team agreement.',
  'Either participant can request move-out. The real rule comes from the backend.',
  'The landlord can edit charge amounts and reasons. Real policy is pending.',
  'Submitting a proposal starts the undisputed refund. Whether posting and refunding are one step is pending the contract owner.',
  'No deadlines or automatic payouts exist. Sample transfers confirm only when you press the demo control.',
];

function defaultStorage(): SampleOptions['storage'] {
  try {
    return typeof sessionStorage === 'undefined' ? null : sessionStorage;
  } catch {
    return null;
  }
}

function isPersisted(value: unknown): value is Persisted {
  const v = value as Persisted | null;
  return !!v && v.schema === 1 && (v.viewerRole === 'tenant' || v.viewerRole === 'landlord') && !!v.state?.phase && Array.isArray(v.seenKeys);
}

const sleep = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms));

export function createSampleService(options: SampleOptions = {}): { service: DepositService; controls: SampleControls } {
  const latency = options.latencyMs ?? 450;
  const storage = options.storage === undefined ? defaultStorage() : options.storage;
  const now = options.now ?? (() => new Date());

  let viewerRole: Role = 'tenant';
  let state: SampleState = initialState();
  let seenKeys = new Set<string>();
  const listeners = new Set<() => void>();

  // Persisted state is non-sensitive sample data kept in this browser tab only.
  try {
    const raw = storage?.getItem(STORAGE_KEY);
    if (raw) {
      const parsed: unknown = JSON.parse(raw);
      if (isPersisted(parsed)) {
        viewerRole = parsed.viewerRole;
        state = parsed.state;
        seenKeys = new Set(parsed.seenKeys);
      }
    }
  } catch {
    /* ignore corrupt or unavailable storage */
  }

  const persist = () => {
    try {
      const data: Persisted = { schema: 1, viewerRole, state, seenKeys: [...seenKeys] };
      storage?.setItem(STORAGE_KEY, JSON.stringify(data));
    } catch {
      /* storage may be unavailable; sample mode still works for this page view */
    }
  };
  const notify = () => {
    persist();
    listeners.forEach((l) => l());
  };

  const assertLease = (leaseId: string) => {
    if (leaseId !== SAMPLE_LEASE_ID) throw new ServiceError('unauthorized', "You don't have access to this lease.");
  };
  const assertAllowed = (action: AllowedAction) => {
    // Simulated server-side authorization. Hiding a button is never the only check.
    if (!allowedActionsFor(state, viewerRole).includes(action)) {
      throw new ServiceError('unauthorized', "You can't do that right now. Reload to see what's available.");
    }
  };

  /** Applies a keyed action once. A repeated key is acknowledged without a second effect. */
  const applyKeyed = async (key: string, action: AllowedAction, update: (s: SampleState) => SampleState) => {
    await sleep(latency);
    if (seenKeys.has(key)) return;
    assertAllowed(action);
    state = update(state);
    seenKeys.add(key);
    notify();
  };

  const service: DepositService = {
    mode: 'sample',
    async getLease(leaseId): Promise<Lease> {
      await sleep(Math.min(latency, 120));
      assertLease(leaseId);
      return selectLease(state, viewerRole);
    },
    async getProposal(leaseId): Promise<Proposal | null> {
      await sleep(Math.min(latency, 120));
      assertLease(leaseId);
      return selectProposal(state, viewerRole);
    },
    async requestMoveOut(leaseId, key) {
      assertLease(leaseId);
      await applyKeyed(key, 'request_move_out', (s) => requestMoveOut(s, now()));
    },
    async saveProposalDraft(leaseId, edits: DraftEdit[]) {
      assertLease(leaseId);
      await sleep(Math.min(latency, 250));
      assertAllowed('edit_proposal');
      state = saveDraft(state, edits, now());
      notify();
      return state.proposal;
    },
    async submitProposal(leaseId, version, key) {
      assertLease(leaseId);
      await applyKeyed(key, 'submit_proposal', (s) => submitProposal(s, version, now()));
    },
    async acceptProposal(leaseId, version, key) {
      assertLease(leaseId);
      await applyKeyed(key, 'accept_proposal', (s) => acceptProposal(s, version, now()));
    },
    async disputeProposal(leaseId, version, input: DisputeInput, key) {
      assertLease(leaseId);
      await applyKeyed(key, 'dispute_proposal', (s) => disputeProposal(s, version, input, now()));
    },
    subscribe(listener) {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
  };

  const controls: SampleControls = {
    scenarios: SCENARIOS,
    assumptions: SAMPLE_ASSUMPTIONS,
    getViewerRole: () => viewerRole,
    setViewerRole(role) {
      viewerRole = role;
      notify();
    },
    loadScenario(id) {
      state = buildScenario(id, now());
      seenKeys = new Set();
      notify();
    },
    hasPendingTransfers: () => state.operations.some((o) => o.status === 'pending'),
    confirmPendingTransfers() {
      state = confirmPendingTransfers(state, now());
      notify();
    },
    canMarkProposalOutdated: () => state.phase === 'proposal_submitted',
    markProposalOutdated() {
      // Deliberately does not notify: a real server would not refresh an open page, so the
      // reader keeps the old version and a later accept fails with stale_proposal.
      state = markProposalOutdated(state, now());
      persist();
    },
    reset() {
      state = initialState();
      seenKeys = new Set();
      notify();
    },
    subscribe(listener) {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
  };

  return { service, controls };
}
