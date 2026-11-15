import { ServiceError } from '../errors';
import type { DepositService } from '../types';

/**
 * Live adapter. NOT IMPLEMENTED: no backend or contract capability exists yet.
 * Every call fails loudly with `integration_blocked` so the UI shows a blocked state.
 * It must never fall back to sample data or pretend an action succeeded.
 * Replace methods one at a time as the backend (Luka) and contract (Pablo) capabilities
 * are agreed and proven.
 */
function blocked(capability: string): never {
  throw new ServiceError(
    'integration_blocked',
    `The ${capability} service isn't connected yet. This screen can't load real data.`,
  );
}

export function createLiveService(): DepositService {
  return {
    mode: 'live',
    getLease: async () => blocked('lease'),
    getProposal: async () => blocked('proposal'),
    requestMoveOut: async () => blocked('lease'),
    saveProposalDraft: async () => blocked('proposal'),
    submitProposal: async () => blocked('proposal'),
    acceptProposal: async () => blocked('response'),
    disputeProposal: async () => blocked('response'),
  };
}
