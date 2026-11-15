import type { Participant } from '../services/types';

/**
 * The other person on the lease. The viewer's own name and role are already shown by the account
 * button, so the page names only the counterparty instead of repeating who is looking.
 */
export function counterpartyOf(participants: Participant[], viewerId: string): Participant | undefined {
  return participants.find((person) => person.id !== viewerId);
}
