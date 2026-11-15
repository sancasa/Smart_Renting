/**
 * Maps what the server says (phase, allowedActions, waitingOn, funds) to plain-language
 * copy: what is happening to the deposit, who acts next, and what to do.
 * It does not decide policy. A button appears only when allowedActions includes it.
 */
import { formatDate, pluralize } from '../../lib/format';
import { formatUsd, isPositive, isZero } from '../../lib/money';
import type { Lease, Participant, Proposal, Role } from '../../services/types';

export type ActionButton =
  | { kind: 'link'; to: 'review'; label: string; variant: 'primary' | 'secondary' }
  | { kind: 'request_move_out'; label: string; variant: 'primary' | 'secondary' };

export interface NextAction {
  headline: string;
  detail: string;
  /** Present when someone or something else must act. */
  waiting?: string;
  primary?: ActionButton;
}

function party(lease: Lease, role: Role): Participant {
  const found = lease.participants.find((p) => p.role === role);
  if (!found) throw new Error(`Lease has no ${role}`);
  return found;
}

function waitingText(lease: Lease): string | undefined {
  const waiting = lease.waitingOn;
  if (!waiting) return undefined;
  if (waiting.actor === 'system') return 'Waiting for the transfer to confirm.';
  if (waiting.actor === 'reviewer') return 'Waiting for a ruling on the disputed deductions.';
  const name = party(lease, waiting.actor).displayName;
  switch (waiting.awaiting) {
    case 'move_out_proposal':
      return `Waiting on ${name} to review the move-out findings and submit a proposal.`;
    case 'tenant_response':
      return `Waiting on ${name} to accept or dispute the deductions.`;
    default:
      return `Waiting on ${name}.`;
  }
}

export function describeLease(lease: Lease, proposal: Proposal | null): NextAction {
  const viewer = lease.viewer.role;
  const tenant = party(lease, 'tenant');
  const landlord = party(lease, 'landlord');
  const funds = lease.funds;
  const allowed = new Set(lease.allowedActions);
  const deductions = proposal?.deductionTotal ?? '0';
  const joinSentences = (...parts: string[]) => parts.filter(Boolean).join(' ');

  switch (lease.phase) {
    case 'active': {
      const owner = viewer === 'tenant' ? 'Your' : 'The';
      const authorized = lease.moveOutAuthorizedRoles
        .filter((role) => role !== viewer)
        .map((role) => party(lease, role).displayName);
      if (allowed.has('request_move_out')) {
        return {
          headline: `${owner} ${formatUsd(funds.original)} deposit is held.`,
          detail: 'Nothing needs doing right now. When you are ready to move out, you can request it here.',
          primary: { kind: 'request_move_out', label: 'Request move-out', variant: 'secondary' },
        };
      }
      return {
        headline: `${owner} ${formatUsd(funds.original)} deposit is held.`,
        detail: authorized.length
          ? `Nothing needs doing right now. ${authorized.join(' or ')} can request move-out.`
          : 'Nothing needs doing right now.',
      };
    }

    case 'move_out': {
      const since = lease.moveOutRequestedOn ? ` on ${formatDate(lease.moveOutRequestedOn)}` : '';
      if (allowed.has('edit_proposal')) {
        return {
          headline: 'Review the proposed charges.',
          detail: `Move-out was requested${since}. Check each finding, adjust or remove charges, then submit your proposal.`,
          primary: { kind: 'link', to: 'review', label: 'Review proposed charges', variant: 'primary' },
        };
      }
      return {
        headline: 'Move-out is in progress.',
        detail: `Move-out was requested${since}.`,
        waiting: waitingText(lease),
      };
    }

    case 'proposal_submitted': {
      if (isZero(deductions)) {
        const whose = viewer === 'tenant' ? 'Your full' : 'The full';
        return {
          headline: `${whose} refund of ${formatUsd(proposal?.refundAmount ?? funds.original)} is processing.`,
          detail:
            viewer === 'tenant'
              ? `${landlord.displayName} proposed no deductions, so there is nothing for you to review.`
              : `You proposed no deductions, so ${tenant.displayName} does not need to respond.`,
          waiting: waitingText(lease),
        };
      }
      if (allowed.has('accept_proposal')) {
        return {
          headline: `${landlord.displayName} proposed ${formatUsd(deductions)} in deductions.`,
          // Disputing does not release money: the amount stays held until it is accepted or the dispute is resolved.
          detail: 'This amount stays held until you accept the deductions or a dispute is resolved.',
          primary: { kind: 'link', to: 'review', label: 'Review deductions', variant: 'primary' },
        };
      }
      return {
        headline: `${tenant.displayName} is reviewing your proposal.`,
        // The actor is named once, in `waiting`. This sentence only states the hold rule.
        detail: 'This amount stays held until the deductions are accepted or a dispute is resolved.',
        waiting: waitingText(lease),
        primary: { kind: 'link', to: 'review', label: 'View submitted proposal', variant: 'secondary' },
      };
    }

    case 'accepted':
      return {
        headline: viewer === 'tenant' ? 'You accepted the deductions.' : `${tenant.displayName} accepted the deductions.`,
        detail: viewer === 'tenant' ? 'You accepted the deductions.' : `${tenant.displayName} accepted the deductions.`,
        waiting: waitingText(lease),
        primary: { kind: 'link', to: 'review', label: 'View proposal', variant: 'secondary' },
      };

    case 'disputed': {
      const count = proposal?.dispute?.itemIds.length ?? 0;
      const who = viewer === 'tenant' ? 'You disputed' : `${tenant.displayName} disputed`;
      return {
        headline: 'Awaiting review',
        detail: `${who} ${pluralize(count, 'deduction')}.`,
        waiting: waitingText(lease),
        primary: { kind: 'link', to: 'review', label: 'View dispute', variant: 'secondary' },
      };
    }

    case 'settled': {
      const received = formatUsd(funds.returnedConfirmed);
      const paid = isPositive(funds.paidToLandlordConfirmed) ? formatUsd(funds.paidToLandlordConfirmed) : null;
      const detail =
        viewer === 'tenant'
          ? joinSentences(`You received ${received}.`, paid ? `${paid} was paid to ${landlord.displayName}.` : '', 'Nothing is held.')
          : joinSentences(`${tenant.displayName} received ${received}.`, paid ? `You received ${paid}.` : '', 'Nothing is held.');
      return {
        headline: 'Settlement complete.',
        detail,
        primary: { kind: 'link', to: 'review', label: 'View settlement', variant: 'primary' },
      };
    }
  }
}
