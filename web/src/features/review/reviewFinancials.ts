/**
 * The compact financial summary shown directly beneath the review heading. It reads what the
 * server supplied (funds, proposal, allowedActions) and decides no policy. Every label says what
 * state the money is in: returned and paid are CONFIRMED, processing transfers are not yet either,
 * and held money is still in escrow. A processing transfer is never presented as returned or held.
 * The first item is the lead: the amount this screen is about.
 */
import { reconcileFunds } from '../../lib/funds';
import { formatUsd, isPositive } from '../../lib/money';
import type { Lease, Proposal } from '../../services/types';

export interface FinancialItem {
  id: string;
  label: string;
  value: string;
  /** Plain-language state under the value, such as "Confirmed" or "Processing, not yet returned". */
  note?: string;
  processing?: boolean;
  /** Money that reached someone, shown in the brand color. */
  outcome?: boolean;
}

export interface ReviewFinancials {
  /** The lead item comes first. */
  items: FinancialItem[];
  /** The hold rule in one sentence, when something is held. */
  note?: string;
}

function nameOf(lease: Lease, role: 'tenant' | 'landlord'): string {
  return lease.participants.find((p) => p.role === role)?.displayName ?? role;
}

/** Returns null when the amounts do not add up to the deposit: we never show numbers we cannot reconcile. */
export function reviewFinancials(lease: Lease, proposal: Proposal): ReviewFinancials | null {
  const funds = lease.funds;
  if (!reconcileFunds(funds).ok) return null;

  const viewer = lease.viewer.role;
  const tenant = nameOf(lease, 'tenant');
  const landlord = nameOf(lease, 'landlord');
  const deductions = proposal.deductionTotal;
  const hasDeductions = isPositive(deductions);

  const original: FinancialItem = { id: 'original', label: 'Original deposit', value: formatUsd(funds.original) };
  const held: FinancialItem = { id: 'held', label: 'Still held', value: formatUsd(funds.held) };
  const holdNote = hasDeductions
    ? `${formatUsd(deductions)} stays held until ${viewer === 'tenant' ? 'you accept' : `${tenant} accepts`} or a dispute is resolved.`
    : undefined;

  /** The tenant's refund: "returned" only once confirmed, "processing" until then. */
  const refund = (): FinancialItem | null => {
    if (isPositive(funds.returnedConfirmed)) {
      return {
        id: 'returned',
        label: viewer === 'tenant' ? 'Returned to you' : `Returned to ${tenant}`,
        value: formatUsd(funds.returnedConfirmed),
        note: 'Confirmed',
      };
    }
    if (isPositive(funds.returnedPending)) {
      return {
        id: 'refund-processing',
        label: viewer === 'tenant' ? 'Refund to you' : `Refund to ${tenant}`,
        value: formatUsd(funds.returnedPending),
        note: 'Processing, not yet returned',
        processing: true,
      };
    }
    return null;
  };
  const present = (...items: (FinancialItem | null)[]): FinancialItem[] => items.filter((i): i is FinancialItem => i !== null);

  // Landlord draft: nothing has moved yet.
  if (proposal.status === 'draft') {
    return {
      items: [
        { id: 'deductions', label: 'Proposed deductions', value: formatUsd(deductions) },
        original,
        {
          id: 'refund',
          label: `Refund to ${tenant}`,
          value: formatUsd(proposal.refundAmount),
          note: 'Starts when you submit',
        },
      ],
      note: hasDeductions ? holdNote : 'No deductions, so nothing stays held.',
    };
  }

  switch (lease.phase) {
    case 'settled': {
      const paid = funds.paidToLandlordConfirmed;
      if (viewer === 'landlord' && isPositive(paid)) {
        return {
          items: [
            { id: 'paid', label: 'Paid to you', value: formatUsd(paid), note: 'Confirmed', outcome: true },
            ...present(original, refund(), held),
          ],
        };
      }
      const returned = refund();
      return {
        items: [
          returned ? { ...returned, outcome: true } : { id: 'returned', label: 'Returned', value: formatUsd('0') },
          original,
          {
            id: 'deductions',
            label: 'Deductions',
            value: formatUsd(paid),
            note: isPositive(paid) ? `Paid to ${landlord}` : undefined,
          },
          held,
        ],
      };
    }

    case 'accepted':
      return {
        items: [
          {
            id: 'payout-processing',
            label: viewer === 'landlord' ? 'Payment to you' : `Payment to ${landlord}`,
            value: formatUsd(funds.paidToLandlordPending),
            note: 'Processing, not yet paid',
            processing: true,
          },
          ...present(original, refund(), held),
        ],
      };

    case 'disputed':
      return {
        items: [{ id: 'held', label: 'Held until the dispute is resolved', value: formatUsd(funds.held) }, ...present(original, refund())],
      };

    default: {
      if (!hasDeductions) {
        const processing = refund();
        return {
          items: [
            processing ?? { id: 'refund', label: 'Refund', value: formatUsd(proposal.refundAmount) },
            original,
            { id: 'deductions', label: 'Deductions', value: formatUsd('0') },
          ],
          note: 'No deductions, so nothing stays held.',
        };
      }
      return {
        items: [{ id: 'deductions', label: 'Deductions under review', value: formatUsd(deductions) }, ...present(original, refund(), held)],
        note: holdNote,
      };
    }
  }
}
