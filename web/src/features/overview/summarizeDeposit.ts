/**
 * Presentation logic for the overview's one deposit summary: which amount is the headline
 * message, and which smaller facts support it. It reads what the server supplied (phase,
 * allowedActions, funds) and decides no policy. Every unit of the deposit stays in exactly
 * one bucket, and the row that duplicates the headline figure is left out of the facts.
 */
import { reconcileFunds } from '../../lib/funds';
import { formatUsd, isPositive, type BaseUnits } from '../../lib/money';
import type { Lease, Proposal, Role } from '../../services/types';

/** outcome: money that reached someone. attention: something needs the viewer. neutral: everything else. */
export type FigureTone = 'outcome' | 'attention' | 'neutral';

export interface Fact {
  id: string;
  label: string;
  value: string;
  /** Small plain-language state under the label, such as "Confirmed" or "Processing". */
  note?: string;
  processing?: boolean;
  /** A zero-value row. It is true but not a meaningful extra value, so it does not earn a breakdown. */
  empty?: boolean;
}

export interface DepositSummary {
  figure: string;
  /** Says what the amount is, shown above it, e.g. "Deductions under review". */
  label: string;
  tone: FigureTone;
  /** The full breakdown. Only present when more than the original deposit is worth showing. */
  facts: Fact[];
  /** When "Original deposit" is the only meaningful supporting value, it is a quiet line under the amount. */
  quiet?: Fact;
  /** Replaces the generic sentence when the headline already says it all (the settled outcome). */
  detail?: string;
  /** True when any transfer is still processing, so the page can explain how it is counted. */
  hasProcessing: boolean;
}

const SETTLED_DETAIL = 'Settlement is complete. Nothing is held.';

function withDetail(summary: DepositSummary, detail: string): DepositSummary {
  return { ...summary, detail };
}

function nameOf(lease: Lease, role: Role): string {
  return lease.participants.find((p) => p.role === role)?.displayName ?? role;
}

/** Returns null when the amounts do not add up to the deposit: we never show numbers we cannot reconcile. */
export function summarizeDeposit(lease: Lease, proposal: Proposal | null): DepositSummary | null {
  const funds = lease.funds;
  if (!reconcileFunds(funds).ok) return null;

  const viewer = lease.viewer.role;
  const tenant = nameOf(lease, 'tenant');
  const landlord = nameOf(lease, 'landlord');
  const deductions: BaseUnits = proposal?.deductionTotal ?? '0';
  const hasProcessing = isPositive(funds.returnedPending) || isPositive(funds.paidToLandlordPending);

  const original: Fact = { id: 'original', label: 'Original deposit', value: formatUsd(funds.original) };
  const held: Fact = { id: 'held', label: 'Still held', value: formatUsd(funds.held) };

  /** The tenant's refund, shown as returned only once confirmed and as processing until then. */
  const refund = (): Fact | null => {
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
  const present = (...facts: (Fact | null)[]): Fact[] => facts.filter((f): f is Fact => f !== null);
  const done = (figure: string, label: string, tone: FigureTone, facts: Fact[]): DepositSummary => {
    // A breakdown is for several meaningful values. With only the original deposit (or a zero row)
    // to add, a quiet line under the amount says it without a separate box or column.
    const meaningful = facts.filter((fact) => fact.id !== 'original' && !fact.empty);
    if (meaningful.length === 0) {
      return { figure, label, tone, facts: [], quiet: facts.find((fact) => fact.id === 'original'), hasProcessing };
    }
    return { figure, label, tone, facts, hasProcessing };
  };

  switch (lease.phase) {
    case 'active':
      return done(formatUsd(funds.original), 'Deposit held', 'neutral', []);

    case 'move_out':
      if (lease.allowedActions.includes('edit_proposal')) {
        return done(formatUsd(deductions), 'Proposed charges to review', 'attention', [original]);
      }
      return done(formatUsd(funds.original), 'Deposit held', 'neutral', []);

    case 'proposal_submitted': {
      if (!isPositive(deductions)) {
        return done(formatUsd(proposal?.refundAmount ?? funds.original), 'Full refund processing', 'neutral', [
          original,
          { id: 'deductions', label: 'Deductions', value: formatUsd('0'), empty: true },
        ]);
      }
      const needsMe = lease.allowedActions.includes('accept_proposal');
      return done(
        formatUsd(deductions),
        'Deductions under review',
        needsMe ? 'attention' : 'neutral',
        present(original, refund(), held),
      );
    }

    case 'accepted':
      return done(
        formatUsd(deductions),
        viewer === 'landlord' ? 'Deductions being paid to you' : `Deductions being paid to ${landlord}`,
        'neutral',
        present(original, refund()),
      );

    case 'disputed':
      return done(formatUsd(funds.held), 'Held until the dispute is resolved', 'neutral', present(original, refund()));

    case 'settled': {
      const paid = funds.paidToLandlordConfirmed;
      if (viewer === 'landlord' && isPositive(paid)) {
        return withDetail(done(formatUsd(paid), 'Paid to you', 'outcome', present(original, refund())), SETTLED_DETAIL);
      }
      return withDetail(
        done(
          formatUsd(funds.returnedConfirmed),
          viewer === 'tenant' ? 'Returned to you' : `Returned to ${tenant}`,
          'outcome',
          [
            original,
            {
              id: 'deductions',
              label: 'Deductions',
              value: formatUsd(paid),
              note: isPositive(paid) ? `Paid to ${landlord}` : undefined,
              empty: !isPositive(paid),
            },
          ],
        ),
        SETTLED_DETAIL,
      );
    }
  }
}
