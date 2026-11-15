import { reconcileFunds } from '../lib/funds';
import { formatUsd, isPositive } from '../lib/money';
import type { Lease, Proposal } from '../services/types';

function Row({ label, value, status, tone }: { label: string; value: string; status?: string; tone?: 'strong' | 'pending' }) {
  return (
    <div className={`money__row${tone ? ` money__row--${tone}` : ''}`}>
      <dt className="money__label">
        {label}
        {status && <span className="money__status">{status}</span>}
      </dt>
      <dd className="money__value">{value}</dd>
    </div>
  );
}

/**
 * Where the deposit is right now. Every unit sits in exactly one row: confirmed returned,
 * confirmed paid, in progress, or still held. Pending transfers are never counted as
 * returned or as held. If the server's amounts do not add up, we refuse to show them.
 */
export function MoneySummary({ lease, headingId }: { lease: Lease; headingId: string }) {
  const { funds } = lease;
  const viewer = lease.viewer.role;
  const reconciliation = reconcileFunds(funds);
  const landlordName = lease.participants.find((p) => p.role === 'landlord')?.displayName ?? 'the landlord';

  if (!reconciliation.ok) {
    return (
      <section className="surface money" aria-labelledby={headingId}>
        <h2 id={headingId} className="section-title">
          Deposit amounts
        </h2>
        <div role="alert" className="callout callout--problem">
          <p className="callout__title">These amounts can't be shown</p>
          <p>They don't add up to the original deposit, so we won't display numbers we can't stand behind. Reload to try again.</p>
        </div>
      </section>
    );
  }

  const hasPending = isPositive(funds.returnedPending) || isPositive(funds.paidToLandlordPending);
  const showPaid = isPositive(funds.paidToLandlordConfirmed);

  return (
    <section className="surface money" aria-labelledby={headingId}>
      <h2 id={headingId} className="section-title">
        Deposit amounts
      </h2>
      <dl className="money__rows">
        <Row label="Original deposit" value={formatUsd(funds.original)} />
        <Row
          label={viewer === 'tenant' ? 'Returned to you' : 'Returned to tenant'}
          status={isPositive(funds.returnedConfirmed) ? 'Confirmed' : undefined}
          value={formatUsd(funds.returnedConfirmed)}
        />
        {showPaid && (
          <Row
            label={viewer === 'landlord' ? 'Paid to you' : `Paid to ${landlordName}`}
            status="Confirmed"
            value={formatUsd(funds.paidToLandlordConfirmed)}
          />
        )}
        {isPositive(funds.returnedPending) && (
          <Row
            tone="pending"
            label={viewer === 'tenant' ? 'Refund to you' : 'Refund to tenant'}
            status="Processing, not yet returned"
            value={formatUsd(funds.returnedPending)}
          />
        )}
        {isPositive(funds.paidToLandlordPending) && (
          <Row
            tone="pending"
            label={viewer === 'landlord' ? 'Payment to you' : `Payment to ${landlordName}`}
            status="Processing, not yet paid"
            value={formatUsd(funds.paidToLandlordPending)}
          />
        )}
        <Row tone="strong" label="Still held" value={formatUsd(funds.held)} />
      </dl>
      {hasPending && (
        <p className="small muted">Processing amounts are counted once they are confirmed. They are not included in returned or held.</p>
      )}
    </section>
  );
}

/** Proposal-based split shown on the review page before the money has fully moved. */
export function RefundBreakdown({ lease, proposal, headingId }: { lease: Lease; proposal: Proposal; headingId: string }) {
  const viewer = lease.viewer.role;
  const tenant = lease.participants.find((p) => p.role === 'tenant')?.displayName ?? 'the tenant';
  const { funds } = lease;
  const refundState = isPositive(funds.returnedConfirmed)
    ? 'Confirmed'
    : isPositive(funds.returnedPending)
      ? 'Processing, not yet returned'
      : proposal.status === 'draft'
        ? 'Starts when you submit'
        : undefined;
  const hasDeductions = isPositive(proposal.deductionTotal);

  return (
    <section className="surface money" aria-labelledby={headingId}>
      <h2 id={headingId} className="section-title">
        Refund breakdown
      </h2>
      <dl className="money__rows">
        <Row label="Original deposit" value={formatUsd(lease.funds.original)} />
        <Row label={proposal.status === 'draft' ? 'Proposed deductions' : 'Deductions'} value={formatUsd(proposal.deductionTotal)} />
        <Row
          tone="strong"
          label={viewer === 'tenant' ? 'Refund to you' : `Refund to ${tenant}`}
          status={refundState}
          value={formatUsd(proposal.refundAmount)}
        />
      </dl>
      <p className="small muted">
        {hasDeductions
          ? `${formatUsd(proposal.deductionTotal)} stays held until ${viewer === 'tenant' ? 'you accept or dispute' : `${tenant} accepts or disputes`}.`
          : 'No deductions, so nothing stays held and no response is needed.'}
      </p>
    </section>
  );
}
