import { useId, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import { ConfirmDialog } from '../../components/ConfirmDialog';
import { Disclosure } from '../../components/Disclosure';
import { LifecycleStatus } from '../../components/LifecycleTimeline';
import { formatDate, formatDateTime, roleLabel } from '../../lib/format';
import { newIdempotencyKey } from '../../lib/idempotency';
import { counterpartyOf } from '../../lib/participants';
import { useMutation } from '../../services/hooks';
import { useServices } from '../../services/ServiceContext';
import type { Lease } from '../../services/types';
import { useLeaseContext } from '../lease/LeaseLayout';
import { DepositSummaryPanel } from './DepositSummaryPanel';
import { describeLease, type ActionButton } from './describeLease';
import { summarizeDeposit } from './summarizeDeposit';

export function OverviewPage() {
  const { lease, proposal } = useLeaseContext();
  const next = describeLease(lease, proposal);
  const summary = summarizeDeposit(lease, proposal);
  const summaryId = useId();
  const other = counterpartyOf(lease.participants, lease.viewer.userId);

  return (
    <div className="overview">
      <div className="overview__head">
        <div className="overview__title">
          <h1 className="overview__property" tabIndex={-1}>
            {lease.property.label}
          </h1>
          {other && (
            <p className="overview__people">
              {roleLabel(other.role)}: {other.displayName}
            </p>
          )}
        </div>

        <LifecycleStatus stage={lease.stage} complete={lease.phase === 'settled'} />
      </div>

      {summary ? (
        <DepositSummaryPanel
          headingId={summaryId}
          summary={summary}
          detail={summary.detail ?? next.detail}
          waiting={next.waiting}
          action={next.primary && <ActionControl lease={lease} action={next.primary} />}
        />
      ) : (
        <section className="callout callout--problem" role="alert">
          <h2 className="callout__title">These amounts can't be shown</h2>
          <p>They don't add up to the original deposit, so we won't display numbers we can't stand behind. Reload to try again.</p>
        </section>
      )}

      <Folds lease={lease} />
    </div>
  );
}

function ActionControl({ lease, action }: { lease: Lease; action: ActionButton }) {
  const className = `button button--hero${action.variant === 'primary' ? ' button--primary' : ''}`;
  if (action.kind === 'link') {
    return (
      <div className="button-row">
        <Link to={action.to} className={className}>
          {action.label}
        </Link>
      </div>
    );
  }
  return <MoveOutAction lease={lease} label={action.label} className={className} />;
}

function MoveOutAction({ lease, label, className }: { lease: Lease; label: string; className: string }) {
  const { service } = useServices().bundle;
  const { reloadNow } = useLeaseContext();
  const [open, setOpen] = useState(false);
  const { run, pending, error, clearError } = useMutation();
  // One key per user intent: retries reuse it so a retry can never request move-out twice.
  const key = useRef(newIdempotencyKey());
  const today = formatDate(new Date().toISOString());

  const close = () => {
    setOpen(false);
    clearError();
  };

  return (
    <>
      <div className="button-row">
        <button type="button" className={className} onClick={() => setOpen(true)}>
          {label}
        </button>
      </div>
      <ConfirmDialog
        open={open}
        title="Request move-out?"
        confirmLabel="Request move-out"
        busyLabel="Requesting…"
        busy={pending}
        error={error?.message}
        onCancel={close}
        onConfirm={async () => {
          // Wait for fresh data before closing, so the old button never shows after success.
          const ok = await run(async () => {
            await service.requestMoveOut(lease.id, key.current);
            await reloadNow();
          });
          if (ok) {
            key.current = newIdempotencyKey();
            setOpen(false);
          }
        }}
      >
        <p>This records today, {today}, as the move-out request date and starts the move-out process.</p>
      </ConfirmDialog>
    </>
  );
}

/** Evidence and recent activity: open rows, not cards, so the summary stays the one focal surface. */
function Folds({ lease }: { lease: Lease }) {
  // Oldest-first sort is stable, so entries sharing a timestamp keep their recorded order; reversing gives newest first.
  const events = [...lease.events].sort((a, b) => a.at.localeCompare(b.at)).reverse().slice(0, 5);
  const receipts = lease.operations.filter((o) => o.txHash);
  const evidenceCount = lease.evidence.length;
  return (
    <div className="folds">
      <Disclosure title="Evidence" meta={`${evidenceCount} ${evidenceCount === 1 ? 'item' : 'items'}`}>
        <ul className="fold__rows">
          {lease.evidence.map((entry) => (
            <li key={entry.id} className="fold__row fold__row--stacked">
              <span className="fold__row-title">{entry.label}</span>
              <span className="fold__row-note">{entry.detail}</span>
            </li>
          ))}
        </ul>
      </Disclosure>

      <Disclosure title="Recent activity" meta={events[0]?.text ?? 'Nothing yet'}>
        <ul className="fold__rows">
          {events.map((event) => (
            <li key={event.id} className="fold__row">
              <span>{event.text}</span>
              <span className="fold__row-note">{formatDateTime(event.at)}</span>
            </li>
          ))}
          {receipts.map((op) => (
            <li key={op.id} className="fold__row">
              <span>{op.kind === 'refund' ? 'Refund receipt' : 'Payment receipt'}</span>
              <span className="fold__row-note">{op.txHash}</span>
            </li>
          ))}
        </ul>
      </Disclosure>
    </div>
  );
}
