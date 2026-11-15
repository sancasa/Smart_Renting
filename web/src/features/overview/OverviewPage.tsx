import { useId, useRef, useState, type ReactNode } from 'react';
import { Link } from 'react-router-dom';
import { ConfirmDialog } from '../../components/ConfirmDialog';
import { MilestoneBar } from '../../components/MilestoneBar';
import { formatDate, formatDateTime } from '../../lib/format';
import { newIdempotencyKey } from '../../lib/idempotency';
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
  const me = lease.participants.find((p) => p.id === lease.viewer.userId);

  return (
    <div className="page overview">
      <div className="overview__title">
        <h1 className="overview__property">{lease.property.label}</h1>
        <ul className="overview__people" aria-label="People on this lease">
          {lease.participants.map((person) => (
            <li key={person.id}>
              {person.role === 'tenant' ? 'Tenant' : 'Landlord'}: {person.displayName}
              {person.id === me?.id ? ' (you)' : ''}
            </li>
          ))}
        </ul>
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

      <MilestoneBar stage={lease.stage} complete={lease.phase === 'settled'} />

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

function Chevron() {
  return (
    <svg className="fold__chevron" width="18" height="18" viewBox="0 0 18 18" aria-hidden="true" focusable="false">
      <path d="m4.5 7 4.5 4.5L13.5 7" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function Fold({ title, meta, children }: { title: string; meta: string; children: ReactNode }) {
  return (
    <details className="fold">
      <summary className="fold__summary">
        <span className="fold__title">{title}</span>
        <span className="fold__meta">{meta}</span>
        <Chevron />
      </summary>
      <div className="fold__body">{children}</div>
    </details>
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
      <Fold title="Evidence" meta={`${evidenceCount} ${evidenceCount === 1 ? 'item' : 'items'}`}>
        <ul className="fold__rows">
          {lease.evidence.map((entry) => (
            <li key={entry.id} className="fold__row fold__row--stacked">
              <span className="fold__row-title">{entry.label}</span>
              <span className="fold__row-note">{entry.detail}</span>
            </li>
          ))}
        </ul>
      </Fold>

      <Fold title="Recent activity" meta={events[0]?.text ?? 'Nothing yet'}>
        <p className="fold__note">
          {receipts.length === 0
            ? 'Demo data: these entries are simulated, and there are no transaction receipts to show.'
            : 'Transaction receipts for confirmed transfers are listed below.'}
        </p>
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
      </Fold>
    </div>
  );
}
