import { useId, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import { ConfirmDialog } from '../../components/ConfirmDialog';
import { MilestoneBar } from '../../components/MilestoneBar';
import { MoneySummary } from '../../components/MoneySummary';
import { formatDate, formatDateTime } from '../../lib/format';
import { newIdempotencyKey } from '../../lib/idempotency';
import { useMutation } from '../../services/hooks';
import { useServices } from '../../services/ServiceContext';
import type { Lease } from '../../services/types';
import { useLeaseContext } from '../lease/LeaseLayout';
import { describeLease, type ActionButton } from './describeLease';

export function OverviewPage() {
  const { lease, proposal } = useLeaseContext();
  const next = describeLease(lease, proposal);
  const moneyId = useId();
  const me = lease.participants.find((p) => p.id === lease.viewer.userId);

  return (
    <div className="page stack">
      <div className="page-title">
        <h1 className="page-title__heading">{lease.property.label}</h1>
        <ul className="people" aria-label="People on this lease">
          {lease.participants.map((person) => (
            <li key={person.id}>
              {person.role === 'tenant' ? 'Tenant' : 'Landlord'}: {person.displayName}
              {person.id === me?.id ? ' (you)' : ''}
            </li>
          ))}
        </ul>
      </div>

      <MilestoneBar stage={lease.stage} complete={lease.phase === 'settled'} />

      <div className="overview-grid">
        <NextAction lease={lease} next={next} />
        <MoneySummary lease={lease} headingId={moneyId} />
      </div>

      <Disclosures lease={lease} />
    </div>
  );
}

function NextAction({ lease, next }: { lease: Lease; next: ReturnType<typeof describeLease> }) {
  const titleId = useId();
  const waitingOnly = !next.primary || next.primary.variant === 'secondary';
  return (
    <section className={`surface next-action${next.waiting && waitingOnly ? ' next-action--waiting' : ''}`} aria-labelledby={titleId}>
      <h2 id={titleId} className="next-action__title">
        {next.headline}
      </h2>
      <p className="next-action__detail">{next.detail}</p>
      {next.waiting && <p className="next-action__waiting">{next.waiting}</p>}
      {next.primary && <ActionControl lease={lease} action={next.primary} />}
    </section>
  );
}

function ActionControl({ lease, action }: { lease: Lease; action: ActionButton }) {
  const className = `button${action.variant === 'primary' ? ' button--primary' : ''}`;
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

function Disclosures({ lease }: { lease: Lease }) {
  const events = [...lease.events].sort((a, b) => b.at.localeCompare(a.at));
  const receipts = lease.operations.filter((o) => o.txHash);
  return (
    <div className="stack--tight">
      <details className="disclosure">
        <summary className="disclosure__summary">Evidence</summary>
        <div className="disclosure__body">
          <ul className="list-rows">
            {lease.evidence.map((entry) => (
              <li key={entry.id} className="list-rows__item">
                <span>
                  <strong>{entry.label}</strong>
                </span>
                <span className="small muted">{entry.detail}</span>
              </li>
            ))}
          </ul>
        </div>
      </details>

      <details className="disclosure">
        <summary className="disclosure__summary">Activity</summary>
        <div className="disclosure__body">
          <p className="small muted">
            {receipts.length === 0
              ? 'This is demo data. These entries are simulated, and there are no transaction receipts to show.'
              : 'Transaction receipts for confirmed transfers:'}
          </p>
          {receipts.length > 0 && (
            <ul className="list-rows">
              {receipts.map((op) => (
                <li key={op.id} className="list-rows__item">
                  <span>{op.kind === 'refund' ? 'Refund' : 'Payment to landlord'}</span>
                  <span className="small muted">{op.txHash}</span>
                </li>
              ))}
            </ul>
          )}
          <ul className="list-rows">
            {events.map((event) => (
              <li key={event.id} className="list-rows__item">
                <span>{event.text}</span>
                <span className="small muted">{formatDateTime(event.at)}</span>
              </li>
            ))}
          </ul>
        </div>
      </details>
    </div>
  );
}
