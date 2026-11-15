import { useEffect, useId, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { ConfirmDialog } from '../../components/ConfirmDialog';
import { Field } from '../../components/Field';
import { newIdempotencyKey } from '../../lib/idempotency';
import { formatUsd, isZero } from '../../lib/money';
import type { ServiceError } from '../../services/errors';
import { useMutation } from '../../services/hooks';
import { useServices } from '../../services/ServiceContext';
import { useLeaseContext } from '../lease/LeaseLayout';
import type { Lease, Proposal } from '../../services/types';

function person(lease: Lease, role: 'tenant' | 'landlord') {
  return lease.participants.find((p) => p.role === role)?.displayName ?? role;
}

function Facts({ rows }: { rows: [string, string][] }) {
  return (
    <dl className="dialog__facts">
      {rows.map(([label, value]) => (
        <div key={label} className="dialog__fact">
          <dt>{label}</dt>
          <dd className="amount">{value}</dd>
        </div>
      ))}
    </dl>
  );
}

/** Error text for a failed money action. A stale proposal gets a clear way forward. */
function ActionError({ error, onReview }: { error: ServiceError; onReview: () => void }) {
  return (
    <>
      <p>{error.message}</p>
      {error.code === 'stale_proposal' && (
        <button type="button" className="button button--small" onClick={onReview}>
          Review the new version
        </button>
      )}
    </>
  );
}

/** Landlord: submit the proposal. Shows the exact split before anything is sent. */
export function DraftSubmit({ lease, proposal, hasUnsavedEdits }: { lease: Lease; proposal: Proposal; hasUnsavedEdits: boolean }) {
  const { service } = useServices().bundle;
  const { reloadNow } = useLeaseContext();
  const navigate = useNavigate();
  const titleId = useId();
  const [open, setOpen] = useState(false);
  const { run, pending, error, clearError } = useMutation();
  const key = useRef(newIdempotencyKey());
  const tenant = person(lease, 'tenant');
  const zero = isZero(proposal.deductionTotal);

  const close = () => {
    setOpen(false);
    clearError();
  };

  return (
    <section className="surface stack--tight" aria-labelledby={titleId}>
      <h2 id={titleId} className="section-title">
        Submit proposal
      </h2>
      <p>
        {zero
          ? `With no charges, the full ${formatUsd(proposal.refundAmount)} refund goes to ${tenant}, and they have nothing to respond to.`
          : `Submitting sends ${formatUsd(proposal.refundAmount)} to ${tenant} now and keeps ${formatUsd(proposal.deductionTotal)} held until they accept or dispute.`}{' '}
        After you submit, you can't edit this proposal.
      </p>
      {hasUnsavedEdits && (
        <p role="status" className="callout callout--caution">
          Save your changes before submitting.
        </p>
      )}
      <div className="button-row">
        <button type="button" className="button button--primary" disabled={hasUnsavedEdits} onClick={() => setOpen(true)}>
          Submit proposal
        </button>
      </div>

      <ConfirmDialog
        open={open}
        title="Submit this proposal?"
        confirmLabel="Submit proposal"
        busyLabel="Submitting…"
        busy={pending}
        confirmDisabled={error?.code === 'stale_proposal'}
        error={
          error && (
            <ActionError
              error={error}
              onReview={() => {
                close();
                key.current = newIdempotencyKey();
                void reloadNow();
              }}
            />
          )
        }
        onCancel={close}
        onConfirm={async () => {
          const ok = await run(async () => {
            await service.submitProposal(lease.id, proposal.version, key.current);
            await reloadNow(); // fresh data first, so the next screen never shows the old state
          });
          if (ok) {
            key.current = newIdempotencyKey();
            setOpen(false);
            navigate(`/leases/${lease.id}`);
          }
        }}
      >
        <Facts
          rows={
            zero
              ? [[`Full refund to ${tenant}`, formatUsd(proposal.refundAmount)]]
              : [
                  [`Refund to ${tenant}`, formatUsd(proposal.refundAmount)],
                  ['Held until they respond', formatUsd(proposal.deductionTotal)],
                ]
          }
        />
        <p>
          {zero ? `${tenant} has nothing to respond to. ` : ''}You can't edit the proposal after you submit it.
        </p>
      </ConfirmDialog>
    </section>
  );
}

/** Tenant: accept the exact version, or open the dispute form. */
export function RespondPanel({ lease, proposal }: { lease: Lease; proposal: Proposal }) {
  const { service } = useServices().bundle;
  const { reloadNow } = useLeaseContext();
  const navigate = useNavigate();
  const titleId = useId();
  const [accepting, setAccepting] = useState(false);
  const [disputing, setDisputing] = useState(false);
  const { run, pending, error, clearError } = useMutation();
  const key = useRef(newIdempotencyKey());
  const landlord = person(lease, 'landlord');
  const canDispute = lease.allowedActions.includes('dispute_proposal');
  const canAccept = lease.allowedActions.includes('accept_proposal');

  const closeAccept = () => {
    setAccepting(false);
    clearError();
  };
  const reviewNewVersion = () => {
    closeAccept();
    key.current = newIdempotencyKey();
    void reloadNow();
  };

  if (disputing) {
    return <DisputeForm lease={lease} proposal={proposal} onCancel={() => setDisputing(false)} />;
  }

  return (
    <section className="surface stack--tight" aria-labelledby={titleId}>
      <h2 id={titleId} className="section-title">
        Your response
      </h2>
      <p>
        Your response applies to version {proposal.version} of this proposal. Your refund of {formatUsd(proposal.refundAmount)} is
        separate and doesn't change.
      </p>
      <div className="button-row">
        {canAccept && (
          <button type="button" className="button button--primary" onClick={() => setAccepting(true)}>
            Accept deductions
          </button>
        )}
        {canDispute && (
          <button type="button" className="button" onClick={() => setDisputing(true)}>
            Dispute a deduction
          </button>
        )}
      </div>

      <ConfirmDialog
        open={accepting}
        title="Accept these deductions?"
        confirmLabel="Accept deductions"
        busyLabel="Accepting…"
        busy={pending}
        confirmDisabled={error?.code === 'stale_proposal'}
        error={error && <ActionError error={error} onReview={reviewNewVersion} />}
        onCancel={closeAccept}
        onConfirm={async () => {
          const ok = await run(async () => {
            await service.acceptProposal(lease.id, proposal.version, key.current);
            await reloadNow();
          });
          if (ok) {
            key.current = newIdempotencyKey();
            setAccepting(false);
            navigate(`/leases/${lease.id}`);
          }
        }}
      >
        <p>
          You are accepting {formatUsd(proposal.deductionTotal)} in deductions from version {proposal.version}.
        </p>
        <Facts
          rows={[
            [`Paid to ${landlord} from the held deposit`, formatUsd(proposal.deductionTotal)],
            ['Your refund, unchanged', formatUsd(proposal.refundAmount)],
          ]}
        />
      </ConfirmDialog>
    </section>
  );
}

function DisputeForm({ lease, proposal, onCancel }: { lease: Lease; proposal: Proposal; onCancel: () => void }) {
  const { service } = useServices().bundle;
  const { reloadNow } = useLeaseContext();
  const navigate = useNavigate();
  const titleId = useId();
  const headingRef = useRef<HTMLHeadingElement>(null);
  const [selected, setSelected] = useState<string[]>([]);
  const [reason, setReason] = useState('');
  const [errors, setErrors] = useState<{ items?: string; reason?: string }>({});
  const { run, pending, error } = useMutation();
  const key = useRef(newIdempotencyKey());

  useEffect(() => {
    headingRef.current?.focus();
  }, []);

  const toggle = (id: string) =>
    setSelected((current) => (current.includes(id) ? current.filter((value) => value !== id) : [...current, id]));

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    const next: { items?: string; reason?: string } = {};
    if (selected.length === 0) next.items = 'Choose at least one deduction to dispute.';
    if (reason.trim() === '') next.reason = 'Explain why you disagree, so the reviewer can see your side.';
    setErrors(next);
    if (next.items || next.reason) return;
    const ok = await run(async () => {
      await service.disputeProposal(lease.id, proposal.version, { itemIds: selected, reason }, key.current);
      await reloadNow();
    });
    if (ok) {
      key.current = newIdempotencyKey();
      navigate(`/leases/${lease.id}`);
    }
  };

  return (
    <form className="surface stack--tight" onSubmit={submit} noValidate aria-labelledby={titleId}>
      <h2 id={titleId} className="section-title" tabIndex={-1} ref={headingRef}>
        Dispute a deduction
      </h2>
      <fieldset className="fieldset" aria-describedby={errors.items ? `${titleId}-items-error` : undefined}>
        <legend className="fieldset__legend">Which deductions do you dispute?</legend>
        {proposal.items.map((item) => (
          <label key={item.id} className="choice">
            <input
              className="choice__input"
              type="checkbox"
              checked={selected.includes(item.id)}
              onChange={() => toggle(item.id)}
            />
            <span>
              {item.label}, <span className="amount">{formatUsd(item.amount)}</span>
            </span>
          </label>
        ))}
        {errors.items && (
          <p id={`${titleId}-items-error`} className="field__error" role="alert">
            {errors.items}
          </p>
        )}
      </fieldset>

      <Field label="Why do you dispute them?" error={errors.reason}>
        {(props) => (
          <textarea
            {...props}
            className="field__control"
            rows={4}
            maxLength={1000}
            value={reason}
            onChange={(event) => setReason(event.target.value)}
          />
        )}
      </Field>

      <div className="form-actions">
        <p>Submitting keeps {formatUsd(lease.funds.held)} held until the dispute is resolved.</p>
        {error && (
          <div role="alert" className="callout callout--problem">
            <ActionError
              error={error}
              onReview={() => {
                key.current = newIdempotencyKey();
                void reloadNow();
                onCancel();
              }}
            />
          </div>
        )}
        <div className="button-row">
          <button type="submit" className="button button--primary" disabled={pending} aria-busy={pending}>
            {pending ? 'Submitting…' : 'Submit dispute'}
          </button>
          <button type="button" className="button" disabled={pending} onClick={onCancel}>
            Cancel
          </button>
        </div>
      </div>
    </form>
  );
}
