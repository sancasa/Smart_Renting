import { useEffect, useState } from 'react';
import { Field } from '../../components/Field';
import { PhotoPair } from '../../components/PhotoPair';
import { formatUsd, parseUsdInput, toInputText } from '../../lib/money';
import { useMutation } from '../../services/hooks';
import { useServices } from '../../services/ServiceContext';
import { useLeaseContext } from '../lease/LeaseLayout';
import type { Finding } from '../../services/types';

/** Read-only finding: matched photos, what was seen, the amount and the reviewed reason. */
export function FindingView({ item, disputed }: { item: Finding; disputed?: boolean }) {
  return (
    <article className="surface finding" aria-label={item.label}>
      <div className="finding__head">
        <h3 className="finding__title">{item.label}</h3>
        <p className="finding__amount amount">{formatUsd(item.amount)}</p>
      </div>
      {disputed && (
        <p>
          <span className="tag tag--problem">Disputed</span>
        </p>
      )}
      <div className="finding__evidence">
        <PhotoPair pair={item.photoPair} />
        <dl className="finding__notes">
          <div className="finding__note">
            <dt>Found</dt>
            <dd>{item.observation}</dd>
          </div>
          <div className="finding__note">
            <dt>Reason</dt>
            <dd>{item.rationale}</dd>
          </div>
        </dl>
      </div>
    </article>
  );
}

interface FindingEditorProps {
  leaseId: string;
  item: Finding;
  onDirtyChange: (id: string, dirty: boolean) => void;
}

/** Landlord draft: edit the amount and reason, or remove a charge that is not supported. */
export function FindingEditor({ leaseId, item, onDirtyChange }: FindingEditorProps) {
  const { service } = useServices().bundle;
  const { reloadNow } = useLeaseContext();
  const [amountText, setAmountText] = useState(toInputText(item.amount));
  const [rationale, setRationale] = useState(item.rationale);
  const [errors, setErrors] = useState<{ amount?: string; rationale?: string }>({});
  const [saved, setSaved] = useState(false);
  const save = useMutation();
  const toggle = useMutation();

  // Re-sync when the server value changes (after a save or a reload).
  useEffect(() => {
    setAmountText(toInputText(item.amount));
    setRationale(item.rationale);
  }, [item.amount, item.rationale]);

  const dirty = amountText !== toInputText(item.amount) || rationale !== item.rationale;
  useEffect(() => {
    onDirtyChange(item.id, dirty);
    return () => onDirtyChange(item.id, false);
  }, [dirty, item.id, onDirtyChange]);

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    const parsed = parseUsdInput(amountText);
    const next: { amount?: string; rationale?: string } = {};
    if (!parsed.ok) next.amount = parsed.error;
    if (rationale.trim() === '') next.rationale = 'Add a reason for this charge.';
    setErrors(next);
    if (!parsed.ok || next.rationale) return;
    const ok = await save.run(async () => {
      await service.saveProposalDraft(leaseId, [{ itemId: item.id, amount: parsed.value, rationale: rationale.trim() }]);
      await reloadNow();
    });
    if (ok) setSaved(true);
  };

  const setIncluded = async (included: boolean) => {
    const ok = await toggle.run(async () => {
      await service.saveProposalDraft(leaseId, [{ itemId: item.id, included }]);
      await reloadNow();
    });
    if (ok) {
      // Discard unsaved edits on a charge that is no longer part of the proposal.
      setAmountText(toInputText(item.amount));
      setRationale(item.rationale);
      setErrors({});
      setSaved(false);
    }
  };

  if (!item.included) {
    return (
      <article className="surface finding finding--removed" aria-label={`${item.label}, removed`}>
        <div className="finding__head">
          <h3 className="finding__title">{item.label}</h3>
          <p className="finding__amount amount">{formatUsd(item.amount)}</p>
        </div>
        <p>
          <span className="tag tag--muted">Removed</span> This charge won't be sent in the proposal.
        </p>
        <PhotoPair pair={item.photoPair} />
        {toggle.error && (
          <p className="field__error" role="alert">
            {toggle.error.message}
          </p>
        )}
        <div className="button-row">
          <button
            type="button"
            className="button"
            disabled={toggle.pending}
            aria-label={`Restore charge for ${item.label}`}
            onClick={() => setIncluded(true)}
          >
            {toggle.pending ? 'Restoring…' : 'Restore charge'}
          </button>
        </div>
      </article>
    );
  }

  return (
    <article className="surface finding" aria-label={item.label}>
      <div className="finding__head">
        <h3 className="finding__title">{item.label}</h3>
        <p className="finding__amount amount">{formatUsd(item.amount)}</p>
      </div>
      <div className="finding__evidence">
        <PhotoPair pair={item.photoPair} />
        <dl className="finding__notes">
          <div className="finding__note">
            <dt>Found</dt>
            <dd>{item.observation}</dd>
          </div>
        </dl>
      </div>

      <form className="stack--tight" onSubmit={submit} noValidate>
        <Field label="Charge amount in dollars" error={errors.amount}>
          {(props) => (
            <input
              {...props}
              className="field__control field__control--short"
              type="text"
              inputMode="decimal"
              autoComplete="off"
              value={amountText}
              onChange={(event) => {
                setAmountText(event.target.value);
                setSaved(false);
              }}
            />
          )}
        </Field>
        <Field label="Reason for this charge" error={errors.rationale}>
          {(props) => (
            <textarea
              {...props}
              className="field__control"
              rows={3}
              maxLength={500}
              value={rationale}
              onChange={(event) => {
                setRationale(event.target.value);
                setSaved(false);
              }}
            />
          )}
        </Field>
        {save.error && (
          <p className="field__error" role="alert">
            {save.error.message}
          </p>
        )}
        {toggle.error && (
          <p className="field__error" role="alert">
            {toggle.error.message}
          </p>
        )}
        <div className="button-row">
          <button
            type="submit"
            className="button"
            disabled={!dirty || save.pending}
            aria-label={`${saved && !dirty ? 'Saved changes to' : 'Save changes to'} ${item.label}`}
          >
            {save.pending ? 'Saving…' : saved && !dirty ? 'Saved' : 'Save changes'}
          </button>
          <button
            type="button"
            className="button"
            disabled={toggle.pending}
            aria-label={`Remove charge for ${item.label}`}
            onClick={() => setIncluded(false)}
          >
            {toggle.pending ? 'Removing…' : 'Remove charge'}
          </button>
        </div>
        {/* Announced to screen readers; sighted users see "Saved" on the button, so no line is reserved. */}
        <p role="status" className="sr-only">
          {saved && !dirty ? `Saved changes to ${item.label}.` : ''}
        </p>
      </form>
    </article>
  );
}
