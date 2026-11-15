import { useEffect, useId, useRef, type ReactNode } from 'react';

interface ConfirmDialogProps {
  open: boolean;
  title: string;
  children: ReactNode;
  confirmLabel: string;
  busyLabel: string;
  busy: boolean;
  /** Disable the confirm action, for example while a stale-version error needs a reload first. */
  confirmDisabled?: boolean;
  /** Plain-language failure shown inside the dialog, beside the action that failed. */
  error?: ReactNode;
  onConfirm: () => void;
  onCancel: () => void;
}

/**
 * Accessible confirmation built on the native modal <dialog>: focus is trapped, Escape
 * cancels, and focus returns to the button that opened it. Cancel comes first in the
 * DOM so keyboard focus starts on the safe choice for a money action.
 */
export function ConfirmDialog({
  open,
  title,
  children,
  confirmLabel,
  busyLabel,
  busy,
  confirmDisabled = false,
  error,
  onConfirm,
  onCancel,
}: ConfirmDialogProps) {
  const ref = useRef<HTMLDialogElement>(null);
  const titleId = useId();

  useEffect(() => {
    const dialog = ref.current;
    if (!dialog) return;
    if (open && !dialog.open) dialog.showModal();
    if (!open && dialog.open) dialog.close();
  }, [open]);

  return (
    <dialog
      ref={ref}
      className="dialog"
      aria-labelledby={titleId}
      onCancel={(event) => {
        event.preventDefault();
        if (!busy) onCancel();
      }}
    >
      {open && (
        <div className="dialog__content">
          <h2 id={titleId} className="dialog__title">
            {title}
          </h2>
          {children}
          {error && (
            <div role="alert" className="callout callout--problem">
              {error}
            </div>
          )}
          <div className="button-row">
            <button type="button" className="button" onClick={onCancel} disabled={busy}>
              Cancel
            </button>
            <button type="button" className="button button--primary" onClick={onConfirm} disabled={busy || confirmDisabled} aria-busy={busy}>
              {busy ? busyLabel : confirmLabel}
            </button>
          </div>
        </div>
      )}
    </dialog>
  );
}
