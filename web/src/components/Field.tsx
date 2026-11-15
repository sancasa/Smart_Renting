import { useId, type ReactNode } from 'react';

interface ControlProps {
  id: string;
  'aria-describedby'?: string;
  'aria-invalid'?: boolean;
}

interface FieldProps {
  label: string;
  hint?: string;
  /** Shown beside the field, in words, with a recovery hint. */
  error?: string;
  children: (props: ControlProps) => ReactNode;
}

/** A visible label, optional hint, and an error placed right beside the control it describes. */
export function Field({ label, hint, error, children }: FieldProps) {
  const id = useId();
  const hintId = `${id}-hint`;
  const errorId = `${id}-error`;
  const describedBy = [hint ? hintId : null, error ? errorId : null].filter(Boolean).join(' ') || undefined;

  return (
    <div className="field">
      <label className="field__label" htmlFor={id}>
        {label}
      </label>
      {hint && (
        <p id={hintId} className="field__hint">
          {hint}
        </p>
      )}
      {children({ id, 'aria-describedby': describedBy, 'aria-invalid': error ? true : undefined })}
      {error && (
        <p id={errorId} className="field__error" role="alert">
          {error}
        </p>
      )}
    </div>
  );
}
