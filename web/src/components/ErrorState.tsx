import type { ServiceError } from '../services/errors';

/** One place for load failures. Integration-blocked is shown as exactly that, never as fake data. */
export function ErrorState({ error, onRetry }: { error: ServiceError; onRetry: () => void }) {
  const blocked = error.code === 'integration_blocked';
  const noAccess = error.code === 'unauthorized';
  return (
    <div className="stack">
      <div role="alert" className={`callout ${blocked ? 'callout--caution' : 'callout--problem'}`}>
        <h1 className="callout__title">
          {blocked ? 'Not connected yet' : noAccess ? "You can't open this lease" : "This didn't load"}
        </h1>
        <p>{error.message}</p>
      </div>
      {!noAccess && (
        <div className="button-row">
          <button type="button" className="button" onClick={onRetry}>
            Try again
          </button>
        </div>
      )}
    </div>
  );
}
