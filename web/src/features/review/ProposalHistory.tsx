import { formatDateTime } from '../../lib/format';
import { formatUsd } from '../../lib/money';
import type { Proposal } from '../../services/types';

/** Original output and every human edit, kept apart and out of the way until asked for. */
export function ProposalHistory({ proposal }: { proposal: Proposal }) {
  const { original, revisions } = proposal.history;
  return (
    <details className="disclosure">
      <summary className="disclosure__summary">View history</summary>
      <div className="disclosure__body">
        <h3>Original draft</h3>
        <p className="small muted">
          {original.source === 'sample_fixture'
            ? 'Sample draft. No AI analysis was run.'
            : 'Prepared by the analysis job before any edits.'}
        </p>
        <ul className="list-rows">
          {original.items.map((item) => (
            <li key={item.id} className="list-rows__item">
              <span>
                <strong>{item.label}</strong>, <span className="amount">{formatUsd(item.amount)}</span>
              </span>
              <span className="small muted">{item.rationale}</span>
            </li>
          ))}
        </ul>

        <h3>Edits</h3>
        {revisions.length === 0 ? (
          <p className="small muted">No edits yet.</p>
        ) : (
          <ul className="list-rows">
            {revisions.map((revision) => (
              <li key={revision.id} className="list-rows__item">
                <span>{revision.summary}</span>
                <span className="small muted">
                  {revision.by}, {formatDateTime(revision.at)}
                </span>
              </li>
            ))}
          </ul>
        )}
      </div>
    </details>
  );
}
