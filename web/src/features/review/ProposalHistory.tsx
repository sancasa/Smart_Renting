import { Disclosure } from '../../components/Disclosure';
import { formatDate, formatDateTime } from '../../lib/format';
import { formatUsd } from '../../lib/money';
import type { Lease, Proposal } from '../../services/types';

/** Original output and every human edit, kept apart and out of the way until asked for. */
export function ProposalHistory({ proposal }: { proposal: Proposal }) {
  const { original, revisions } = proposal.history;
  return (
    <Disclosure title="View history" standalone>
      <h3 className="fold__heading">Original draft</h3>
      <p className="small muted">
        {original.source === 'sample_fixture'
          ? 'Sample draft. No AI analysis was run.'
          : 'Prepared by the analysis job before any edits.'}
      </p>
      <ul className="fold__rows">
        {original.items.map((item) => (
          <li key={item.id} className="fold__row fold__row--stacked">
            <span>
              {item.label}, <span className="amount">{formatUsd(item.amount)}</span>
            </span>
            <span className="fold__row-note">{item.rationale}</span>
          </li>
        ))}
      </ul>

      <h3 className="fold__heading">Edits</h3>
      {revisions.length === 0 ? (
        <p className="small muted">No edits yet.</p>
      ) : (
        <ul className="fold__rows">
          {revisions.map((revision) => (
            <li key={revision.id} className="fold__row fold__row--stacked">
              <span>{revision.summary}</span>
              <span className="fold__row-note">
                {revision.by}, {formatDateTime(revision.at)}
              </span>
            </li>
          ))}
        </ul>
      )}
    </Disclosure>
  );
}

/**
 * Proposal version and exact timestamps. They matter for the record (the response binds to this
 * version) but are not what a reader needs first, so they sit behind a disclosure.
 */
export function ProposalDetails({ lease, proposal }: { lease: Lease; proposal: Proposal }) {
  const rows: [string, string][] = [
    ['Version', String(proposal.version)],
    ['Submitted by', proposal.submittedBy ?? 'Not submitted'],
  ];
  if (proposal.submittedAt) rows.push(['Submitted', formatDateTime(proposal.submittedAt)]);
  if (proposal.dispute) rows.push(['Dispute submitted', formatDateTime(proposal.dispute.submittedAt)]);
  if (lease.moveOutRequestedOn) rows.push(['Move-out requested', formatDate(lease.moveOutRequestedOn)]);
  return (
    <Disclosure title="Proposal details" standalone>
      <dl className="details">
        {rows.map(([label, value]) => (
          <div key={label} className="details__row">
            <dt>{label}</dt>
            <dd>{value}</dd>
          </div>
        ))}
      </dl>
    </Disclosure>
  );
}
