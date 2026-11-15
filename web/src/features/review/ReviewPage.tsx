import { useCallback, useId, useState } from 'react';
import { Link } from 'react-router-dom';
import { formatDate } from '../../lib/format';
import { pluralize } from '../../lib/format';
import { formatUsd } from '../../lib/money';
import type { Lease, Proposal } from '../../services/types';
import { useLeaseContext } from '../lease/LeaseLayout';
import { FindingEditor, FindingView } from './FindingCards';
import { ProposalDetails, ProposalHistory } from './ProposalHistory';
import { FinancialSummary } from './FinancialSummary';
import { DraftSubmit, RespondPanel } from './ReviewActions';

type Mode = 'empty' | 'draft' | 'respond' | 'disputed' | 'settled' | 'readonly';

/** Same underlying data for both roles; the mode only decides which actions appear. */
function reviewMode(lease: Lease, proposal: Proposal | null): Mode {
  if (!proposal) return 'empty';
  if (proposal.status === 'draft') return lease.allowedActions.includes('edit_proposal') ? 'draft' : 'empty';
  if (lease.phase === 'settled') return 'settled';
  if (lease.phase === 'disputed') return 'disputed';
  if (lease.allowedActions.includes('accept_proposal') || lease.allowedActions.includes('dispute_proposal')) return 'respond';
  return 'readonly';
}

function name(lease: Lease, role: 'tenant' | 'landlord') {
  return lease.participants.find((p) => p.role === role)?.displayName ?? role;
}

function pageCopy(mode: Mode, lease: Lease): { title: string; lead: string } {
  const landlord = name(lease, 'landlord');
  switch (mode) {
    case 'draft':
      return {
        title: 'Review proposed charges',
        lead: `Move-out requested${lease.moveOutRequestedOn ? ` ${formatDate(lease.moveOutRequestedOn)}` : ''}. Remove or adjust charges before you submit.`,
      };
    case 'respond':
      return {
        title: 'Review deductions',
        lead: `${landlord} proposed these deductions.`,
      };
    case 'disputed':
      return {
        title: 'Awaiting review',
        lead: 'Waiting for a ruling. The held amount stays held until the dispute is resolved.',
      };
    case 'settled':
      return { title: 'Settlement', lead: 'All transfers are confirmed. Nothing is held.' };
    case 'readonly':
      return {
        title: lease.viewer.role === 'landlord' ? 'Submitted proposal' : 'Proposal',
        lead: lease.viewer.role === 'landlord' ? "You submitted this proposal. It can't be edited." : `${landlord} submitted this proposal. It can't be edited.`,
      };
    case 'empty':
      return {
        title: 'No proposal to review yet',
        lead:
          lease.viewer.role === 'tenant'
            ? 'A proposal appears here after your landlord submits one.'
            : "There's no proposal to review right now.",
      };
  }
}

function BackIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true" focusable="false">
      <path d="M10 3 5 8l5 5" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

export function ReviewPage() {
  const { lease, proposal } = useLeaseContext();
  const findingsId = useId();
  const [dirtyIds, setDirtyIds] = useState<Set<string>>(new Set());
  const onDirtyChange = useCallback((id: string, dirty: boolean) => {
    setDirtyIds((current) => {
      if (current.has(id) === dirty) return current;
      const next = new Set(current);
      if (dirty) next.add(id);
      else next.delete(id);
      return next;
    });
  }, []);

  const mode = reviewMode(lease, proposal);
  const { title, lead } = pageCopy(mode, lease);
  const disputedIds = new Set(proposal?.dispute?.itemIds ?? []);

  return (
    <div className="review">
      <Link className="back-link" to={`/leases/${lease.id}`}>
        <BackIcon />
        Back to overview
      </Link>
      <div className="page-title">
        <h1 className="page-title__heading" tabIndex={-1}>
          {title}
        </h1>
        {/* The header carries the property on wider screens; narrow screens show it here instead. */}
        <p className="review-property">{lease.property.label}</p>
        <p className="lead">{lead}</p>
      </div>

      {mode === 'empty' || !proposal ? null : (
        <>
          <FinancialSummary lease={lease} proposal={proposal} />

          <section className="stack" aria-labelledby={findingsId}>
            <h2 id={findingsId} className="section-title">
              {mode === 'respond' ? 'Proposed charges' : 'Findings'} ({proposal.items.filter((i) => i.included).length})
            </h2>
            {proposal.items.length === 0 ? (
              <p className="surface">No deductions were proposed.</p>
            ) : (
              <ul className="stack">
                {proposal.items.map((item) => (
                  <li key={item.id}>
                    {mode === 'draft' ? (
                      <FindingEditor leaseId={lease.id} item={item} onDirtyChange={onDirtyChange} />
                    ) : (
                      <FindingView item={item} disputed={disputedIds.has(item.id)} />
                    )}
                  </li>
                ))}
              </ul>
            )}
          </section>

          {mode === 'disputed' && proposal.dispute && <DisputeSummary lease={lease} proposal={proposal} />}
          {mode === 'respond' ? (
            // Tenant deduction review: the decision comes right after the charges; the record sits last, collapsed.
            <>
              <RespondPanel lease={lease} proposal={proposal} />
              <ProposalDetails lease={lease} proposal={proposal} />
            </>
          ) : (
            <>
              {mode === 'draft' ? <ProposalHistory proposal={proposal} /> : <ProposalDetails lease={lease} proposal={proposal} />}
              {mode === 'draft' && <DraftSubmit lease={lease} proposal={proposal} hasUnsavedEdits={dirtyIds.size > 0} />}
            </>
          )}
        </>
      )}
    </div>
  );
}

function DisputeSummary({ lease, proposal }: { lease: Lease; proposal: Proposal }) {
  const titleId = useId();
  const dispute = proposal.dispute!;
  const disputed = proposal.items.filter((item) => dispute.itemIds.includes(item.id));
  const who = lease.viewer.role === 'tenant' ? 'You' : name(lease, 'tenant');
  return (
    <section className="surface stack--tight" aria-labelledby={titleId}>
      <h2 id={titleId} className="section-title">
        The dispute
      </h2>
      <p>
        {who} disputed {pluralize(disputed.length, 'deduction')}.
      </p>
      <ul className="list-rows">
        {disputed.map((item) => (
          <li key={item.id} className="list-rows__item">
            {item.label}, <span className="amount">{formatUsd(item.amount)}</span>
          </li>
        ))}
      </ul>
      <dl className="finding__notes">
        <div className="finding__note">
          <dt>Explanation</dt>
          <dd>{dispute.reason}</dd>
        </div>
      </dl>
      <p className="callout">{dispute.ruling ? `Ruling: ${dispute.ruling}` : 'No ruling has been made yet.'}</p>
    </section>
  );
}
