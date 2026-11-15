import { useCallback, useId, useState } from 'react';
import { Link } from 'react-router-dom';
import { MoneySummary, RefundBreakdown } from '../../components/MoneySummary';
import { formatDate, formatDateTime } from '../../lib/format';
import { formatUsd } from '../../lib/money';
import type { Lease, Proposal } from '../../services/types';
import { useLeaseContext } from '../lease/LeaseLayout';
import { FindingEditor, FindingView } from './FindingCards';
import { ProposalHistory } from './ProposalHistory';
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

function pageCopy(mode: Mode, lease: Lease, proposal: Proposal | null): { title: string; lead: string } {
  const landlord = name(lease, 'landlord');
  const submitted = proposal?.submittedAt ? formatDate(proposal.submittedAt) : '';
  switch (mode) {
    case 'draft':
      return {
        title: 'Review proposed charges',
        lead: `Move-out was requested${lease.moveOutRequestedOn ? ` on ${formatDate(lease.moveOutRequestedOn)}` : ''}. Remove charges you can't support, or adjust an amount or reason.`,
      };
    case 'respond':
      return {
        title: 'Review deductions',
        lead: `${landlord} submitted version ${proposal?.version} on ${submitted}. Accept all the deductions, or dispute the ones you disagree with.`,
      };
    case 'disputed':
      return {
        title: 'Awaiting review',
        lead: 'The disputed deductions are waiting for a ruling. The held amount stays held until the dispute is resolved.',
      };
    case 'settled':
      return { title: 'Settlement', lead: 'Every transfer is confirmed and nothing is held.' };
    case 'readonly':
      return {
        title: lease.viewer.role === 'landlord' ? 'Submitted proposal' : 'Proposal',
        lead: `Version ${proposal?.version} was submitted on ${submitted}. It can't be edited.`,
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
  const summaryId = useId();
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
  const { title, lead } = pageCopy(mode, lease, proposal);
  const disputedIds = new Set(proposal?.dispute?.itemIds ?? []);

  return (
    <div className="page stack">
      <Link className="back-link" to={`/leases/${lease.id}`}>
        <BackIcon />
        Back to overview
      </Link>
      <div className="page-title">
        <h1 className="page-title__heading">{title}</h1>
        <p className="lead">{lead}</p>
      </div>

      {mode === 'empty' || !proposal ? null : (
        <div className="review-grid">
          <div className="review-grid__summary">
            {lease.phase === 'move_out' || lease.phase === 'proposal_submitted' ? (
              <RefundBreakdown lease={lease} proposal={proposal} headingId={summaryId} />
            ) : (
              <MoneySummary lease={lease} headingId={summaryId} />
            )}
          </div>

          <div className="review-grid__main">
            <section className="stack" aria-labelledby={findingsId}>
              <h2 id={findingsId} className="section-title">
                Findings ({proposal.items.filter((i) => i.included).length})
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

            {mode === 'draft' && <ProposalHistory proposal={proposal} />}
            {mode === 'draft' && <DraftSubmit lease={lease} proposal={proposal} hasUnsavedEdits={dirtyIds.size > 0} />}
            {mode === 'respond' && <RespondPanel lease={lease} proposal={proposal} />}
            {mode === 'disputed' && proposal.dispute && <DisputeSummary lease={lease} proposal={proposal} />}
          </div>
        </div>
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
        {who} submitted this dispute on {formatDateTime(dispute.submittedAt)}, about version {dispute.proposalVersion}.
      </p>
      <ul className="list-rows">
        {disputed.map((item) => (
          <li key={item.id} className="list-rows__item">
            {item.label}, <span className="amount">{formatUsd(item.amount)}</span>
          </li>
        ))}
      </ul>
      <p>
        <strong>Explanation: </strong>
        {dispute.reason}
      </p>
      <p className="callout">{dispute.ruling ? `Ruling: ${dispute.ruling}` : 'No ruling has been made yet.'}</p>
    </section>
  );
}
