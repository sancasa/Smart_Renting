import type { ReactNode } from 'react';
import type { DepositSummary } from './summarizeDeposit';

interface DepositSummaryPanelProps {
  summary: DepositSummary;
  /** One plain sentence: what is happening to the deposit. */
  detail: string;
  /** Present when someone or something else has to act. */
  waiting?: string;
  /** The single action for this role and stage. */
  action?: ReactNode;
  headingId: string;
}

/**
 * The one primary surface on the overview: the headline amount and what it means, with the
 * supporting accounting beside it only when there is more than one meaningful value to show.
 * With nothing to add, or just the original deposit, there is no breakdown column: the original
 * deposit becomes a quiet line under the amount, and the sentence and action sit beside it.
 * Figures are separate buckets, so a processing transfer is never shown as returned or as held.
 */
export function DepositSummaryPanel({ summary, detail, waiting, action, headingId }: DepositSummaryPanelProps) {
  const hasBreakdown = summary.facts.length > 0;
  return (
    <section className={`summary${hasBreakdown ? '' : ' summary--lean'}`} aria-labelledby={headingId}>
      <div className="summary__main">
        <div className="summary__lead">
          <h2 id={headingId} className="summary__title">
            <span className="summary__label">{summary.label}</span>
            <span className={`summary__figure summary__figure--${summary.tone}`}>{summary.figure}</span>
          </h2>
          {summary.quiet && (
            <p className="summary__quiet">
              {summary.quiet.label} <span className="amount">{summary.quiet.value}</span>
            </p>
          )}
        </div>
        <p className="summary__detail">{detail}</p>
        {waiting && <p className="summary__waiting">{waiting}</p>}
        {action}
      </div>

      {hasBreakdown && (
        <div className="summary__aside">
          <dl className="facts">
            {summary.facts.map((fact) => (
              <div key={fact.id} className="fact">
                <dt className="fact__label">
                  {fact.label}
                  {fact.note && (
                    <span className={`fact__note${fact.processing ? ' fact__note--processing' : ''}`}>{fact.note}</span>
                  )}
                </dt>
                <dd className="fact__value">{fact.value}</dd>
              </div>
            ))}
          </dl>
          {summary.hasProcessing && (
            <p className="summary__footnote">Processing amounts count once confirmed.</p>
          )}
        </div>
      )}
    </section>
  );
}
