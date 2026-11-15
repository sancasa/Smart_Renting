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
 * The one primary surface on the overview: the headline amount and what it means on the
 * left, and the supporting accounting beside it. Figures are separate buckets, so a
 * processing transfer is never presented as returned or as held.
 */
export function DepositSummaryPanel({ summary, detail, waiting, action, headingId }: DepositSummaryPanelProps) {
  const solo = summary.facts.length === 0;
  return (
    <section className={`summary${solo ? ' summary--solo' : ''}`} aria-labelledby={headingId}>
      <div className="summary__main">
        <h2 id={headingId} className="summary__title">
          <span className={`summary__figure summary__figure--${summary.tone}`}>{summary.figure}</span>
          <span className="summary__label">{summary.label}</span>
        </h2>
        <p className="summary__detail">{detail}</p>
        {waiting && <p className="summary__waiting">{waiting}</p>}
        {action}
      </div>

      {!solo && (
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
            <p className="summary__footnote">Processing amounts are not counted as returned or held until they are confirmed.</p>
          )}
        </div>
      )}
    </section>
  );
}
