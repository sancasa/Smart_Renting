import { useId } from 'react';
import type { Lease, Proposal } from '../../services/types';
import { reviewFinancials } from './reviewFinancials';

/**
 * Compact financial summary directly beneath the page heading. The lead amount comes first;
 * on narrow screens the items stack as plain rows. If the server's amounts do not add up we
 * refuse to show them rather than display numbers we cannot stand behind.
 */
export function FinancialSummary({ lease, proposal }: { lease: Lease; proposal: Proposal }) {
  const titleId = useId();
  const data = reviewFinancials(lease, proposal);

  if (!data) {
    return (
      <section role="alert" className="callout callout--problem">
        <h2 className="callout__title">These amounts can't be shown</h2>
        <p>They don't add up to the original deposit, so we won't display numbers we can't stand behind. Reload to try again.</p>
      </section>
    );
  }

  const leadIndex = Math.max(0, data.items.findIndex((item) => item.lead));
  return (
    <section className="fin" aria-labelledby={titleId}>
      <h2 id={titleId} className="sr-only">
        Financial summary
      </h2>
      <dl className="fin__items">
        {data.items.map((item, index) => (
          <div key={item.id} className={`fin__item${index === leadIndex ? ' fin__item--lead' : ''}`}>
            <dt className="fin__label">{item.label}</dt>
            <dd className={`fin__value${item.outcome ? ' fin__value--outcome' : ''}`}>{item.value}</dd>
            {item.note && <dd className={`fin__note${item.processing ? ' fin__note--processing' : ''}`}>{item.note}</dd>}
          </div>
        ))}
      </dl>
      {data.note && <p className="fin__footnote">{data.note}</p>}
    </section>
  );
}
