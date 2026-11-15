import type { ReactNode } from 'react';

function Chevron() {
  return (
    <svg className="fold__chevron" width="18" height="18" viewBox="0 0 18 18" aria-hidden="true" focusable="false">
      <path d="m4.5 7 4.5 4.5L13.5 7" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

interface DisclosureProps {
  title: string;
  /** Short teaser shown beside the title while closed, such as a count. */
  meta?: string;
  /** Style the teaser as the action word, such as "View timeline". */
  metaIsAction?: boolean;
  /** Add a bottom rule when it is not part of a stacked group. */
  standalone?: boolean;
  children: ReactNode;
}

/**
 * Secondary information behind a native <details>, so it is keyboard and screen reader
 * accessible without script. The label says what is inside and the chevron shows it opens.
 * Opening adds content below the row and never moves anything above it, so the page does not jump.
 */
export function Disclosure({ title, meta, metaIsAction = false, standalone = false, children }: DisclosureProps) {
  return (
    <details className={`fold${standalone ? ' fold--standalone' : ''}`}>
      <summary className="fold__summary">
        <span className="fold__title">{title}</span>
        {meta && <span className={`fold__meta${metaIsAction ? ' fold__meta--action' : ''}`}>{meta}</span>}
        <Chevron />
      </summary>
      <div className="fold__body">{children}</div>
    </details>
  );
}
