import { useEffect, useRef } from 'react';
import { Outlet, useLocation, useMatch, useNavigationType, useOutletContext, useParams } from 'react-router-dom';
import { ErrorState } from '../../components/ErrorState';
import { SiteTop } from '../../components/SiteTop';
import { useLeaseData, type LeaseData } from '../../services/hooks';

export interface LeaseOutletContext extends LeaseData {
  reload: () => void;
  /** Resolves once fresh data is on screen. Await it before closing a dialog or navigating. */
  reloadNow: () => Promise<void>;
}

export function useLeaseContext(): LeaseOutletContext {
  return useOutletContext<LeaseOutletContext>();
}

/**
 * Loads the lease once for both screens and owns the single centered content container, the
 * sticky top bar, and navigation behavior: after moving between screens the reader lands on the
 * new page heading with keyboard focus there, instead of wherever the old scroll position was.
 */
export function LeaseLayout() {
  const { leaseId = '' } = useParams();
  const { data, error, loading, reload, reloadNow } = useLeaseData(leaseId);
  const location = useLocation();
  const navigationType = useNavigationType();
  const mainRef = useRef<HTMLElement>(null);
  const firstRender = useRef(true);
  // The overview names the property in its main content, so its header stays compact.
  const onOverview = useMatch({ path: '/leases/:leaseId', end: true }) !== null;

  useEffect(() => {
    if (firstRender.current) {
      firstRender.current = false;
      return;
    }
    // A new screen starts at its top. Back/forward keeps the browser's own scroll restoration.
    if (navigationType !== 'POP') window.scrollTo({ top: 0, left: 0, behavior: 'auto' });
    mainRef.current?.querySelector<HTMLElement>('h1')?.focus({ preventScroll: true });
  }, [location.pathname, navigationType]);

  const content = (() => {
    if (error && !data) return <ErrorState error={error} onRetry={reload} />;
    if (!data) return <p role="status">Loading your deposit…</p>;
    const context: LeaseOutletContext = { ...data, reload, reloadNow };
    return (
      <>
        {error && <ErrorState error={error} onRetry={reload} />}
        <Outlet context={context} />
      </>
    );
  })();

  return (
    <>
      <SiteTop lease={data?.lease} showProperty={!onOverview} />
      <main id="main" className="main" tabIndex={-1} ref={mainRef} aria-busy={loading}>
        <div className="page">{content}</div>
      </main>
    </>
  );
}
