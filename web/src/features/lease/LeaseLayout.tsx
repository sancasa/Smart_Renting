import { useEffect, useRef } from 'react';
import { Outlet, useLocation, useMatch, useOutletContext, useParams } from 'react-router-dom';
import { ErrorState } from '../../components/ErrorState';
import { Header } from '../../components/Header';
import { useLeaseData, type LeaseData } from '../../services/hooks';

export interface LeaseOutletContext extends LeaseData {
  reload: () => void;
  /** Resolves once fresh data is on screen. Await it before closing a dialog or navigating. */
  reloadNow: () => Promise<void>;
}

export function useLeaseContext(): LeaseOutletContext {
  return useOutletContext<LeaseOutletContext>();
}

/** Loads the lease once for both surfaces and keeps the header, focus and loading states consistent. */
export function LeaseLayout() {
  const { leaseId = '' } = useParams();
  const { data, error, loading, reload, reloadNow } = useLeaseData(leaseId);
  const location = useLocation();
  // The overview names the property in its main content, so its header stays compact.
  const onOverview = useMatch({ path: '/leases/:leaseId', end: true }) !== null;
  const mainRef = useRef<HTMLElement>(null);
  const firstRender = useRef(true);

  // After navigating between overview and review, move focus to the new page so keyboard
  // and screen reader users start at the top of it.
  useEffect(() => {
    if (firstRender.current) {
      firstRender.current = false;
      return;
    }
    mainRef.current?.focus();
  }, [location.pathname]);

  if (error && !data) {
    return (
      <>
        <Header />
        <main id="main" className="main" tabIndex={-1} ref={mainRef}>
          <ErrorState error={error} onRetry={reload} />
        </main>
      </>
    );
  }

  if (!data) {
    return (
      <>
        <Header />
        <main id="main" className="main" tabIndex={-1} ref={mainRef}>
          <div className="page">
            <p role="status">Loading your deposit…</p>
          </div>
        </main>
      </>
    );
  }

  const context: LeaseOutletContext = { ...data, reload, reloadNow };
  return (
    <>
      <Header lease={data.lease} showProperty={!onOverview} />
      <main id="main" className="main" tabIndex={-1} ref={mainRef} aria-busy={loading}>
        {error && (
          <div className="page">
            <ErrorState error={error} onRetry={reload} />
          </div>
        )}
        <Outlet context={context} />
      </main>
    </>
  );
}
