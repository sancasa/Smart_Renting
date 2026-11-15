import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom';
import { SiteTop } from './components/SiteTop';
import { LeaseLayout } from './features/lease/LeaseLayout';
import { OverviewPage } from './features/overview/OverviewPage';
import { ReviewPage } from './features/review/ReviewPage';
import { ServiceProvider, useServices } from './services/ServiceContext';

function Home() {
  const { bundle } = useServices();
  return <Navigate to={`/leases/${bundle.defaultLeaseId}`} replace />;
}

function NotFound() {
  return (
    <>
      <SiteTop />
      <main id="main" className="main" tabIndex={-1}>
        <div className="page stack">
          <h1 className="page-title__heading">We can't find that page</h1>
          <p>
            <a href="/">Go to your deposit</a>
          </p>
        </div>
      </main>
    </>
  );
}

export default function App() {
  return (
    <ServiceProvider>
      <BrowserRouter>
        <a className="skip-link" href="#main">
          Skip to main content
        </a>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/leases/:leaseId" element={<LeaseLayout />}>
            <Route index element={<OverviewPage />} />
            <Route path="review" element={<ReviewPage />} />
          </Route>
          <Route path="*" element={<NotFound />} />
        </Routes>
      </BrowserRouter>
    </ServiceProvider>
  );
}
