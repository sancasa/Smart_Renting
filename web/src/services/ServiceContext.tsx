import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from 'react';
import { createServices, type ServiceBundle } from './index';

interface ServiceContextValue {
  bundle: ServiceBundle;
  /** Bumps whenever data may have changed, so screens reload. */
  revision: number;
  refresh: () => void;
}

const ServiceContext = createContext<ServiceContextValue | null>(null);

export function ServiceProvider({ children }: { children: ReactNode }) {
  const [bundle] = useState<ServiceBundle>(createServices);
  const [revision, setRevision] = useState(0);
  const refresh = useCallback(() => setRevision((r) => r + 1), []);

  useEffect(() => bundle.service.subscribe?.(refresh), [bundle, refresh]);

  const value = useMemo(() => ({ bundle, revision, refresh }), [bundle, revision, refresh]);
  return <ServiceContext.Provider value={value}>{children}</ServiceContext.Provider>;
}

export function useServices(): ServiceContextValue {
  const value = useContext(ServiceContext);
  if (!value) throw new Error('useServices must be used inside ServiceProvider');
  return value;
}
