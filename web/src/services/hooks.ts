import { useCallback, useEffect, useRef, useState } from 'react';
import { ServiceError, toServiceError } from './errors';
import { useServices } from './ServiceContext';
import type { Lease, Proposal } from './types';

export interface LeaseData {
  lease: Lease;
  proposal: Proposal | null;
}

export interface LeaseDataState {
  data?: LeaseData;
  error?: ServiceError;
  loading: boolean;
}

/**
 * Loads the lease and the viewer's proposal. Keeps the last good data while reloading so
 * screens do not flash. `reloadNow` resolves only after the newest data has been applied,
 * so an action can wait for fresh state before it closes a dialog or navigates. That stops
 * the next screen from briefly showing the old state with actions that no longer apply.
 */
export function useLeaseData(leaseId: string): LeaseDataState & { reload: () => void; reloadNow: () => Promise<void> } {
  const { bundle } = useServices();
  const [state, setState] = useState<LeaseDataState>({ loading: true });
  const latest = useRef(0);
  const inflight = useRef<Promise<void>>(Promise.resolve());

  const reloadNow = useCallback((): Promise<void> => {
    const id = ++latest.current;
    setState((s) => ({ ...s, loading: true }));
    const request = (async () => {
      try {
        const [lease, proposal] = await Promise.all([bundle.service.getLease(leaseId), bundle.service.getProposal(leaseId)]);
        if (id === latest.current) setState({ data: { lease, proposal }, loading: false });
      } catch (error) {
        if (id === latest.current) setState({ error: toServiceError(error), loading: false });
      }
    })();
    inflight.current = request;
    // If a newer request started meanwhile, wait for that one too.
    return request.then(() => inflight.current);
  }, [bundle, leaseId]);

  useEffect(() => {
    void reloadNow();
    return () => {
      latest.current += 1; // drop results that arrive after leaving this lease
    };
  }, [reloadNow]);

  // Sample adapter changes (demo controls, role switch) push updates; a live adapter will poll.
  useEffect(() => bundle.service.subscribe?.(() => void reloadNow()), [bundle, reloadNow]);

  const reload = useCallback(() => void reloadNow(), [reloadNow]);
  return { ...state, reload, reloadNow };
}

/**
 * Runs one user action at a time. A second call while one is running is ignored, which
 * stops double-clicks from creating duplicate operations. Returns true on success.
 */
export function useMutation() {
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<ServiceError | null>(null);
  const running = useRef(false);

  const run = useCallback(async (action: () => Promise<unknown>): Promise<boolean> => {
    if (running.current) return false;
    running.current = true;
    setPending(true);
    setError(null);
    try {
      await action();
      return true;
    } catch (caught) {
      setError(toServiceError(caught));
      return false;
    } finally {
      running.current = false;
      setPending(false);
    }
  }, []);

  const clearError = useCallback(() => setError(null), []);
  return { run, pending, error, clearError };
}
