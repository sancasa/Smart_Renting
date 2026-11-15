import { createLiveService } from './live';
import { SAMPLE_LEASE_ID } from './sample/engine';
import { createSampleService, type SampleControls } from './sample';
import type { DepositService } from './types';

export interface ServiceBundle {
  service: DepositService;
  /** Present only in sample mode. Live mode never offers persona or simulation controls. */
  sampleControls?: SampleControls;
  /** Where "/" goes. Live mode will replace this with a lease list when the backend supports several leases. */
  defaultLeaseId: string;
}

/** Adapter is chosen explicitly. Default is sample for this milestone; there is no silent fallback. */
export function createServices(): ServiceBundle {
  const adapter = import.meta.env.VITE_ADAPTER === 'live' ? 'live' : 'sample';
  if (adapter === 'live') return { service: createLiveService(), defaultLeaseId: 'current' };
  const { service, controls } = createSampleService();
  return { service, sampleControls: controls, defaultLeaseId: SAMPLE_LEASE_ID };
}

export type { SampleControls };
