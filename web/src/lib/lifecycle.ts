import type { Stage } from '../services/types';

export const LIFECYCLE_STEPS: { stage: Stage; label: string }[] = [
  { stage: 'setup', label: 'Setup' },
  { stage: 'move_in', label: 'Move-in' },
  { stage: 'active', label: 'Active' },
  { stage: 'move_out', label: 'Move-out' },
  { stage: 'settlement', label: 'Settlement' },
];

export type StepState = 'done' | 'current' | 'upcoming';

export interface TimelineStep {
  stage: Stage;
  label: string;
  state: StepState;
}

/**
 * Where each step is. Settlement stays "current" (in progress) until the lease is complete,
 * which the server reports only when every transfer is confirmed and nothing is held.
 */
export function stepStates(stage: Stage, complete: boolean): TimelineStep[] {
  const currentIndex = LIFECYCLE_STEPS.findIndex((s) => s.stage === stage);
  return LIFECYCLE_STEPS.map((step, index) => ({
    ...step,
    state: complete || index < currentIndex ? 'done' : index === currentIndex ? 'current' : 'upcoming',
  }));
}
