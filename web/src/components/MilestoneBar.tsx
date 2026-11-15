import type { Stage } from '../services/types';

const STEPS: { stage: Stage; label: string }[] = [
  { stage: 'setup', label: 'Setup' },
  { stage: 'move_in', label: 'Move-in' },
  { stage: 'active', label: 'Active' },
  { stage: 'move_out', label: 'Move-out' },
  { stage: 'settlement', label: 'Settlement' },
];

function CheckIcon() {
  return (
    <svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true" focusable="false">
      <path d="M2 6.5 4.7 9 10 3.2" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

/**
 * Shows progress only. It is not navigation and cannot skip a required step.
 * Each step says its state in words as well as shape, so color is never the only cue.
 */
export function MilestoneBar({ stage, complete }: { stage: Stage; complete: boolean }) {
  const currentIndex = STEPS.findIndex((s) => s.stage === stage);
  return (
    <ol className="milestones" aria-label="Deposit progress">
      {STEPS.map((step, index) => {
        const state = complete || index < currentIndex ? 'done' : index === currentIndex ? 'current' : 'upcoming';
        const spoken = state === 'done' ? 'completed' : state === 'current' ? 'current step' : 'not started';
        return (
          <li key={step.stage} className={`milestones__item milestones__item--${state}`} aria-current={state === 'current' ? 'step' : undefined}>
            <span className="milestones__marker" aria-hidden="true">
              {state === 'done' && <CheckIcon />}
            </span>
            <span>
              {step.label}
              <span className="sr-only">, {spoken}</span>
            </span>
          </li>
        );
      })}
    </ol>
  );
}
