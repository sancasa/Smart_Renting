import type { Stage } from '../services/types';

const STEPS: { stage: Stage; label: string }[] = [
  { stage: 'setup', label: 'Setup' },
  { stage: 'move_in', label: 'Move-in' },
  { stage: 'active', label: 'Active' },
  { stage: 'move_out', label: 'Move-out' },
  { stage: 'settlement', label: 'Settlement' },
];

/**
 * Secondary progress: a slim segmented line with the current step in words. It shows where
 * the lease is and cannot skip a step. Every step says its state in text for screen readers,
 * and the step names stay visible on wider screens, so color is never the only cue.
 */
export function MilestoneBar({ stage, complete }: { stage: Stage; complete: boolean }) {
  const currentIndex = STEPS.findIndex((s) => s.stage === stage);
  const summary = complete ? 'All 5 steps complete' : `Step ${currentIndex + 1} of ${STEPS.length}: ${STEPS[currentIndex].label}`;
  return (
    <div className="progress">
      <p className="progress__text">{summary}</p>
      <ol className="progress__bar" aria-label="Deposit progress">
        {STEPS.map((step, index) => {
          const state = complete || index < currentIndex ? 'done' : index === currentIndex ? 'current' : 'upcoming';
          const spoken = state === 'done' ? 'completed' : state === 'current' ? 'current step' : 'not started';
          return (
            <li key={step.stage} className={`progress__step progress__step--${state}`} aria-current={state === 'current' ? 'step' : undefined}>
              <span className="progress__segment" aria-hidden="true" />
              <span className="progress__name">
                {step.label}
                <span className="sr-only">, {spoken}</span>
              </span>
            </li>
          );
        })}
      </ol>
    </div>
  );
}
