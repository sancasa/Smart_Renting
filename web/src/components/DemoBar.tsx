import { useEffect, useId, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useServices } from '../services/ServiceContext';
import type { ScenarioId } from '../services/sample/engine';
import { Disclosure } from './Disclosure';

/**
 * Sample-only. Rendered only when the sample adapter is active, so live mode can never expose
 * a persona switch. The row keeps the simulated-data label visible and compact; the controls
 * stay closed until asked for. Sample transfers are not blockchain transactions and are never
 * shown as verified.
 */
export function DemoBar() {
  const { bundle } = useServices();
  const [open, setOpen] = useState(false);
  if (!bundle.sampleControls) return null;

  return (
    <>
      <div className="demo-bar" role="region" aria-label="Demo notice">
        <div className="page">
          <div className="demo-bar__row">
            <p className="demo-bar__label">Demo · simulated data</p>
            <button type="button" className="demo-bar__toggle" aria-haspopup="dialog" onClick={() => setOpen(true)}>
              Demo controls
            </button>
          </div>
        </div>
      </div>
      <DemoControlsDialog open={open} onClose={() => setOpen(false)} />
    </>
  );
}

function DemoControlsDialog({ open, onClose }: { open: boolean; onClose: () => void }) {
  const { bundle, revision } = useServices();
  const controls = bundle.sampleControls!;
  const navigate = useNavigate();
  const ref = useRef<HTMLDialogElement>(null);
  const titleId = useId();
  const [scenario, setScenario] = useState<ScenarioId>('active');
  const [notice, setNotice] = useState('');

  useEffect(() => {
    const dialog = ref.current;
    if (!dialog) return;
    if (open && !dialog.open) dialog.showModal();
    if (!open && dialog.open) dialog.close();
  }, [open]);

  void revision; // re-render when sample state changes
  const role = controls.getViewerRole();
  const showResult = () => {
    navigate(`/leases/${bundle.defaultLeaseId}`);
    onClose();
  };

  return (
    <dialog
      ref={ref}
      className="dialog"
      aria-labelledby={titleId}
      onCancel={(event) => {
        event.preventDefault();
        onClose();
      }}
    >
      {open && (
        <div className="dialog__content">
          <h2 id={titleId} className="dialog__title">
            Demo controls
          </h2>
          <p className="small muted">Nothing here is a real payment or blockchain transaction.</p>

          <fieldset className="demo-bar__group">
            <legend className="demo-bar__legend">Preview as</legend>
            {(['tenant', 'landlord'] as const).map((value) => (
              <label key={value} className="demo-bar__radio">
                <input type="radio" name="preview-role" checked={role === value} onChange={() => controls.setViewerRole(value)} />
                {value === 'tenant' ? 'Tenant' : 'Landlord'}
              </label>
            ))}
          </fieldset>

          <div className="demo-bar__group">
            <label htmlFor="demo-scenario" className="demo-bar__legend">
              Jump to a state
            </label>
            <select id="demo-scenario" className="demo-bar__select" value={scenario} onChange={(event) => setScenario(event.target.value as ScenarioId)}>
              {controls.scenarios.map((s) => (
                <option key={s.id} value={s.id}>
                  {s.label}
                </option>
              ))}
            </select>
            <button
              type="button"
              className="button button--small"
              onClick={() => {
                controls.loadScenario(scenario);
                showResult();
              }}
            >
              Load state
            </button>
          </div>

          <div className="demo-bar__group">
            <button type="button" className="button button--small" disabled={!controls.hasPendingTransfers()} onClick={() => controls.confirmPendingTransfers()}>
              Confirm pending transfers
            </button>
            <button
              type="button"
              className="button button--small"
              disabled={!controls.canMarkProposalOutdated()}
              onClick={() => {
                controls.markProposalOutdated();
                setNotice('A newer proposal version now exists. An open review page still shows the old one, so accepting it fails until the proposal is reloaded.');
              }}
            >
              Make proposal outdated
            </button>
            <button
              type="button"
              className="button button--small"
              onClick={() => {
                controls.reset();
                showResult();
              }}
            >
              Reset demo
            </button>
          </div>

          <p role="status" className="small">
            {notice}
          </p>

          <Disclosure title="What this demo assumes" standalone>
            <ul className="demo-bar__list">
              {controls.assumptions.map((text) => (
                <li key={text}>{text}</li>
              ))}
            </ul>
            <p className="small muted">Demo state stays in this browser tab only. Don't enter real personal information.</p>
          </Disclosure>

          <div className="button-row">
            <button type="button" className="button button--primary" onClick={onClose}>
              Close
            </button>
          </div>
        </div>
      )}
    </dialog>
  );
}
