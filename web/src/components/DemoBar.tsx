import { useId, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useServices } from '../services/ServiceContext';
import type { ScenarioId } from '../services/sample/engine';

/**
 * Sample-only controls. Rendered only when the sample adapter is active, so live mode
 * can never expose a persona switch. It carries the one persistent "simulated" label.
 * Sample transfers are not blockchain transactions and are never shown as verified.
 */
export function DemoBar() {
  const { bundle, revision } = useServices();
  const controls = bundle.sampleControls;
  const navigate = useNavigate();
  const [open, setOpen] = useState(false);
  const [scenario, setScenario] = useState<ScenarioId>('active');
  const [notice, setNotice] = useState('');
  const panelId = useId();
  if (!controls) return null;

  void revision; // re-render when sample state changes
  const role = controls.getViewerRole();
  const goHome = () => navigate(`/leases/${bundle.defaultLeaseId}`);

  return (
    <div className="demo-bar" role="region" aria-label="Demo controls">
      <div className="page">
        <div className="demo-bar__row">
          <p className="demo-bar__label">Demo · simulated data</p>
          <p className="demo-bar__hint">No real payments or blockchain transactions.</p>
          <button
            type="button"
            className="button button--small demo-bar__toggle"
            aria-expanded={open}
            aria-controls={panelId}
            onClick={() => setOpen((value) => !value)}
          >
            {open ? 'Hide demo controls' : 'Show demo controls'}
            <svg className="demo-bar__chevron" width="16" height="16" viewBox="0 0 18 18" aria-hidden="true" focusable="false">
              <path d="m4.5 7 4.5 4.5L13.5 7" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </button>
        </div>

        <div id={panelId} className="demo-bar__panel" hidden={!open}>
          <p className="demo-bar__panel-hint demo-bar__wide">Nothing here is a real payment or blockchain transaction.</p>
          <fieldset className="demo-bar__group">
            <legend className="demo-bar__legend">Preview as</legend>
            {(['tenant', 'landlord'] as const).map((value) => (
              <label key={value} className="demo-bar__radio">
                <input
                  type="radio"
                  name="preview-role"
                  checked={role === value}
                  onChange={() => controls.setViewerRole(value)}
                />
                {value === 'tenant' ? 'Tenant' : 'Landlord'}
              </label>
            ))}
          </fieldset>

          <div className="demo-bar__group">
            <label htmlFor="demo-scenario" className="demo-bar__legend">
              Jump to a state
            </label>
            <select
              id="demo-scenario"
              className="demo-bar__select"
              value={scenario}
              onChange={(event) => setScenario(event.target.value as ScenarioId)}
            >
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
                goHome();
              }}
            >
              Load state
            </button>
          </div>

          <div className="demo-bar__group">
            <button
              type="button"
              className="button button--small"
              disabled={!controls.hasPendingTransfers()}
              onClick={() => controls.confirmPendingTransfers()}
            >
              Confirm pending transfers
            </button>
            <button
              type="button"
              className="button button--small"
              disabled={!controls.canMarkProposalOutdated()}
              onClick={() => {
                controls.markProposalOutdated();
                setNotice(
                  'A newer proposal version now exists. An open review page still shows the old one, so accepting it fails until the proposal is reloaded.',
                );
              }}
            >
              Make proposal outdated
            </button>
            <button
              type="button"
              className="button button--small"
              onClick={() => {
                controls.reset();
                goHome();
              }}
            >
              Reset demo
            </button>
          </div>

          <p role="status" className="small demo-bar__wide">
            {notice}
          </p>

          <div className="demo-bar__wide">
            <p className="demo-bar__legend">What this demo assumes</p>
            <ul className="demo-bar__list">
              {controls.assumptions.map((text) => (
                <li key={text}>{text}</li>
              ))}
            </ul>
          </div>
          <p className="small demo-bar__wide">Demo state stays in this browser tab only. Don't enter real personal information.</p>
        </div>
      </div>
    </div>
  );
}
