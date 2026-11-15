import { useEffect, useId, useRef, useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { initialsOf, roleLabel } from '../lib/format';
import type { Participant } from '../services/types';

interface AccountMenuProps {
  person: Participant;
  propertyLabel: string;
  leaseId: string;
}

function Chevron() {
  return (
    <svg className="account__chevron" width="16" height="16" viewBox="0 0 18 18" aria-hidden="true" focusable="false">
      <path d="m4.5 7 4.5 4.5L13.5 7" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

/**
 * Account button and menu. It shows who is signed in and offers only what the app can really do:
 * go to the lease overview. There is no sign-out or settings here because no session service
 * exists yet, and the menu does not pretend otherwise. Demo persona switching lives in the demo
 * controls, never here. Add real account actions to the list below when the session service lands.
 *
 * Pattern: a disclosure, not role="menu", because it holds identity details and one link and has
 * no arrow-key navigation to promise. Escape closes it and returns focus to the button, an outside
 * click closes it without stealing focus, and tabbing out of it closes it.
 */
export function AccountMenu({ person, propertyLabel, leaseId }: AccountMenuProps) {
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);
  const buttonRef = useRef<HTMLButtonElement>(null);
  const panelRef = useRef<HTMLDivElement>(null);
  const panelId = useId();
  const location = useLocation();
  const overviewPath = `/leases/${leaseId}`;
  const onOverview = location.pathname === overviewPath;

  useEffect(() => {
    if (!open) return;
    const onPointerDown = (event: PointerEvent) => {
      if (!rootRef.current?.contains(event.target as Node)) setOpen(false);
    };
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key !== 'Escape') return;
      event.preventDefault();
      setOpen(false);
      buttonRef.current?.focus();
    };
    document.addEventListener('pointerdown', onPointerDown);
    document.addEventListener('keydown', onKeyDown);
    return () => {
      document.removeEventListener('pointerdown', onPointerDown);
      document.removeEventListener('keydown', onKeyDown);
    };
  }, [open]);

  // Opening moves focus to the first action, so keyboard users land inside the menu.
  useEffect(() => {
    if (open) panelRef.current?.querySelector<HTMLElement>('a')?.focus();
  }, [open]);

  // Going to another screen closes the menu.
  useEffect(() => setOpen(false), [location.pathname]);

  const role = roleLabel(person.role);
  const initials = initialsOf(person.displayName);

  return (
    <div
      className="account"
      ref={rootRef}
      onBlur={(event) => {
        // Tabbing out closes it. A click on non-focusable text inside leaves relatedTarget empty, so ignore that.
        const next = event.relatedTarget as Node | null;
        if (next && !rootRef.current?.contains(next)) setOpen(false);
      }}
    >
      <button
        ref={buttonRef}
        type="button"
        className="account__button"
        aria-expanded={open}
        aria-controls={open ? panelId : undefined}
        onClick={() => setOpen((value) => !value)}
      >
        <span className="account__avatar" aria-hidden="true">
          {initials}
        </span>
        <span className="account__text">
          <span className="sr-only">Account menu: </span>
          <span className="account__name">{person.displayName}</span>
          <span className="account__role">{role}</span>
        </span>
        <Chevron />
      </button>

      {open && (
        <div id={panelId} ref={panelRef} className="account__panel" role="group" aria-label="Account">
          <div className="account__identity">
            <span className="account__avatar account__avatar--large" aria-hidden="true">
              {initials}
            </span>
            <div className="account__identity-text">
              <p className="account__identity-name">{person.displayName}</p>
              <p className="account__identity-role">{role}</p>
              <p className="account__identity-property">{propertyLabel}</p>
            </div>
          </div>
          <ul className="account__actions">
            <li>
              <Link
                to={overviewPath}
                className="account__item"
                aria-current={onOverview ? 'page' : undefined}
                onClick={(event) => {
                  setOpen(false);
                  if (onOverview) {
                    // Already here: there is no new page to land on, so don't navigate. Put focus back on the button.
                    event.preventDefault();
                    buttonRef.current?.focus();
                  }
                }}
              >
                Lease overview
              </Link>
            </li>
          </ul>
        </div>
      )}
    </div>
  );
}
