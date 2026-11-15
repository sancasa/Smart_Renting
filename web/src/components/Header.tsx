import { Link } from 'react-router-dom';
import type { Lease } from '../services/types';

/** Product name, property context and the current identity. No sidebar, and no controls that do nothing. */
export function Header({ lease }: { lease?: Lease }) {
  const me = lease?.participants.find((p) => p.id === lease.viewer.userId);
  return (
    <header className="site-header">
      <div className="page site-header__row">
        <Link to="/" className="site-header__brand">
          Deposit Keeper
        </Link>
        {lease && <p className="site-header__context">{lease.property.label}</p>}
        {me && (
          <p className="site-header__account">
            <span className="sr-only">Current account: </span>
            {me.displayName}, {me.role}
          </p>
        )}
      </div>
    </header>
  );
}
