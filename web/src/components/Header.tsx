import { Link } from 'react-router-dom';
import type { Lease } from '../services/types';

interface HeaderProps {
  lease?: Lease;
  /**
   * Show the property in the header. Screens that already name the property in their main
   * content turn this off so the identity appears once.
   */
  showProperty?: boolean;
}

/** Compact: product name and the current identity. No sidebar, and no controls that do nothing. */
export function Header({ lease, showProperty = true }: HeaderProps) {
  const me = lease?.participants.find((p) => p.id === lease.viewer.userId);
  return (
    <header className="site-header">
      <div className="page site-header__row">
        <Link to="/" className="site-header__brand">
          Deposit Keeper
        </Link>
        {lease && showProperty && <p className="site-header__context">{lease.property.label}</p>}
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
