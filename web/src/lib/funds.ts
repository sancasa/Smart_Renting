import { sumUnits, toBig, type BaseUnits } from './money';

/**
 * Where every unit of the deposit is. Each unit lives in exactly one bucket, so the
 * parts always add up to the original deposit:
 * - confirmed buckets: transfers the authoritative source has verified
 * - pending buckets: transfers in progress (NOT yet returned or paid, and NOT held)
 * - held: still in escrow
 * The server supplies these. The frontend only displays them and refuses to show
 * numbers that do not reconcile.
 */
export interface Funds {
  original: BaseUnits;
  returnedConfirmed: BaseUnits;
  returnedPending: BaseUnits;
  paidToLandlordConfirmed: BaseUnits;
  paidToLandlordPending: BaseUnits;
  held: BaseUnits;
}

export type Reconciliation = { ok: true } | { ok: false; difference: string };

export function reconcileFunds(funds: Funds): Reconciliation {
  const parts = sumUnits([
    funds.returnedConfirmed,
    funds.returnedPending,
    funds.paidToLandlordConfirmed,
    funds.paidToLandlordPending,
    funds.held,
  ]);
  const difference = toBig(parts) - toBig(funds.original);
  return difference === 0n ? { ok: true } : { ok: false, difference: difference.toString() };
}
