/**
 * Money helpers. Amounts cross the API as integer base-unit strings and are
 * handled as bigint here. Never use floating point for deposit totals.
 *
 * ASSUMPTION (confirm with the contracts owner): the app-level token view is the
 * 6-decimal ERC-20 USDC interface. Arc's native balance uses 18 decimals and must
 * never be mixed with these values.
 */
export const USDC_DECIMALS = 6;
const UNIT = 10n ** BigInt(USDC_DECIMALS);
const CENT = UNIT / 100n;

/** Integer base units as a decimal string, e.g. "1000000000" for $1,000.00. */
export type BaseUnits = string;

export function toBig(value: BaseUnits): bigint {
  if (!/^\d+$/.test(value)) throw new Error(`Invalid base-unit amount: ${value}`);
  return BigInt(value);
}

export function sumUnits(values: BaseUnits[]): BaseUnits {
  return values.reduce((total, v) => total + toBig(v), 0n).toString();
}

export function subUnits(a: BaseUnits, b: BaseUnits): BaseUnits {
  const result = toBig(a) - toBig(b);
  if (result < 0n) throw new Error('Amount would be negative');
  return result.toString();
}

export function compareUnits(a: BaseUnits, b: BaseUnits): -1 | 0 | 1 {
  const x = toBig(a);
  const y = toBig(b);
  return x < y ? -1 : x > y ? 1 : 0;
}

export function isZero(value: BaseUnits): boolean {
  return toBig(value) === 0n;
}

export function isPositive(value: BaseUnits): boolean {
  return toBig(value) > 0n;
}

/** "$1,000.00". Whole-cent amounts show two decimals; anything finer shows up to six. */
export function formatUsd(value: BaseUnits): string {
  const n = toBig(value);
  const whole = n / UNIT;
  const fraction = n % UNIT;
  const wholeText = whole.toLocaleString('en-US');
  let fractionText = fraction.toString().padStart(USDC_DECIMALS, '0');
  fractionText = fraction % CENT === 0n ? fractionText.slice(0, 2) : fractionText.replace(/0+$/, '');
  return `$${wholeText}.${fractionText}`;
}

/** "$120" when there are no cents, otherwise the exact amount ("$120.50"). For short action labels. */
export function formatUsdWhole(value: BaseUnits): string {
  const n = toBig(value);
  if (n % UNIT !== 0n) return formatUsd(value);
  return `$${(n / UNIT).toLocaleString('en-US')}`;
}

export type ParsedAmount = { ok: true; value: BaseUnits } | { ok: false; error: string };

/** Parses user input such as "70", "70.5", "$1,250.00" into base units (whole cents only). */
export function parseUsdInput(text: string): ParsedAmount {
  const cleaned = text.trim().replace(/^\$/, '').replace(/,/g, '');
  if (cleaned === '') return { ok: false, error: 'Enter an amount, like 70 or 70.50.' };
  if (!/^\d+(\.\d{1,2})?$/.test(cleaned)) {
    return { ok: false, error: 'Use numbers with up to two decimals, like 70 or 70.50.' };
  }
  const [whole, fraction = ''] = cleaned.split('.');
  const units = BigInt(whole) * UNIT + BigInt(fraction.padEnd(2, '0')) * CENT;
  return { ok: true, value: units.toString() };
}

/** Plain-number form for prefilling an input, e.g. "70.00". */
export function toInputText(value: BaseUnits): string {
  const n = toBig(value);
  const whole = n / UNIT;
  const cents = (n % UNIT) / CENT;
  return `${whole}.${cents.toString().padStart(2, '0')}`;
}
