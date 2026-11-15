import { describe, expect, it } from 'vitest';
import { formatUsd, formatUsdWhole, parseUsdInput, subUnits, sumUnits, toInputText } from './money';
import { reconcileFunds } from './funds';

describe('formatUsd', () => {
  it('formats whole-cent amounts with two decimals and grouping', () => {
    expect(formatUsd('1000000000')).toBe('$1,000.00');
    expect(formatUsd('880000000')).toBe('$880.00');
    expect(formatUsd('0')).toBe('$0.00');
    expect(formatUsd('70500000')).toBe('$70.50');
  });
  it('keeps sub-cent precision instead of rounding settlement amounts', () => {
    expect(formatUsd('1234567')).toBe('$1.234567');
  });
  it('rejects non-integer strings', () => {
    expect(() => formatUsd('12.5')).toThrow();
    expect(() => formatUsd('-1')).toThrow();
  });
});

describe('formatUsdWhole', () => {
  it('drops the cents only when there are none', () => {
    expect(formatUsdWhole('120000000')).toBe('$120');
    expect(formatUsdWhole('1000000000')).toBe('$1,000');
    expect(formatUsdWhole('120500000')).toBe('$120.50');
    expect(formatUsdWhole('1234567')).toBe('$1.234567');
  });
});

describe('parseUsdInput', () => {
  it('parses plain, decimal, symbol and comma input to base units', () => {
    expect(parseUsdInput('70')).toEqual({ ok: true, value: '70000000' });
    expect(parseUsdInput('70.5')).toEqual({ ok: true, value: '70500000' });
    expect(parseUsdInput('$1,250.00')).toEqual({ ok: true, value: '1250000000' });
  });
  it('rejects empty, negative, text and sub-cent input with a recovery hint', () => {
    for (const bad of ['', '  ', '-5', 'abc', '1.234', '1e3']) {
      const result = parseUsdInput(bad);
      expect(result.ok).toBe(false);
      if (!result.ok) expect(result.error.length).toBeGreaterThan(0);
    }
  });
  it('round-trips through toInputText', () => {
    expect(toInputText('70000000')).toBe('70.00');
    expect(parseUsdInput(toInputText('123450000'))).toEqual({ ok: true, value: '123450000' });
  });
});

describe('integer arithmetic', () => {
  it('adds and subtracts without floating point drift', () => {
    expect(sumUnits(['100000', '200000'])).toBe('300000');
    expect(subUnits('1000000000', '120000000')).toBe('880000000');
    expect(() => subUnits('1', '2')).toThrow();
  });
});

describe('reconcileFunds', () => {
  const base = {
    original: '1000000000',
    returnedConfirmed: '0',
    returnedPending: '0',
    paidToLandlordConfirmed: '0',
    paidToLandlordPending: '0',
    held: '1000000000',
  };
  it('accepts funds that add up to the deposit', () => {
    expect(reconcileFunds(base)).toEqual({ ok: true });
    expect(
      reconcileFunds({ ...base, returnedConfirmed: '880000000', held: '120000000' }),
    ).toEqual({ ok: true });
  });
  it('flags a pending transfer that is also counted as returned (double count)', () => {
    const result = reconcileFunds({
      ...base,
      returnedConfirmed: '880000000',
      returnedPending: '880000000',
      held: '120000000',
    });
    expect(result.ok).toBe(false);
  });
});
