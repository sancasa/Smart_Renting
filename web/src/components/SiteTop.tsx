import { useLayoutEffect, useRef } from 'react';
import type { Lease } from '../services/types';
import { DemoBar } from './DemoBar';
import { Header } from './Header';

/**
 * The sticky top bar: the compact simulated-data notice plus the header. It publishes its real
 * height as --top-h so scroll padding and sticky panels can sit below it. That is what keeps a
 * focused element from ever being hidden behind the bar, whatever the screen size or wrapping.
 */
export function SiteTop({ lease, showProperty }: { lease?: Lease; showProperty?: boolean }) {
  const ref = useRef<HTMLDivElement>(null);

  useLayoutEffect(() => {
    const element = ref.current;
    if (!element) return;
    const publish = () => document.documentElement.style.setProperty('--top-h', `${element.offsetHeight}px`);
    publish();
    const observer = new ResizeObserver(publish);
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  return (
    <div className="site-top" ref={ref}>
      <DemoBar />
      <Header lease={lease} showProperty={showProperty} />
    </div>
  );
}
