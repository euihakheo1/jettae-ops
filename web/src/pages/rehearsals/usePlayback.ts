import { useCallback, useEffect, useRef, useState } from 'react';

/** 단계 재생 (0 = 시작 전, total = 모두 공개) */
export function usePlayback(total: number, { intervalMs = 900, initial = total, autoplay = false } = {}) {
  const [step, setStepState] = useState(Math.min(initial, total));
  const [playing, setPlaying] = useState(autoplay);
  const timer = useRef<number | null>(null);

  const stop = () => {
    if (timer.current !== null) window.clearInterval(timer.current);
    timer.current = null;
  };

  useEffect(() => {
    if (!playing) return stop;
    timer.current = window.setInterval(() => {
      setStepState((s) => {
        if (s >= total) {
          setPlaying(false);
          return s;
        }
        return s + 1;
      });
    }, intervalMs);
    return stop;
  }, [playing, total, intervalMs]);

  const play = useCallback(() => {
    setStepState((s) => (s >= total ? 0 : s));
    setPlaying(true);
  }, [total]);
  const pause = useCallback(() => setPlaying(false), []);
  const reset = useCallback(() => {
    setPlaying(false);
    setStepState(0);
  }, []);
  const setStep = useCallback(
    (n: number) => {
      setPlaying(false);
      setStepState(Math.max(0, Math.min(total, n)));
    },
    [total],
  );
  /** 처음부터 다시 재생 */
  const restart = useCallback(() => {
    setStepState(0);
    setPlaying(true);
  }, []);

  return { step, playing, play, pause, reset, setStep, restart };
}
