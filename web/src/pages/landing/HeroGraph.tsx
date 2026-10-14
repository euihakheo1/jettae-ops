/**
 * 랜딩 히어로 그림: 런북 한 문장이 운영 그래프가 되고, 끊긴 곳이 빨갛게 드러나는 장면 (순수 SVG + CSS 애니메이션).
 * "결제 장애가 계속되면 김대리에게 연락, 안 되면 롤백."
 */
export function HeroGraph() {
  return (
    <div className="relative mx-auto w-full max-w-[620px]" aria-hidden>
      <div className="absolute inset-6 rounded-[32px] bg-brand/10 blur-3xl" />
      <div className="relative rounded-[28px] border border-line bg-surface/90 p-4 shadow-2xl backdrop-blur">
        <div className="flex items-center gap-2 border-b border-line px-2 pb-3">
          <span className="size-2.5 rounded-full bg-danger/70" />
          <span className="size-2.5 rounded-full bg-warn/70" />
          <span className="size-2.5 rounded-full bg-ok/70" />
          <span className="ml-3 truncate font-mono text-[13px] text-fg-subtle">runbooks/01-결제-주문-장애대응.md : 13</span>
        </div>
        <p className="mx-2 mt-3 rounded-lg bg-surface-2 px-3 py-2 text-[15.5px] leading-relaxed">
          “결제 장애가 계속되면 <mark className="quote-hit">김대리에게 연락</mark>, 안 되면 <mark className="quote-hit">롤백</mark>.”
        </p>
        <svg viewBox="0 0 560 300" className="mt-2 w-full">
          <defs>
            <marker id="hg-a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
              <path d="M0,0 L10,5 L0,10 z" fill="var(--edge)" />
            </marker>
            <marker id="hg-r" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
              <path d="M0,0 L10,5 L0,10 z" fill="var(--danger)" />
            </marker>
          </defs>
          {/* edges */}
          <path d="M126 150 C 152 150, 152 150, 178 150" stroke="var(--edge)" strokeWidth="2" fill="none" markerEnd="url(#hg-a)" />
          <path d="M300 136 C 330 110, 340 92, 362 86" stroke="var(--danger)" strokeWidth="3" fill="none" markerEnd="url(#hg-r)" className="hero-flow" />
          <path d="M300 166 C 330 196, 340 212, 362 218" stroke="var(--edge)" strokeWidth="2" fill="none" markerEnd="url(#hg-a)" />
          <path d="M470 86 C 488 86, 496 60, 508 50" stroke="var(--ghost)" strokeWidth="2.5" strokeDasharray="6 5" fill="none" markerEnd="url(#hg-r)" className="hero-flow" />
          <path d="M470 218 C 488 218, 496 246, 508 256" stroke="var(--ghost)" strokeWidth="2.5" strokeDasharray="6 5" fill="none" markerEnd="url(#hg-r)" className="hero-flow" />
          {/* labels */}
          <EdgeLabel x={152} y={136} text="감지" />
          <EdgeLabel x={326} y={98} text="담당" danger />
          <EdgeLabel x={326} y={210} text="조치" />

          {/* trigger */}
          <g>
            <rect x="2" y="120" width="124" height="60" rx="14" fill="var(--surface-2)" stroke="var(--line-strong)" />
            <circle cx="24" cy="150" r="13" fill="color-mix(in oklab, var(--warn) 22%, transparent)" />
            <path d="M19 154h10M21 154v-5a3 3 0 0 1 6 0v5" stroke="var(--warn)" strokeWidth="1.8" fill="none" strokeLinecap="round" />
            <text x="44" y="146" fontSize="14.5" fontWeight="800" fill="var(--fg)">실패율 알람</text>
            <text x="44" y="165" fontSize="12.5" fill="var(--fg-subtle)">5% · 3분</text>
          </g>
          {/* service */}
          <g>
            <rect x="180" y="112" width="120" height="76" rx="16" fill="color-mix(in oklab, var(--brand) 12%, var(--surface-2))" stroke="var(--brand)" />
            <text x="194" y="142" fontSize="19" fontWeight="900" fill="var(--fg)">결제</text>
            <rect x="262" y="126" width="26" height="17" rx="5" fill="var(--surface)" stroke="var(--line-strong)" />
            <text x="268" y="138.5" fontSize="10" fontWeight="700" fill="var(--fg-muted)">T1</text>
            <rect x="194" y="153" width="64" height="23" rx="11.5" fill="color-mix(in oklab, var(--danger) 16%, transparent)" stroke="color-mix(in oklab, var(--danger) 45%, transparent)" />
            <text x="206" y="169" fontSize="14" fontWeight="900" fill="var(--danger)">22.0</text>
          </g>
          {/* departed person */}
          <g className="hero-pulse">
            <rect x="364" y="58" width="106" height="56" rx="16" fill="color-mix(in oklab, var(--danger) 12%, var(--surface-2))" stroke="var(--danger)" />
            <circle cx="388" cy="86" r="14" fill="var(--danger)" />
            <path d="M383 81l10 10M393 81l-10 10" stroke="#fff" strokeWidth="2" strokeLinecap="round" />
            <text x="407" y="85" fontSize="16" fontWeight="900" fill="var(--danger)" textDecoration="line-through">김대리</text>
            <text x="407" y="103" fontSize="12.5" fill="var(--fg-muted)">8/31 퇴사</text>
            <rect x="430" y="48" width="34" height="17" rx="8.5" fill="var(--danger)" />
            <text x="447" y="60.5" textAnchor="middle" fontSize="11" fontWeight="800" fill="#fff">퇴사</text>
          </g>
          {/* action */}
          <g>
            <rect x="364" y="190" width="106" height="56" rx="16" fill="var(--surface-2)" stroke="var(--line-strong)" />
            <circle cx="388" cy="218" r="13" fill="color-mix(in oklab, var(--ok) 20%, transparent)" />
            <path d="M383 214a6 6 0 1 1-1 7M382 210v5h5" stroke="var(--ok)" strokeWidth="1.8" fill="none" strokeLinecap="round" />
            <text x="407" y="216" fontSize="16" fontWeight="800" fill="var(--fg)">롤백</text>
            <text x="407" y="235" fontSize="12.5" fill="var(--fg-subtle)">승인 필요</text>
          </g>
          {/* ghosts */}
          <Ghost x={466} y={12} label="상향 없음" />
          <Ghost x={466} y={252} label="승인자 없음" />
        </svg>
      </div>
      {/* floating cards */}
      <div className="absolute -bottom-7 -left-4 w-max max-w-[calc(100%-1rem)] animate-float rounded-2xl border border-ok/40 bg-surface px-4 py-3 shadow-xl sm:-left-10">
        <p className="text-[12.5px] font-bold text-ok">수정안 · 팀 리드 승인 대기</p>
        <p className="mt-1 text-[15px] leading-snug font-bold whitespace-nowrap">+ 상향: 김서연 사원 → 박재현 팀장 (5분, 전화)</p>
      </div>
      <div className="absolute -top-6 -right-3 animate-float rounded-2xl border border-danger/40 bg-surface px-4 py-2.5 shadow-xl [animation-delay:1.5s] sm:-right-8">
        <p className="text-[12.5px] font-bold text-fg-subtle">리허설 · 추석 새벽 결제 장애</p>
        <p className="text-[17px] font-black text-danger">FAIL · 김대리에서 막힘</p>
      </div>
      <style>{`
        .hero-flow { stroke-dasharray: 8 6; animation: flow 1.1s linear infinite; }
        .hero-pulse { animation: hero-pulse 2.2s ease-in-out infinite; transform-origin: 417px 86px; }
        @keyframes hero-pulse { 0%,100% { opacity: 1 } 50% { opacity: .72 } }
      `}</style>
    </div>
  );
}

function EdgeLabel({ x, y, text, danger = false }: { x: number; y: number; text: string; danger?: boolean }) {
  const w = text.length * 14 + 16;
  return (
    <g>
      <rect x={x - w / 2} y={y - 11} width={w} height={22} rx={11} fill="var(--surface)" stroke={danger ? 'var(--danger)' : 'var(--line-strong)'} />
      <text x={x} y={y + 4.5} textAnchor="middle" fontSize="12.5" fontWeight="800" fill={danger ? 'var(--danger)' : 'var(--fg-muted)'}>
        {text}
      </text>
    </g>
  );
}

function Ghost({ x, y, label }: { x: number; y: number; label: string }) {
  return (
    <g>
      <rect x={x - 6} y={y} width={96} height={36} rx={18} fill="color-mix(in oklab, var(--danger) 8%, transparent)" stroke="var(--ghost)" strokeWidth="2" strokeDasharray="5 4" />
      <text x={x + 42} y={y + 23} textAnchor="middle" fontSize="13" fontWeight="800" fill="var(--ghost)">
        {label}
      </text>
    </g>
  );
}
