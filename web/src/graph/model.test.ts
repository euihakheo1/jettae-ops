import ElkBundled from 'elkjs/lib/elk.bundled.js';
import { describe, expect, it } from 'vitest';
import type { Finding, GraphView } from '../api/types';
import { findings, readiness, view } from '../test/fixtures';
import { layoutWithElk, toElkGraph } from './layout';
import { splinePath, splinePoint } from './OpsEdge';
import {
  applyHighlight,
  buildGraphModel,
  buildIndex,
  defaultFocusService,
  deriveGhosts,
  focusSubgraph,
  pathSubgraphModel,
  type OpsEdgeData,
} from './model';

const PAY = 'service:결제';
const focusPay = { focus: { mode: 'service', serviceId: PAY } } as const;

describe('defaultFocusService', () => {
  it('가장 중요한 등급(1등급) 중 점수가 가장 낮은 결제를 고른다', () => {
    expect(defaultFocusService(readiness)).toBe(PAY);
  });

  it('준비도가 없으면 첫 서비스 노드를 고른다', () => {
    expect(defaultFocusService(null, view)).toBe('service:결제'); // 코드 포인트 순: 결 < 광
  });
});

describe('focusSubgraph', () => {
  const ix = buildIndex(view);

  it('결제: 감지 신호·담당(퇴사자)·조치·연락 채널을 포함하고 무관한 서비스는 뺀다', () => {
    const { nodeIds, edgeIds } = focusSubgraph(ix, PAY);
    expect(nodeIds.has(PAY)).toBe(true);
    expect(nodeIds.has('trigger:결제-서비스-실패율-알람5-초과-3분-지속')).toBe(true);
    expect(nodeIds.has('trigger:결제-서비스-장애-지속')).toBe(true);
    expect(nodeIds.has('person:김민준')).toBe(true);
    expect(nodeIds.has('action:결제-서비스-롤백')).toBe(true);
    expect(nodeIds.has('channel:phone')).toBe(true);
    expect(nodeIds.has('service:주문')).toBe(false);
    expect(nodeIds.has('person:박재현')).toBe(false);
    expect(edgeIds.has('owned_by:service:결제->person:김민준')).toBe(true);
    // 구조 엣지(소속)는 집중 보기에 넣지 않는다
    expect([...edgeIds].some((e) => e.startsWith('member_of:'))).toBe(false);
  });

  it('주문: 상향 순환(정민혁 ↔ 최영훈)과 승인자 한서윤을 따라간다', () => {
    const { nodeIds, edgeIds } = focusSubgraph(ix, 'service:주문');
    expect(nodeIds.has('person:정민혁')).toBe(true);
    expect(nodeIds.has('person:최영훈')).toBe(true);
    expect(nodeIds.has('person:한서윤')).toBe(true);
    expect(edgeIds.has('escalates_to:person:정민혁->person:최영훈')).toBe(true);
    expect(edgeIds.has('escalates_to:person:최영훈->person:정민혁')).toBe(true);
  });

  it('웹 프론트엔드: 박재현 → 팀장 역할 → 한서윤 CTO 상향까지 역할을 해석한다', () => {
    const { nodeIds, edgeIds } = focusSubgraph(ix, 'service:웹-프론트엔드');
    expect(nodeIds.has('person:박재현')).toBe(true);
    expect(nodeIds.has('role:팀장')).toBe(true);
    expect(nodeIds.has('person:한서윤')).toBe(true);
    expect(edgeIds.has('holds_role:person:박재현->role:팀장')).toBe(true);
    expect(edgeIds.has('escalates_to:role:팀장->person:한서윤')).toBe(true);
    // 사내 메신저 → 전화 대체 채널
    expect(edgeIds.has('fallback_to:channel:internal_messenger->channel:phone')).toBe(true);
  });

  it('없는 서비스면 빈 집합', () => {
    expect(focusSubgraph(ix, 'service:없음').nodeIds.size).toBe(0);
  });
});

describe('buildGraphModel — 유령(빠진 요소) 노드', () => {
  const model = buildGraphModel({ view, findings, readiness }, focusPay);
  const ghost = (id: string) => model.nodes.find((n) => n.id === id);

  it('JT-PATH-020 → 롤백 조치에 붙은 "승인자 없음"', () => {
    const g = ghost('ghost:JT-PATH-020:action:결제-서비스-롤백');
    expect(g?.data.kind).toBe('ghost');
    expect(g?.data.label).toBe('승인자 없음');
    expect(g?.data.ghostOf).toBe('person');
    const e = model.edges.find((x) => x.target === g?.id);
    expect(e?.source).toBe('action:결제-서비스-롤백');
    expect((e?.data as OpsEdgeData).label).toBe('승인');
    expect((e?.data as OpsEdgeData).tone).toBe('ghost');
    expect(e?.animated).toBe(true);
  });

  it('JT-ESC-001 → 담당자(김민준)에서 나가는 "상향 경로 없음"', () => {
    const g = ghost('ghost:JT-ESC-001:person:김민준');
    expect(g?.data.label).toBe('상향 경로 없음');
    expect(model.edges.some((e) => e.source === 'person:김민준' && e.target === g?.id)).toBe(true);
  });

  it('JT-PATH-030 → "대체 채널 없음", JT-PATH-010 → "담당자 없음"', () => {
    expect(ghost('ghost:JT-PATH-030:person:김민준')?.data.label).toBe('대체 채널 없음');
    expect(ghost(`ghost:JT-PATH-010:${PAY}`)?.data.label).toBe('담당자 없음');
  });

  it('국내 신고 기준(JT-KR-001/003) 유령은 서비스에 붙는다', () => {
    expect(ghost('ghost:JT-KR-001')?.data.label).toBe('침해사고 신고 단계 없음');
    expect(model.edges.some((e) => e.source === PAY && e.target === 'ghost:JT-KR-003')).toBe(true);
  });

  it('감지 경로 없음(JT-PATH-001)은 서비스 왼쪽(FIRST 레이어)으로 들어온다', () => {
    const m = buildGraphModel({ view, findings, readiness }, { focus: { mode: 'service', serviceId: 'service:배송' } });
    const g = m.nodes.find((n) => n.id === 'ghost:JT-PATH-001:service:배송');
    expect(g?.data.label).toBe('감지 신호 없음');
    expect(m.layer[g!.id]).toBe('FIRST');
    expect(m.edges.find((e) => e.source === g!.id)?.target).toBe('service:배송');
  });

  it('ghosts:false 이면 유령 노드를 만들지 않는다', () => {
    const m = buildGraphModel({ view, findings, readiness }, { ...focusPay, ghosts: false });
    expect(m.nodes.some((n) => n.data.kind === 'ghost')).toBe(false);
  });

  it('같은 유령은 하나로 합친다 (규칙+앵커 기준)', () => {
    const f = findings.find((x) => x.rule_id === 'JT-PATH-020' && x.service_id === PAY)!;
    const dup: Finding = { ...f, id: `${f.id}-dup` };
    const ghosts = deriveGhosts([f, dup], new Set([PAY, 'action:결제-서비스-롤백']));
    expect(ghosts).toHaveLength(1);
    expect(ghosts[0]!.findingIds).toEqual([f.id, dup.id].sort());
  });

  it('앵커가 화면에 없으면 유령을 만들지 않는다', () => {
    const f = findings.find((x) => x.rule_id === 'JT-PATH-020' && x.service_id === PAY)!;
    expect(deriveGhosts([f], new Set([PAY]))).toHaveLength(0);
  });
});

describe('buildGraphModel — 상태 표시', () => {
  const model = buildGraphModel({ view, findings, readiness }, focusPay);

  it('퇴사자 김민준: departed 플래그 + "퇴사" 배지 + 치명 심각도', () => {
    const kim = model.nodes.find((n) => n.id === 'person:김민준')!;
    expect(kim.data.departed).toBe(true);
    expect(kim.data.personStatus).toBe('departed');
    expect(kim.data.badges).toContainEqual({ text: '퇴사', tone: 'danger' });
    expect(kim.data.severity).toBe('critical');
    expect(kim.data.sublabel).toContain('대리');
  });

  it('끊김 subject 엣지는 빨강·애니메이션, 무관한 엣지는 보통', () => {
    const owned = model.edges.find((e) => e.id === 'owned_by:service:결제->person:김민준')!;
    expect((owned.data as OpsEdgeData).tone).toBe('broken');
    expect(owned.animated).toBe(true);
    const reach = model.edges.find((e) => e.id === 'reachable_via:person:김민준->channel:phone')!;
    expect((reach.data as OpsEdgeData).tone).toBe('normal');
    expect(reach.animated).toBe(false);
  });

  it('감지 엣지는 감지 신호 → 서비스 방향으로 뒤집어 그린다', () => {
    const det = model.edges.find((e) => e.id === 'detected_by:service:결제->trigger:결제-서비스-장애-지속')!;
    expect(det.source).toBe('trigger:결제-서비스-장애-지속');
    expect(det.target).toBe(PAY);
    expect((det.data as OpsEdgeData).reversed).toBe(true);
    expect(model.layer['trigger:결제-서비스-장애-지속']).toBe('FIRST');
  });

  it('서비스 노드는 등급·점수를 가진다', () => {
    const pay = model.nodes.find((n) => n.id === PAY)!;
    expect(pay.data.tier).toBe(1);
    expect(pay.data.score).toBe(22);
    expect(pay.data.isFocus).toBe(true);
  });

  it('상향 엣지 라벨에 대기 시간을 붙인다', () => {
    const m = buildGraphModel({ view, findings, readiness }, { focus: { mode: 'service', serviceId: 'service:주문' } });
    const e = m.edges.find((x) => x.id === 'escalates_to:person:정민혁->person:최영훈')!;
    expect((e.data as OpsEdgeData).label).toBe('상향 15분');
  });

  it('HOLD 요소는 "판단 보류" 배지와 hold 플래그', () => {
    const held: GraphView = {
      ...view,
      nodes: view.nodes.map((n) => (n.id === 'person:박재현' ? { ...n, verification: 'hold' as const, hold_reason: '동명이인' } : n)),
    };
    const m = buildGraphModel({ view: held, findings, readiness }, { focus: { mode: 'service', serviceId: 'service:장바구니' } });
    const park = m.nodes.find((n) => n.id === 'person:박재현')!;
    expect(park.data.hold).toBe(true);
    expect(park.data.holdReason).toBe('동명이인');
    expect(park.data.badges).toContainEqual({ text: '판단 보류', tone: 'warn' });
  });

  it('휴직자는 onLeave, 전체 보기에서 구조 엣지는 흐리게', () => {
    const m = buildGraphModel({ view, findings, readiness }, { focus: { mode: 'all' } });
    const bae = m.nodes.find((n) => n.id === 'person:배은지')!;
    expect(bae.data.onLeave).toBe(true);
    expect(bae.data.badges).toContainEqual({ text: '휴직', tone: 'warn' });
    const member = m.edges.find((e) => e.id === 'member_of:person:배은지->team:개발팀')!;
    expect((member.data as OpsEdgeData).tone).toBe('faded');
  });

  it('hideStructural 이면 소속·역할·위치·의존 엣지를 뺀다', () => {
    const m = buildGraphModel({ view, findings, readiness }, { focus: { mode: 'all' }, hideStructural: true });
    expect(m.edges.some((e) => e.id.startsWith('member_of:') || e.id.startsWith('runs_in:'))).toBe(false);
    // 떠 있는 위치 노드는 빠지고, 연결이 없는 서비스(정산)는 남는다
    expect(m.nodes.some((n) => n.id === 'location:상암-idc')).toBe(false);
    expect(m.nodes.some((n) => n.id === 'service:정산')).toBe(true);
  });
});

describe('결정성', () => {
  it('입력 순서가 달라도 같은 노드·엣지 순서를 낸다', () => {
    const a = buildGraphModel({ view, findings, readiness }, focusPay);
    const shuffled: GraphView = { ...view, nodes: [...view.nodes].reverse(), edges: [...view.edges].reverse() };
    const b = buildGraphModel({ view: shuffled, findings: [...findings].reverse(), readiness }, focusPay);
    expect(b.nodes.map((n) => n.id)).toEqual(a.nodes.map((n) => n.id));
    expect(b.edges.map((e) => e.id)).toEqual(a.edges.map((e) => e.id));
    expect(JSON.stringify(b.nodes.map((n) => n.data))).toEqual(JSON.stringify(a.nodes.map((n) => n.data)));
  });

  it('노드는 id 순으로 정렬되어 있다', () => {
    const m = buildGraphModel({ view, findings, readiness }, { focus: { mode: 'all' } });
    const ids = m.nodes.map((n) => n.id);
    expect(ids).toEqual([...ids].sort());
  });

  it('ELK 입력이 같고, 레이아웃 결과(좌표)도 같다 · 감지 → 서비스 → 담당 순으로 왼쪽에서 오른쪽', async () => {
    const model = buildGraphModel({ view, findings, readiness }, focusPay);
    expect(toElkGraph(model)).toEqual(toElkGraph(buildGraphModel({ view, findings, readiness }, focusPay)));
    const elk = new ElkBundled();
    const r1 = await layoutWithElk(elk, model);
    const r2 = await layoutWithElk(elk, model);
    expect(r1.positions).toEqual(r2.positions);
    const x = (id: string) => r1.positions[id]!.x;
    expect(x('trigger:결제-서비스-장애-지속')).toBeLessThan(x(PAY));
    expect(x(PAY)).toBeLessThan(x('person:김민준'));
    expect(x('person:김민준')).toBeLessThan(x('channel:phone'));
  });
});

describe('ELK 연결 경로', () => {
  it('모든 연결에 3차 베지어 조각 경로(1 + 3k 점)가 있고, 양 끝이 두 노드 경계에 붙는다', async () => {
    const model = buildGraphModel({ view, findings, readiness }, focusPay);
    const r = await layoutWithElk(new ElkBundled(), model);
    for (const e of model.edges) {
      const route = r.routes[e.id];
      expect(route, e.id).toBeDefined();
      expect((route!.points.length - 1) % 3, e.id).toBe(0);
      const src = model.nodes.find((n) => n.id === e.source)!;
      const p0 = route!.points[0]!;
      const pos = r.positions[e.source]!;
      // 시작점은 출발 노드의 테두리 위(오른쪽 변 근처)에 있다
      expect(p0.x).toBeGreaterThanOrEqual(pos.x - 1);
      expect(p0.x).toBeLessThanOrEqual(pos.x + (src.width ?? 0) + 1);
    }
  });

  it('스플라인 path·중간점 계산', () => {
    const pts = [
      { x: 0, y: 0 },
      { x: 10, y: 0 },
      { x: 20, y: 10 },
      { x: 30, y: 10 },
    ];
    expect(splinePath(pts)).toBe('M0,0 C10,0 20,10 30,10');
    expect(splinePoint(pts, 0)).toEqual({ x: 0, y: 0 });
    expect(splinePoint(pts, 1)).toEqual({ x: 30, y: 10 });
    expect(splinePoint(pts, 0.5)).toEqual({ x: 15, y: 5 });
  });
});

describe('applyHighlight / pathSubgraphModel', () => {
  it('리허설 경로만 강조하고 나머지는 흐리게', () => {
    const model = buildGraphModel({ view, findings, readiness }, focusPay);
    const h = applyHighlight(model, {
      ids: [PAY, 'person:김민준'],
      currentId: 'person:김민준',
      failedId: 'person:김민준',
      dimOthers: true,
    });
    const kim = h.nodes.find((n) => n.id === 'person:김민준')!;
    expect(kim.data.highlighted && kim.data.current && kim.data.failed).toBe(true);
    expect(h.nodes.find((n) => n.id === 'action:결제-서비스-롤백')!.data.dimmed).toBe(true);
    // 원본은 그대로
    expect(model.nodes.find((n) => n.id === 'person:김민준')!.data.highlighted).toBeUndefined();
  });

  it('끊김 강조(finding)는 빨강을 유지하고, 지정한 요소(집중 서비스)는 흐리지 않는다', () => {
    const model = buildGraphModel({ view, findings, readiness }, focusPay);
    const h = applyHighlight(model, { ids: ['person:김민준'], dimOthers: true, style: 'finding', keepIds: [PAY] });
    const kim = h.nodes.find((n) => n.id === 'person:김민준')!;
    expect(kim.data.highlighted).toBe(true);
    expect(kim.data.highlightStyle).toBe('finding');
    expect(h.nodes.find((n) => n.id === PAY)!.data.dimmed).toBe(false);
    expect(h.nodes.find((n) => n.id === 'action:결제-서비스-롤백')!.data.dimmed).toBe(true);
  });

  it('경로 그래프는 path_ids의 노드·엣지와 실패 지점만 담는다', () => {
    const path = [PAY, 'detected_by:service:결제->trigger:결제-서비스-장애-지속', 'notifies:trigger:결제-서비스-장애-지속->person:김민준'];
    const m = pathSubgraphModel({ view, findings, readiness }, path, 'person:김민준');
    expect(m.nodes.map((n) => n.id).sort()).toEqual([PAY, 'person:김민준', 'trigger:결제-서비스-장애-지속'].sort());
    expect(m.edges).toHaveLength(2);
  });
});
