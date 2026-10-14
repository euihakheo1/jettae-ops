import { clsx } from 'clsx';
import { ArrowRight, CircleCheck, CircleX, FlaskConical, Pause, Play, PlayCircle, RotateCcw, SkipBack, SkipForward, TriangleAlert } from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';
import { useParams, useSearchParams } from 'react-router';
import { errorMessage } from '../api/errors';
import { useGraph, useReport, useRunRehearsals } from '../api/hooks';
import type { DiagnosisReport, GraphView, RehearsalResult, Scenario } from '../api/types';
import { usePermissions } from '../app/permissions';
import { Button } from '../components/ui/Button';
import { EmptyState, ErrorState, LoadingBlock } from '../components/ui/States';
import { useToast } from '../components/ui/Toast';
import { DisabledReason } from '../components/ui/Tooltip';
import { GraphCanvas } from '../graph/GraphCanvas';
import { applyHighlight, pathSubgraphModel } from '../graph/model';
import { formatMinutes, formatPercent, hashPrefix, toServiceId } from '../lib/format';
import { ELEMENT_LABEL, INCIDENT_TYPE_LABEL } from '../lib/labels';
import { PageHeader } from './PageHeader';
import { DeadlineBar, nodeLabel, OutcomeBadge, PerturbationChip, ScenarioCard, Timeline } from './rehearsals/parts';
import { usePlayback } from './rehearsals/usePlayback';
import { currentRehearsals } from '../lib/rehearsals';

export function Rehearsals() {
  const { ws = '' } = useParams();
  const [params, setParams] = useSearchParams();
  const report = useReport(ws);
  const base = useReport(ws, 1);
  const graph = useGraph(ws);
  const toast = useToast();
  const [overrides, setOverrides] = useState<Record<string, RehearsalResult>>({});
  const [runningId, setRunningId] = useState<string | null>(null);
  const [replayToken, setReplayToken] = useState(0);
  const run = useRunRehearsals(ws);
  // 리허설 실행은 편집자 이상(공개 체험판 방문자는 저장된 결과만 재생해 본다)
  const runBlocked = usePermissions().editBlocked;

  const hash = report.data?.graph_hash;
  useEffect(() => setOverrides({}), [hash]);

  const onlyFail = params.get('only') === 'fail';
  const svc = params.get('svc') ? toServiceId(params.get('svc')!) : '';

  const results = useMemo(() => {
    const m = new Map<string, RehearsalResult>();
    for (const r of report.data ? currentRehearsals(report.data) : []) m.set(r.scenario_id, r);
    for (const [k, r] of Object.entries(overrides)) if (r.graph_hash === hash) m.set(k, r);
    return m;
  }, [report.data, overrides, hash]);

  if (report.isError) return <ErrorState error={report.error} onRetry={() => void report.refetch()} />;
  if (!report.data) return <LoadingBlock className="p-8" />;
  const data = report.data;
  const scenarios = data.scenarios;
  if (!scenarios.length)
    return (
      <EmptyState icon={FlaskConical} title="시나리오가 없습니다" className="mt-12">
        진단을 실행하면 템플릿 시나리오(기준·담당 부재·채널 다운·센터 장애·동시 장애)가 만들어집니다.
      </EmptyState>
    );

  const svcName = (id: string) => data.readiness.services.find((s) => s.service_id === id)?.service_name ?? id.replace(/^service:/, '');
  const visible = scenarios.filter((s) => (!onlyFail || results.get(s.id)?.outcome === 'fail') && (!svc || s.service_id === svc));
  const human = visible.filter((s) => s.origin !== 'template');
  const templates = visible.filter((s) => s.origin === 'template');
  const selectedId =
    params.get('scenario') ??
    scenarios.find((s) => s.origin !== 'template' && results.get(s.id)?.outcome === 'fail')?.id ??
    scenarios.find((s) => results.get(s.id)?.outcome === 'fail')?.id ??
    scenarios[0]!.id;
  const selected = scenarios.find((s) => s.id === selectedId) ?? scenarios[0]!;
  const passed = [...results.values()].filter((r) => r.outcome === 'pass').length;

  const setParam = (k: string, v: string | null) =>
    setParams(
      (prev) => {
        const n = new URLSearchParams(prev);
        if (v === null) n.delete(k);
        else n.set(k, v);
        if (k === 'scenario') {
          n.delete('step');
          n.delete('play');
        }
        return n;
      },
      { replace: true },
    );

  const runOne = async (s: Scenario) => {
    setRunningId(s.id);
    setParam('scenario', s.id);
    try {
      const res = await run.mutateAsync({ scenario_ids: [s.id] });
      const r = res.find((x) => x.scenario_id === s.id);
      if (r) {
        setOverrides((o) => ({ ...o, [s.id]: r }));
        setReplayToken((t) => t + 1);
        toast.show({
          tone: r.outcome === 'pass' ? 'success' : 'error',
          title: `${s.title} — ${r.outcome === 'pass' ? `통과 (${formatMinutes(r.time_to_mitigate)})` : '실패'}`,
          description: r.reason,
        });
      }
    } catch (e) {
      toast.show({ tone: 'error', title: '리허설을 실행하지 못했습니다', description: errorMessage(e) });
    } finally {
      setRunningId(null);
    }
  };

  const runAll = async () => {
    setRunningId('__all');
    try {
      const res = await run.mutateAsync({});
      setOverrides(Object.fromEntries(res.map((r) => [r.scenario_id, r])));
      const ok = res.filter((r) => r.outcome === 'pass').length;
      toast.show({ tone: 'info', title: `리허설 ${res.length}건 재생 · 통과 ${ok} · 실패 ${res.length - ok}`, description: '결정적 리플레이 — 같은 그래프·같은 시나리오는 언제나 같은 결과' });
    } catch (e) {
      toast.show({ tone: 'error', title: '리허설을 실행하지 못했습니다', description: errorMessage(e) });
    } finally {
      setRunningId(null);
    }
  };

  const baseline = base.data && base.data.graph_version !== data.graph_version ? base.data.rehearsals.find((r) => r.scenario_id === selected.id) : undefined;
  const services = [...new Set(scenarios.map((s) => s.service_id))].sort((a, b) => svcName(a).localeCompare(svcName(b), 'ko'));

  return (
    <div className="flex min-h-full flex-col p-4 2xl:p-6">
      <PageHeader
        icon={FlaskConical}
        title="모의 장애 리허설"
        description="운영 그래프 위에서 장애를 결정적으로 재생합니다. 실제 시스템은 건드리지 않고, 같은 그래프·같은 시나리오는 언제나 같은 결과를 냅니다."
        actions={
          <>
            <span className="tnum hidden text-[15px] text-fg-muted lg:inline">
              작동률 <strong className="text-fg">{formatPercent(passed, results.size)}</strong> ({passed}/{results.size})
            </span>
            <DisabledReason reason={runBlocked}>
              <Button
                variant="primary"
                icon={<PlayCircle className="size-4" aria-hidden />}
                loading={runningId === '__all'}
                disabled={!!runBlocked}
                onClick={() => void runAll()}
              >
                전체 다시 재생
              </Button>
            </DisabledReason>
          </>
        }
      />
      <div className="mt-5 grid min-h-0 flex-1 gap-4 xl:grid-cols-[400px_minmax(0,1fr)] 2xl:grid-cols-[440px_minmax(0,1fr)]">
        <div className="flex min-h-0 flex-col gap-3">
          <div className="flex flex-wrap items-center gap-2">
            <select
              aria-label="서비스 필터"
              value={svc}
              onChange={(e) => setParam('svc', e.target.value ? e.target.value.replace(/^service:/, '') : null)}
              className="h-9 rounded-xl border border-line bg-surface-2 px-3 text-[14px]"
            >
              <option value="">모든 서비스</option>
              {services.map((s) => (
                <option key={s} value={s}>
                  {svcName(s)}
                </option>
              ))}
            </select>
            <button
              type="button"
              aria-pressed={onlyFail}
              onClick={() => setParam('only', onlyFail ? null : 'fail')}
              className={clsx('h-9 rounded-xl border px-3 text-[14px] font-semibold', onlyFail ? 'border-danger/60 bg-danger/12 text-danger' : 'border-line text-fg-muted hover:text-fg')}
            >
              실패만 보기
            </button>
            <span className="tnum ml-auto text-[13px] text-fg-subtle">{visible.length}개</span>
          </div>
          <div className="min-h-0 flex-1 space-y-5 overflow-y-auto pr-1 xl:max-h-[calc(100vh-230px)]">
            {human.length ? (
              <section>
                <h2 className="mb-2 text-[14px] font-bold text-fg-muted">사람이 작성한 시나리오</h2>
                <div className="space-y-2.5">
                  {human.map((s) => (
                    <ScenarioCard key={s.id} scenario={s} result={results.get(s.id)} view={graph.data} serviceName={svcName(s.service_id)} active={s.id === selected.id} running={runningId === s.id} runBlocked={runBlocked} onSelect={() => setParam('scenario', s.id)} onRun={() => void runOne(s)} />
                  ))}
                </div>
              </section>
            ) : null}
            {templates.length ? (
              <section>
                <h2 className="mb-2 text-[14px] font-bold text-fg-muted">템플릿 시나리오 {templates.length}</h2>
                <div className="space-y-2.5">
                  {templates.map((s) => (
                    <ScenarioCard key={s.id} scenario={s} result={results.get(s.id)} view={graph.data} serviceName={svcName(s.service_id)} active={s.id === selected.id} running={runningId === s.id} runBlocked={runBlocked} onSelect={() => setParam('scenario', s.id)} onRun={() => void runOne(s)} />
                  ))}
                </div>
              </section>
            ) : null}
            {!visible.length ? <EmptyState title="조건에 맞는 시나리오가 없습니다" /> : null}
          </div>
        </div>
        {(() => {
          const result = results.get(selected.id);
          return result ? (
            <ResultPanel
              key={`${selected.id}:${result.graph_hash}:${replayToken}`}
              scenario={selected}
              result={result}
              baseline={baseline}
              report={data}
              baseVersion={base.data?.graph_version ?? 1}
              view={graph.data}
              autoplay={params.get('play') === '1' || replayToken > 0}
              initialStep={params.get('step') ? Number(params.get('step')) : undefined}
              serviceName={svcName(selected.service_id)}
              running={runningId === selected.id}
              runBlocked={runBlocked}
              onRun={() => void runOne(selected)}
            />
          ) : (
            <div className="card grid place-items-center p-10">
              <EmptyState
                icon={Play}
                title="아직 실행하지 않은 시나리오입니다"
                action={
                  <DisabledReason reason={runBlocked}>
                    <Button variant="primary" disabled={!!runBlocked} onClick={() => void runOne(selected)} loading={runningId === selected.id}>
                      지금 실행
                    </Button>
                  </DisabledReason>
                }
              />
            </div>
          );
        })()}
      </div>
    </div>
  );
}

function ResultPanel({
  scenario,
  result,
  baseline,
  report,
  baseVersion,
  view,
  autoplay,
  initialStep,
  serviceName,
  running,
  runBlocked,
  onRun,
}: {
  scenario: Scenario;
  result: RehearsalResult;
  baseline: RehearsalResult | undefined;
  report: DiagnosisReport;
  baseVersion: number;
  view: GraphView | undefined;
  autoplay: boolean;
  initialStep: number | undefined;
  serviceName: string;
  running: boolean;
  runBlocked: string | null;
  onRun: () => void;
}) {
  const total = result.events.length;
  const pb = usePlayback(total, { initial: autoplay ? 0 : (initialStep ?? total), autoplay, intervalMs: 1000 });
  const pass = result.outcome === 'pass';

  const pathModel = useMemo(
    () => (view ? pathSubgraphModel({ view, findings: report.findings, readiness: report.readiness }, result.path_ids, result.failed_at_id) : null),
    [view, report, result],
  );
  const model = useMemo(() => {
    if (!pathModel) return null;
    const present = new Set([...pathModel.nodes.map((n) => n.id), ...pathModel.edges.map((e) => e.id)]);
    const ids = new Set<string>([scenario.service_id]);
    let current: string | null = null;
    for (const e of result.events.slice(0, pb.step)) {
      for (const id of [e.edge_id, e.actor_id, e.target_id]) if (id && present.has(id)) ids.add(id);
      current = (e.edge_id && present.has(e.edge_id) ? e.edge_id : null) ?? (e.target_id && present.has(e.target_id) ? e.target_id : null) ?? current;
    }
    const done = pb.step >= total;
    if (done) for (const id of result.path_ids) if (present.has(id)) ids.add(id);
    return applyHighlight(pathModel, {
      ids: [...ids],
      currentId: done ? null : current,
      failedId: done && !pass ? result.failed_at_id : null,
      dimOthers: true,
    });
  }, [pathModel, result, pb.step, total, pass, scenario.service_id]);

  return (
    <section aria-label="리허설 결과" className="card flex min-w-0 flex-col overflow-hidden">
      <header className="flex flex-wrap items-start gap-4 border-b border-line p-5">
        <div
          className={clsx(
            'grid size-[88px] shrink-0 place-items-center rounded-2xl border-2 text-center',
            pass ? 'border-ok/60 bg-ok/12 text-ok' : 'border-danger/60 bg-danger/12 text-danger',
          )}
        >
          <div>
            {pass ? <CircleCheck className="mx-auto size-8" aria-hidden /> : <CircleX className="mx-auto size-8" aria-hidden />}
            <p className="mt-1 text-[22px] leading-none font-black tracking-wider">{pass ? 'PASS' : 'FAIL'}</p>
          </div>
        </div>
        <div className="min-w-0 flex-1">
          <p className="text-[13px] font-semibold text-fg-subtle">
            {serviceName} · {INCIDENT_TYPE_LABEL[scenario.incident_type]} · {pass ? '대응 경로 작동' : '대응 경로 막힘'}
          </p>
          <h2 className="mt-0.5 text-[22px] leading-snug font-extrabold">{scenario.title}</h2>
          {scenario.description ? <p className="mt-1 text-[14.5px] leading-relaxed text-fg-muted">{scenario.description}</p> : null}
          {scenario.perturbations.length ? (
            <div className="mt-2 flex flex-wrap gap-1.5">
              {scenario.perturbations.map((p, i) => (
                <PerturbationChip key={i} p={p} view={view} />
              ))}
            </div>
          ) : null}
        </div>
        <DisabledReason reason={runBlocked}>
          <Button variant="secondary" icon={<RotateCcw className="size-4" aria-hidden />} loading={running} disabled={!!runBlocked} onClick={onRun}>
            다시 실행
          </Button>
        </DisabledReason>
      </header>

      {/* 대응 경로 재생 — 화면의 주인공이라 패널 폭 전체를 쓴다 */}
      <div className="border-b border-line px-5 pt-4 pb-5">
        <div className="mb-2 flex items-center gap-2">
          <h3 className="text-[13px] font-bold tracking-wide text-fg-subtle">대응 경로 재생</h3>
          <span className="text-[13px] text-fg-subtle">· 감지 → 호출 → 응답 → 승인 → 조치 순서로 한 단계씩</span>
          <div className="ml-auto flex items-center gap-1">
            <IconBtn label="처음으로" onClick={pb.reset} icon={SkipBack} />
            {pb.playing ? <IconBtn label="일시 정지" onClick={pb.pause} icon={Pause} primary /> : <IconBtn label="재생" onClick={pb.play} icon={Play} primary />}
            <IconBtn label="다음 단계" onClick={() => pb.setStep(pb.step + 1)} icon={SkipForward} />
            <span className="tnum ml-2 text-[13px] text-fg-muted">
              {pb.step}/{total}
            </span>
          </div>
        </div>
        <div className="h-[300px] overflow-hidden rounded-2xl border border-line bg-bg/40 2xl:h-[320px]">
          {model ? <GraphCanvas model={model} ariaLabel="리허설 경로 그래프" compact minimap={false} controls={false} /> : null}
        </div>
      </div>

      <div className="grid gap-5 p-5 2xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
        <div className="min-w-0 space-y-4">
          <div>
            <h3 className="mb-2 text-[13px] font-bold tracking-wide text-fg-subtle">완화 시간 vs 목표 시한</h3>
            <DeadlineBar result={result} />
          </div>
          {!pass ? (
            <div role="alert" className="flex gap-3 rounded-2xl border border-danger/50 bg-danger/10 p-4">
              <TriangleAlert className="mt-0.5 size-6 shrink-0 text-danger" aria-hidden />
              <div>
                <p className="text-[16px] font-extrabold text-danger">
                  끊긴 지점{result.failed_element ? ` · ${ELEMENT_LABEL[result.failed_element]}` : ''}
                  {result.failed_at_id ? ` — ${nodeLabel(view, result.failed_at_id)}` : ''}
                </p>
                <p className="mt-1 text-[15px] leading-relaxed">{result.reason}</p>
              </div>
            </div>
          ) : (
            <p className="rounded-2xl border border-ok/40 bg-ok/8 p-4 text-[15px] font-semibold">{result.reason}</p>
          )}
          {baseline ? <Compare baseline={baseline} current={result} baseVersion={baseVersion} currentVersion={report.graph_version} /> : null}
          <p className="truncate font-mono text-[12px] text-fg-subtle" title="같은 입력이면 같은 결과">
            graph {hashPrefix(result.graph_hash, 12)} · config {hashPrefix(result.config_hash, 12)} · scenario {result.scenario_fingerprint} — 같은 입력이면 같은 결과
          </p>
        </div>
        <div className="min-w-0">
          <h3 className="mb-2 text-[13px] font-bold tracking-wide text-fg-subtle">타임라인 (분 단위 · 장애 시작 = t+0)</h3>
          <Timeline events={result.events} revealed={pb.step} onJump={pb.setStep} view={view} />
        </div>
      </div>
    </section>
  );
}

function IconBtn({ label, onClick, icon: Icon, primary = false }: { label: string; onClick: () => void; icon: typeof Play; primary?: boolean }) {
  return (
    <button
      type="button"
      aria-label={label}
      title={label}
      onClick={onClick}
      className={clsx('grid size-9 place-items-center rounded-lg border', primary ? 'border-brand bg-brand text-brand-fg' : 'border-line bg-surface-2 text-fg-muted hover:text-fg')}
    >
      <Icon className="size-4" aria-hidden />
    </button>
  );
}

function Compare({ baseline, current, baseVersion, currentVersion }: { baseline: RehearsalResult; current: RehearsalResult; baseVersion: number; currentVersion: number }) {
  const changed = baseline.outcome !== current.outcome || baseline.time_to_mitigate !== current.time_to_mitigate;
  return (
    <div className="rounded-2xl border border-line bg-surface-2 p-4">
      <h3 className="mb-3 text-[13px] font-bold tracking-wide text-fg-subtle">수정 전후 비교 {changed ? '' : '· 변화 없음'}</h3>
      <div className="grid grid-cols-[1fr_auto_1fr] items-center gap-3">
        <Side label={`v${baseVersion} (수정 전)`} r={baseline} />
        <ArrowRight className="size-6 text-fg-subtle" aria-hidden />
        <Side label={`v${currentVersion} (지금)`} r={current} />
      </div>
    </div>
  );
}

function Side({ label, r }: { label: string; r: RehearsalResult }) {
  return (
    <div className={clsx('rounded-xl border p-3', r.outcome === 'pass' ? 'border-ok/50 bg-ok/8' : 'border-danger/50 bg-danger/8')}>
      <p className="text-[12.5px] font-semibold text-fg-subtle">{label}</p>
      <div className="mt-1 flex items-center gap-2">
        <OutcomeBadge result={r} size="lg" />
        <span className="tnum text-[15px] font-bold">{r.time_to_mitigate !== null ? formatMinutes(r.time_to_mitigate) : '완화 못 함'}</span>
      </div>
      <p className="mt-1 line-clamp-2 text-[13px] text-fg-muted">{r.reason}</p>
    </div>
  );
}
