#!/usr/bin/env node
/**
 * 제때옵스 웹 — 화면 스크린숏 (발표 자료·시각 QA용)
 *
 * 기본(목업): VITE_USE_MOCKS=1 로 별도 폴더(.shots-dist)에 빌드 → vite preview(빈 포트) → 캡처 → 서버 종료
 *   node scripts/screenshots.mjs
 *
 * 실제 API: 이미 떠 있는 서버(jettae serve 또는 npm run dev)에 붙어서 캡처만 한다
 *   node scripts/screenshots.mjs --live --base http://127.0.0.1:8765 [--ws dmc-shop]
 *
 * 옵션
 *   --out <dir>          저장 폴더 (기본 web/screenshots)
 *   --only a,b           이름 일부가 맞는 장면만 (예: --only dashboard,trust)
 *   --themes dark,projector
 *   --qa                 스크롤 화면은 아래쪽도 추가로 찍어 <out>/qa/ 에 저장
 *   --no-build           목업 모드에서 기존 .shots-dist 재사용
 *   --headed             브라우저 창을 띄워서 찍기
 *   --url <path> [--name n]  목록 대신 한 화면만 (예: --url "/app/w/dmc-shop?focus=all" --name overview)
 *   --finding <id>       대시보드 장면에서 열 끊김 (기본: 결제의 '퇴사자에게 걸린 호출' JT-PPL-001)
 *
 * 브라우저는 내려받지 않는다: 설치된 Google Chrome(channel 'chrome') → 없으면 Edge(channel 'msedge').
 */
import { mkdir, rm } from 'node:fs/promises';
import { createServer } from 'node:net';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

function parseArgs(argv) {
  const o = { live: false, base: null, ws: 'dmc-shop', out: path.join(ROOT, 'screenshots'), only: null, themes: ['dark', 'projector'], qa: false, build: true, headed: false, url: null, name: 'adhoc', finding: 'JT-PPL-001:9dfa62f887' };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    const next = () => {
      const v = argv[++i];
      if (v === undefined) throw new Error(`${a} 값이 없습니다`);
      return v;
    };
    if (a === '--live') o.live = true;
    else if (a === '--base') o.base = next().replace(/\/+$/, '');
    else if (a === '--ws') o.ws = next();
    else if (a === '--out') o.out = path.resolve(next());
    else if (a === '--only') o.only = next().split(',').filter(Boolean);
    else if (a === '--themes') o.themes = next().split(',').filter(Boolean);
    else if (a === '--qa') o.qa = true;
    else if (a === '--no-build') o.build = false;
    else if (a === '--headed') o.headed = true;
    else if (a === '--url') o.url = next();
    else if (a === '--name') o.name = next();
    else if (a === '--finding') o.finding = next();
    else if (a === '-h' || a === '--help') {
      console.log('사용법: node scripts/screenshots.mjs [--live --base URL] [--ws ID] [--out DIR] [--only a,b] [--themes dark,projector] [--qa] [--no-build] [--headed]');
      process.exit(0);
    } else throw new Error(`알 수 없는 옵션: ${a}`);
  }
  if (o.live && !o.base) o.base = 'http://127.0.0.1:8765';
  return o;
}

function freePort() {
  return new Promise((resolve, reject) => {
    const srv = createServer();
    srv.unref();
    srv.on('error', reject);
    srv.listen(0, '127.0.0.1', () => {
      const { port } = srv.address();
      srv.close(() => resolve(port));
    });
  });
}

/** 목업 빌드 + preview 서버. 반환값의 close()로 정리한다. */
async function startMockServer({ build: doBuild }) {
  process.env.VITE_USE_MOCKS = '1';
  const vite = await import('vite');
  const outDir = path.join(ROOT, '.shots-dist');
  if (doBuild) {
    console.log('· 목업 모드로 빌드 중 (.shots-dist)…');
    await vite.build({ root: ROOT, mode: 'mock', logLevel: 'warn', build: { outDir, emptyOutDir: true } });
  }
  const port = await freePort();
  const server = await vite.preview({
    root: ROOT,
    mode: 'mock',
    logLevel: 'warn',
    build: { outDir },
    preview: { host: '127.0.0.1', port, strictPort: true, open: false },
  });
  const base = `http://127.0.0.1:${port}`;
  console.log(`· preview: ${base}`);
  return {
    base,
    close: () => new Promise((resolve) => server.httpServer.close(() => resolve())),
  };
}

async function launchBrowser(headed) {
  const errors = [];
  for (const channel of ['chrome', 'msedge']) {
    try {
      const b = await chromium.launch({ channel, headless: !headed });
      console.log(`· 브라우저: ${channel} ${b.version()}`);
      return b;
    } catch (e) {
      errors.push(`${channel}: ${String(e).split('\n')[0]}`);
    }
  }
  throw new Error(`설치된 Chrome/Edge를 찾지 못했습니다.\n${errors.join('\n')}`);
}

/** 그래프가 배치·맞춤까지 끝날 때까지 */
async function waitGraph(page) {
  await page.waitForSelector('.react-flow__node', { timeout: 30_000 });
  await page.waitForFunction(
    () => {
      const nodes = document.querySelectorAll('.react-flow__node');
      if (!nodes.length) return false;
      return [...nodes].every((n) => n.getBoundingClientRect().width > 0 && getComputedStyle(n).visibility !== 'hidden');
    },
    null,
    { timeout: 30_000 },
  );
  // fitView 애니메이션 + 엣지 그리기 여유
  await page.waitForTimeout(1600);
}

async function settle(page, ms = 700) {
  await page.waitForLoadState('networkidle').catch(() => {});
  await page.evaluate(() => document.fonts.ready);
  await page.waitForTimeout(ms);
}

/** 장면 목록 — 같은 장면을 테마별로 한 장씩 */
function scenes(ws, live, finding) {
  const w = `/app/w/${encodeURIComponent(ws)}`;
  // 목업에서는 '승인 이후' 상태를 ?mock=after 로 만든다 (실제 API에서는 이미 승인된 상태를 그대로 찍는다)
  const after = live ? '' : 'mock=after';
  const q = (path, extra) => (extra ? `${path}${path.includes('?') ? '&' : '?'}${extra}` : path);
  return [
    { name: '01-landing', path: '/', scroll: 'window' },
    { name: '02-app-home', path: '/app', wait: 'text=별빛상점', scroll: 'main' },
    {
      name: '03-dashboard',
      // 기본 집중(결제) + '퇴사자에게 걸린 호출' 끊김 선택 → 근거 서랍이 열린 상태
      path: `${w}?finding=${encodeURIComponent(finding)}`,
      graph: true,
      wait: 'aside[aria-label="상세 정보"]',
      check: async (page) => ((await page.locator('aside[aria-label="상세 정보"]').innerText()).includes('찾을 수 없습니다') ? `끊김 ${finding}을(를) 찾지 못했습니다 — --finding 으로 다른 id를 주세요` : null),
    },
    // 서랍 없이 그래프만 — 결제의 빨간 끊김이 한눈에 (발표 자료 표지용)
    { name: '03b-dashboard-graph', path: w, graph: true },
    { name: '04-dashboard-overview', path: `${w}?focus=all`, graph: true },
    { name: '05-rehearsal-fail', path: `${w}/rehearsals?scenario=${encodeURIComponent('dmc-s1-chuseok-payment')}`, wait: 'text=FAIL', scroll: 'main', graphOptional: true },
    { name: '06-patches', path: `${w}/patches`, scroll: 'main' },
    { name: '07-versions', path: q(`${w}/versions`, after), wait: 'text=v2', scroll: 'main' },
    { name: '08-report', path: `${w}/report`, scroll: 'main' },
    { name: '09-trust', path: `${w}/trust`, scroll: 'main' },
    // 발표 자료용 전후 비교: 수정안 승인 뒤 대시보드 · 같은 리허설이 PASS로
    { name: '10-dashboard-after', path: q(w, after), graph: true, wait: 'text=v4' },
    { name: '11-rehearsal-pass-after', path: q(`${w}/rehearsals?scenario=${encodeURIComponent('dmc-s1-chuseok-payment')}`, after), wait: 'text=PASS', scroll: 'main', graphOptional: true },
  ];
}

async function capture(browser, base, scene, theme, opts) {
  const context = await browser.newContext({
    viewport: { width: 1920, height: 1080 },
    deviceScaleFactor: 1,
    locale: 'ko-KR',
    timezoneId: 'Asia/Seoul',
    colorScheme: theme === 'projector' ? 'light' : 'dark',
  });
  const page = await context.newPage();
  const problems = [];
  page.on('pageerror', (e) => problems.push(`pageerror: ${e.message}`));
  page.on('console', (m) => {
    if (m.type() === 'error') problems.push(`console: ${m.text()}`);
  });
  const sep = scene.path.includes('?') ? '&' : '?';
  const url = `${base}${scene.path}${sep}theme=${theme}&role=lead`;
  await page.goto(url, { waitUntil: 'domcontentloaded' });
  if (scene.wait) await page.waitForSelector(scene.wait, { timeout: 30_000 }).catch(() => problems.push(`대기 실패: ${scene.wait}`));
  if (scene.graph) await waitGraph(page);
  else if (scene.graphOptional && (await page.$('.react-flow'))) await waitGraph(page).catch(() => {});
  await settle(page);
  if (scene.check) {
    const msg = await scene.check(page).catch(() => null);
    if (msg) problems.push(msg);
  }
  const file = path.join(opts.out, `${scene.name}-${theme}.png`);
  await page.screenshot({ path: file });
  const files = [file];

  if (opts.qa && scene.scroll) {
    const qaDir = path.join(opts.out, 'qa');
    await mkdir(qaDir, { recursive: true });
    for (let i = 1; i <= 6; i++) {
      const moved = await page.evaluate(
        ([mode]) => {
          const el = mode === 'window' ? document.scrollingElement : document.querySelector('main');
          if (!el) return false;
          const before = el.scrollTop;
          el.scrollTop = before + el.clientHeight * 0.9;
          return el.scrollTop > before;
        },
        [scene.scroll],
      );
      if (!moved) break;
      await page.waitForTimeout(500);
      const f = path.join(qaDir, `${scene.name}-${theme}-scroll${i}.png`);
      await page.screenshot({ path: f });
      files.push(f);
    }
  }
  await context.close();
  return { files, problems };
}

async function main() {
  const opts = parseArgs(process.argv.slice(2));
  await mkdir(opts.out, { recursive: true });
  if (opts.qa) await rm(path.join(opts.out, 'qa'), { recursive: true, force: true });

  const server = opts.live ? { base: opts.base, close: async () => {} } : await startMockServer(opts);
  let browser;
  let failed = 0;
  try {
    browser = await launchBrowser(opts.headed);
    const list = opts.url
      ? [{ name: opts.name, path: opts.url, graphOptional: true, scroll: 'main' }]
      : scenes(opts.ws, opts.live, opts.finding).filter((s) => !opts.only || opts.only.some((o) => s.name.includes(o)));
    for (const scene of list) {
      for (const theme of opts.themes) {
        const t0 = Date.now();
        try {
          const { files, problems } = await capture(browser, server.base, scene, theme, opts);
          console.log(`✓ ${path.basename(files[0])} (${Date.now() - t0}ms)${files.length > 1 ? ` +${files.length - 1} qa` : ''}`);
          for (const p of problems) console.log(`    ! ${p}`);
        } catch (e) {
          failed++;
          console.error(`✗ ${scene.name}-${theme}: ${e instanceof Error ? e.message : e}`);
        }
      }
    }
  } finally {
    await browser?.close();
    await server.close();
  }
  if (failed) {
    console.error(`${failed}장 실패`);
    process.exit(1);
  }
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
