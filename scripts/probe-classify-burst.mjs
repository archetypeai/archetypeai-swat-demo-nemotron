#!/usr/bin/env node
// Reproduce the live-classify burst against Omega and show the RAW Archetype AI
// responses (including timeout/capacity errors).
//
// One playback window is classified exactly the way the running app does it:
// all 6 stages embedded concurrently, and within each stage every channel sent
// as its own /query — so a single window fans out to ~40 simultaneous calls.
// A stage "drops" (returns no result) if ANY one of its channel calls fails,
// because embedWindow uses Promise.all (all-or-nothing).
//
// Usage:
//   node scripts/probe-classify-burst.mjs [--row=20000] [--timeout=15000]
// --timeout defaults to 15000 to match the app's OMEGA_TIMEOUT_MS.

import fs from 'fs';

const MODEL = 'OmegaEncoder::omega_embeddings_1_4';
const STAGE_COLUMNS = {
	P1: ['FIT101', 'LIT101', 'MV101', 'P101'],
	P2: ['AIT201', 'AIT202', 'AIT203', 'FIT201', 'MV201', 'P203', 'P205'],
	P3: ['DPIT301', 'FIT301', 'LIT301', 'MV301', 'MV302', 'MV303', 'MV304', 'P301', 'P302'],
	P4: ['AIT401', 'AIT402', 'FIT401', 'LIT401', 'P402', 'UV401'],
	P5: ['AIT501', 'AIT502', 'AIT503', 'AIT504', 'FIT501', 'FIT502', 'FIT503', 'FIT504', 'P501', 'PIT501', 'PIT502', 'PIT503'],
	P6: ['FIT601', 'P602']
};
const WINDOW = 128;

function loadEnv() {
	const env = {};
	for (const line of fs.readFileSync('.env', 'utf-8').split('\n')) {
		const m = line.match(/^([A-Z_][A-Z0-9_]*)=(.*)$/);
		if (m) env[m[1]] = m[2].trim();
	}
	return env;
}
function arg(name, def) {
	const a = process.argv.find((x) => x.startsWith(`--${name}=`));
	return a ? parseInt(a.split('=')[1]) : def;
}

const env = loadEnv();
const ENDPOINT = env.ATAI_API_ENDPOINT.replace(/\/$/, '') + '/v0.5/query';
const KEY = env.ATAI_API_KEY;
const row = arg('row', 20000);
const timeoutMs = arg('timeout', 15000);
const concurrency = arg('concurrency', 999); // default ≈ unbounded (old live behavior)
const retries = arg('retries', 1); // default 1 = no retry

const scaler = JSON.parse(fs.readFileSync('data/scaler.json', 'utf-8'));
const lines = fs.readFileSync('data/swat_playback.csv', 'utf-8').split(/\r?\n/).filter((l) => l);
const header = lines[0].split(',');
const idx = Object.fromEntries(header.map((h, i) => [h, i]));

// Build the scaled channel-first window for one stage.
function scaledChannels(stage) {
	const cols = STAGE_COLUMNS[stage];
	return cols.map((col) => {
		const m = scaler.mean[col] ?? 0,
			s = scaler.std[col] ?? 1;
		const out = new Array(WINDOW);
		for (let r = 0; r < WINDOW; r++) {
			const v = parseFloat(lines[1 + row + r].split(',')[idx[col]]);
			out[r] = ((isNaN(v) ? 0 : v) - m) / s;
		}
		return out;
	});
}

// One single-channel /query, capturing the RAW response/error verbatim.
async function call(stage, channelIdx, channel) {
	const ctrl = new AbortController();
	const t = setTimeout(() => ctrl.abort(), timeoutMs);
	const t0 = Date.now();
	try {
		const res = await fetch(ENDPOINT, {
			method: 'POST',
			headers: { Authorization: `Bearer ${KEY}`, 'Content-Type': 'application/json' },
			body: JSON.stringify({
				query: '',
				model: MODEL,
				normalize_input: false,
				events: [{ type: 'data.numeric_array', event_data: { contents: [channel] } }]
			}),
			signal: ctrl.signal
		});
		const ms = Date.now() - t0;
		if (!res.ok) {
			const body = await res.text();
			return { stage, channelIdx, ok: false, ms, status: res.status, body: body.slice(0, 300) };
		}
		await res.json();
		return { stage, channelIdx, ok: true, ms, status: 200 };
	} catch (e) {
		const ms = Date.now() - t0;
		return { stage, channelIdx, ok: false, ms, status: 'abort/err', body: String(e).slice(0, 200) };
	} finally {
		clearTimeout(t);
	}
}

// One channel call with retry (mirrors embedChannel in src/lib/server/newton.js).
async function callWithRetry(spec) {
	let last;
	for (let a = 0; a < retries; a++) {
		last = await call(spec.stage, spec.i, spec.ch);
		if (last.ok) return { ...last, attempts: a + 1 };
		if (a < retries - 1) await new Promise((r) => setTimeout(r, 400 * (a + 1)));
	}
	return { ...last, attempts: retries };
}

console.log(`Endpoint: ${ENDPOINT}`);
console.log(`Window row ${row}, timeout ${timeoutMs}ms, concurrency ${concurrency}, retries ${retries}\n`);

// Per-channel jobs across all 6 stages, run through a bounded worker pool so
// `concurrency` controls how many are in flight at once (concurrency=999 ≈ the
// old unbounded live behavior; concurrency=6 ≈ the bounded fan-out fix).
const specs = [];
for (const stage of Object.keys(STAGE_COLUMNS)) {
	scaledChannels(stage).forEach((ch, i) => specs.push({ stage, i, ch }));
}
console.log(`Dispatching ${specs.length} per-channel calls...\n`);
const results = new Array(specs.length);
let nextJob = 0;
async function worker() {
	while (nextJob < specs.length) {
		const idx = nextJob++;
		results[idx] = await callWithRetry(specs[idx]);
	}
}
await Promise.all(Array.from({ length: Math.min(concurrency, specs.length) }, worker));

// Per-call failures (the raw Archetype messages)
const fails = results.filter((r) => !r.ok);
const oks = results.filter((r) => r.ok);
console.log(`Calls: ${oks.length}/${results.length} ok, ${fails.length} failed.`);
const okMs = oks.map((r) => r.ms).sort((a, b) => a - b);
if (okMs.length) console.log(`Latency of successful calls (ms): min ${okMs[0]}, median ${okMs[okMs.length >> 1]}, max ${okMs[okMs.length - 1]}`);
if (fails.length) {
	console.log('\nFAILURES (raw response from Archetype AI):');
	for (const f of fails) console.log(`  ${f.stage} ch${f.channelIdx}  [${f.status}] ${f.ms}ms  ${f.body || ''}`);
}

// Per-stage drop = any channel failed (mirrors embedWindow's Promise.all)
console.log('\nPer-stage result (a stage DROPS if any one channel fails):');
for (const stage of Object.keys(STAGE_COLUMNS)) {
	const stageRes = results.filter((r) => r.stage === stage);
	const failed = stageRes.filter((r) => !r.ok).length;
	console.log(`  ${stage} (${STAGE_COLUMNS[stage].length}ch): ${failed === 0 ? '✓ returns' : `✗ DROPS (${failed} channel call(s) failed)`}`);
}
