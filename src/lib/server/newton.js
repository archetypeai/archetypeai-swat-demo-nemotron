import { ATAI_API_KEY, ATAI_API_ENDPOINT } from '$env/static/private';
import { readFileSync, existsSync } from 'fs';
import { resolve } from 'path';
import { projectEmbedding } from './projections.js';

const API_VERSION = 'v0.5';
const OMEGA_MODEL = 'OmegaEncoder::omega_embeddings_1_4';

// SWaT stage → sensor column mapping. Identical to the original Lens-based demo
// so the same n-shot files and KNN library work without conversion.
export const STAGE_COLUMNS = {
	P1: ['FIT101', 'LIT101', 'MV101', 'P101'],
	P2: ['AIT201', 'AIT202', 'AIT203', 'FIT201', 'MV201', 'P203', 'P205'],
	P3: ['DPIT301', 'FIT301', 'LIT301', 'MV301', 'MV302', 'MV303', 'MV304', 'P301', 'P302'],
	P4: ['AIT401', 'AIT402', 'FIT401', 'LIT401', 'P402', 'UV401'],
	P5: [
		'AIT501', 'AIT502', 'AIT503', 'AIT504',
		'FIT501', 'FIT502', 'FIT503', 'FIT504',
		'P501', 'PIT501', 'PIT502', 'PIT503'
	],
	P6: ['FIT601', 'P602']
};
export const STAGE_IDS = Object.keys(STAGE_COLUMNS);
export const MONITORED_STAGE_IDS = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6'];

export const DEFAULT_CONFIG = {
	windowSize: 128,
	stepSize: 128,
	nNeighbors: 3
};

function apiUrl(path) {
	return `${ATAI_API_ENDPOINT.replace(/\/$/, '')}/${API_VERSION}${path}`;
}

// ──────────────────────────────────────────────────────────────────────
// Global per-channel StandardScaler (data/scaler.json). Loaded lazily.
// Pre-normalizing every window with these fixed stats — and passing
// normalize_input=false to Omega — preserves cross-window amplitude signal
// that per-window normalization would erase. See scripts/build-scaler.js.
// ──────────────────────────────────────────────────────────────────────

let SCALER = null;
let SCALER_ERROR = null;
function ensureScaler() {
	if (SCALER || SCALER_ERROR) return;
	const path = resolve('data/scaler.json');
	if (!existsSync(path)) {
		SCALER_ERROR = new Error(
			'Missing data/scaler.json — run `node scripts/build-scaler.js` first.'
		);
		return;
	}
	SCALER = JSON.parse(readFileSync(path, 'utf-8'));
}

function applyScaler(channelFirstWindow, columns) {
	ensureScaler();
	if (SCALER_ERROR) throw SCALER_ERROR;
	const out = new Array(columns.length);
	for (let c = 0; c < columns.length; c++) {
		const col = columns[c];
		const m = SCALER.mean[col] ?? 0;
		const s = SCALER.std[col] ?? 1;
		const src = channelFirstWindow[c];
		const dst = new Array(src.length);
		for (let i = 0; i < src.length; i++) dst[i] = (src[i] - m) / s;
		out[c] = dst;
	}
	return out;
}

// ──────────────────────────────────────────────────────────────────────
// KNN library (loaded once at boot from data/knn-library.json)
// ──────────────────────────────────────────────────────────────────────

let LIBRARY = null;
let LIBRARY_ERROR = null;
function ensureLibrary() {
	if (LIBRARY || LIBRARY_ERROR) return;
	const path = resolve('data/knn-library.json');
	if (!existsSync(path)) {
		LIBRARY_ERROR = new Error(
			'Missing data/knn-library.json — run `node scripts/build-knn-library.js` first.'
		);
		return;
	}
	const raw = JSON.parse(readFileSync(path, 'utf-8'));
	for (const stageId of Object.keys(raw.stages)) {
		raw.stages[stageId].embeddings = raw.stages[stageId].embeddings.map((e) => ({
			label: e.label,
			vec: Float32Array.from(e.vec)
		}));
	}
	LIBRARY = raw;
}

export function getLibraryConfig() {
	ensureLibrary();
	if (LIBRARY_ERROR) throw LIBRARY_ERROR;
	return LIBRARY.config;
}

// ──────────────────────────────────────────────────────────────────────
// Direct Query: Omega embedding
// ──────────────────────────────────────────────────────────────────────

// Omega 1.4 inference is ~0.45s, but the GPQ dispatcher adds a quantized
// 3-8s queue wait even on an idle fleet. A 15s timeout sat under the p95 and
// caused self-inflicted aborts -> retries -> more queue load. Budget for the
// dispatcher, not the model.
const OMEGA_TIMEOUT_MS = 60000;

async function postQuery(body, timeoutMs = OMEGA_TIMEOUT_MS) {
	const controller = new AbortController();
	const t = setTimeout(() => controller.abort(), timeoutMs);
	try {
		const res = await fetch(apiUrl('/query'), {
			method: 'POST',
			headers: {
				Authorization: `Bearer ${ATAI_API_KEY}`,
				'Content-Type': 'application/json'
			},
			body: JSON.stringify(body),
			signal: controller.signal
		});
		if (!res.ok) {
			const err = await res.text();
			throw new Error(`/query failed: ${res.status} ${err.slice(0, 300)}`);
		}
		// Await the body inside the try: a bare `return res.json()` runs `finally` (clearing the
		// timeout) as soon as headers arrive, so a stalled body hangs forever and holds an
		// Omega pool slot — enough of those and every classify request queues indefinitely.
		return await res.json();
	} finally {
		clearTimeout(t);
	}
}

// ──────────────────────────────────────────────────────────────────────
// Bounded per-channel fan-out (matches the Omega skill's thread-pool / `embed()`
// pattern). We keep one /query per channel, but cap how many run at once so a
// window's fan-out (e.g. 40 channels across 6 stages) doesn't overrun a
// capacity-limited GPQ node. Each per-channel call also retries transient
// failures (504 / timeout) instead of dropping the whole stage.
// ──────────────────────────────────────────────────────────────────────

// A window fans out to 40 channels (P1..P6 = 4+7+9+6+12+2). The Omega 1.4 fleet
// runs 7-8 GPQ nodes and the per-call wait is dispatcher overhead, not GPU
// contention — measured p50 barely moves (10.0s -> 12.4s) going from 6 in flight
// to 40, while window wall-clock drops 77.5s -> 21.6s. So size the pool to the
// whole window and let it go out in a single wave.
const OMEGA_MAX_CONCURRENCY = 40;
const OMEGA_RETRIES = 2;

let omegaInFlight = 0;
const omegaQueue = [];
function withOmegaSlot(fn) {
	return new Promise((resolve, reject) => {
		const run = () => {
			omegaInFlight++;
			Promise.resolve()
				.then(fn)
				.then(resolve, reject)
				.finally(() => {
					omegaInFlight--;
					const next = omegaQueue.shift();
					if (next) next();
				});
		};
		if (omegaInFlight < OMEGA_MAX_CONCURRENCY) run();
		else omegaQueue.push(run);
	});
}

// Embed a single channel (flat 768-d vector), retrying transient failures.
async function embedChannel(channel) {
	let lastErr;
	for (let attempt = 0; attempt < OMEGA_RETRIES; attempt++) {
		try {
			const data = await postQuery({
				query: '',
				model: OMEGA_MODEL,
				// Pre-normalized at the call site via applyScaler(); Omega should NOT
				// re-normalize per-window or it would erase cross-window amplitude.
				normalize_input: false,
				events: [{ type: 'data.numeric_array', event_data: { contents: [channel] } }]
			});
			const vec = data.response?.response;
			if (!Array.isArray(vec) || typeof vec[0] !== 'number') {
				throw new Error(`unexpected Omega response shape: ${JSON.stringify(data).slice(0, 200)}`);
			}
			return vec;
		} catch (err) {
			lastErr = err;
			if (attempt < OMEGA_RETRIES - 1) {
				await new Promise((r) => setTimeout(r, 400 * (attempt + 1)));
			}
		}
	}
	throw lastErr;
}

// Send a [num_columns x window_size] channel-first array to Omega and return
// the [num_columns x 768] embedding matrix, flattened to a single Float32Array
// for KNN distance comparisons against the library.
export async function embedWindow(channelFirstWindow) {
	// Per the Omega skill: one /query per channel. The calls fan out through a
	// shared bounded pool (OMEGA_MAX_CONCURRENCY) so all stages in a window share
	// the same in-flight cap; each is retried on transient failure. Concatenate
	// the per-channel 768-d vectors in channel order into the joint KNN feature.
	const perChannel = await Promise.all(
		channelFirstWindow.map((channel) => withOmegaSlot(() => embedChannel(channel)))
	);
	const numChannels = perChannel.length;
	const dim = perChannel[0].length;
	const out = new Float32Array(numChannels * dim);
	for (let c = 0; c < numChannels; c++) {
		for (let d = 0; d < dim; d++) {
			out[c * dim + d] = perChannel[c][d];
		}
	}
	return out;
}

// ──────────────────────────────────────────────────────────────────────
// Local KNN classifier
// ──────────────────────────────────────────────────────────────────────

function euclideanSq(a, b) {
	let s = 0;
	for (let i = 0; i < a.length; i++) {
		const d = a[i] - b[i];
		s += d * d;
	}
	return s;
}

function classifyEmbedding(stageId, embedding, k = DEFAULT_CONFIG.nNeighbors) {
	ensureLibrary();
	if (LIBRARY_ERROR) throw LIBRARY_ERROR;
	const lib = LIBRARY.stages[stageId];
	if (!lib) throw new Error(`no library for stage ${stageId}`);
	const dists = new Array(lib.embeddings.length);
	for (let i = 0; i < lib.embeddings.length; i++) {
		dists[i] = { d: euclideanSq(lib.embeddings[i].vec, embedding), label: lib.embeddings[i].label };
	}
	dists.sort((a, b) => a.d - b.d);
	const top = dists.slice(0, k);
	const votes = {};
	for (const t of top) votes[t.label] = (votes[t.label] || 0) + 1;
	let winner = null;
	let max = -1;
	for (const [label, n] of Object.entries(votes)) {
		if (n > max) {
			max = n;
			winner = label;
		}
	}
	return { label: winner, neighbors: top.map((t) => ({ label: t.label, dist: Math.sqrt(t.d) })) };
}

function extractStageWindow(stageId, rows) {
	const cols = STAGE_COLUMNS[stageId];
	return cols.map((col) =>
		rows.map((row) => {
			const v = parseFloat(row[col]);
			return isNaN(v) ? 0 : v;
		})
	);
}

// Run Direct Query → KNN for one stage. Returns the label, neighbors,
// and the raw embedding (so the client can run PCA-2 projection for the
// embedding-viz panel without re-querying Omega).
export async function classifyStage(stageId, rows, { k = DEFAULT_CONFIG.nNeighbors } = {}) {
	const win = extractStageWindow(stageId, rows);
	const scaled = applyScaler(win, STAGE_COLUMNS[stageId]);
	const embedding = await embedWindow(scaled);
	const { label, neighbors } = classifyEmbedding(stageId, embedding, k);
	// Project to 2D for the embedding-viz panel. Cheap (~10-50ms total for
	// PCA + UMAP transform). If projection ever fails, the classification
	// itself still returns — viz coords are best-effort.
	let coords = null;
	try {
		coords = await projectEmbedding(stageId, embedding);
	} catch {
		coords = null;
	}
	return { stageId, label, neighbors, coords };
}

export async function classifyAllStages(rows, opts = {}) {
	const results = await Promise.allSettled(
		MONITORED_STAGE_IDS.map((stageId) => classifyStage(stageId, rows, opts))
	);
	const out = {};
	const errors = [];
	for (let i = 0; i < results.length; i++) {
		const stageId = MONITORED_STAGE_IDS[i];
		const r = results[i];
		if (r.status === 'fulfilled') {
			out[stageId] = r.value;
		} else {
			errors.push({ stageId, error: r.reason?.message || String(r.reason) });
		}
	}
	return { stages: out, errors };
}

// ──────────────────────────────────────────────────────────────────────
// Text reasoning Direct Query (used for operator suggestions, unchanged)
// ──────────────────────────────────────────────────────────────────────

export function getApiKey() {
	return ATAI_API_KEY;
}

export async function queryNewton({ query, systemPrompt = '', maxNewTokens = 1024 }) {
	const controller = new AbortController();
	const timeoutId = setTimeout(() => controller.abort(), 120000);
	try {
		const res = await fetch(apiUrl('/query'), {
			method: 'POST',
			headers: {
				Authorization: `Bearer ${ATAI_API_KEY}`,
				'Content-Type': 'application/json'
			},
			body: JSON.stringify({
				query,
				// C 2.6 honors `instruction_prompt`; the legacy `system_prompt`
				// field is inert on this checkpoint, so we send only the former.
				instruction_prompt: systemPrompt,
				file_ids: [],
				model: 'Newton::c2_6_8b_fp8_260424d7a55d5e',
				max_new_tokens: maxNewTokens
			}),
			signal: controller.signal
		});
		if (!res.ok) {
			const err = await res.json().catch(() => ({}));
			throw new Error(`query failed: ${res.status} - ${JSON.stringify(err)}`);
		}
		const data = await res.json();
		if (data.response?.response && Array.isArray(data.response.response)) {
			return data.response.response[0] || '';
		}
		if (Array.isArray(data.response)) return data.response[0] || '';
		if (typeof data.response === 'string') return data.response;
		if (data.text) return data.text;
		return '';
	} finally {
		clearTimeout(timeoutId);
	}
}
