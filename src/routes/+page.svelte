<script>
	import { onMount } from 'svelte';
	import { cn } from '$lib/utils.js';
	import Menubar from '$lib/components/ui/patterns/menubar/index.js';
	import Button from '$lib/components/ui/primitives/button/index.js';
	import Badge from '$lib/components/ui/primitives/badge/index.js';
	import SpinnerIcon from '@lucide/svelte/icons/loader';
	import ChevronDownIcon from '@lucide/svelte/icons/chevron-down';
	import ChevronUpIcon from '@lucide/svelte/icons/chevron-up';
	import StageCard from '$lib/components/ui/custom/stage-card.svelte';
	import SuggestedActions from '$lib/components/ui/custom/suggested-actions.svelte';
	import PlaybackControls from '$lib/components/ui/custom/playback-controls.svelte';
	import PlantSchematic from '$lib/components/ui/custom/plant-schematic.svelte';
	import EmbeddingPanel from '$lib/components/ui/custom/embedding-panel.svelte';
	import { fetchChunk, classifyWindow, fetchProjections, fetchSuggestions } from '$lib/api/swat.js';

	// Mirrors src/lib/server/newton.js STAGE_COLUMNS; kept in sync manually.
	const STAGE_COLUMNS = {
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

	const STAGE_META = {
		P1: 'Raw water intake',
		P2: 'Chemical dosing',
		P3: 'Ultrafiltration',
		P4: 'UV dechlorination',
		P5: 'Reverse osmosis',
		P6: 'Backwash'
	};

	const STAGE_IDS = Object.keys(STAGE_COLUMNS);
	const MONITORED_STAGE_IDS = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6'];

	const WINDOW_SIZE = 128;
	const STEP_SIZE = 128;
	const CHUNK_SIZE = 10000;
	const REPLAY_SPEED = 10; // tick every 100ms, advance 1 row → 10× real time on 1Hz data
	// Playback streams the dedicated held-out file (data/swat_playback.csv), which
	// shares no timestamps with the n-shot KNN library. We start partway in
	// (~row 19,000) so the demo quickly reaches the attack windows the model
	// actually detects — see README "What the held-out numbers show". Set to 0 to
	// replay the full normal lead-in → transition → early attack from the top.
	const INITIAL_OFFSET = 17000;
	// Live cursor trail length per stage (oldest entries drop off).
	const TRAIL_LENGTH = 8;

	let rows = $state([]);
	let total = $state(0);
	let startOffset = $state(INITIAL_OFFSET);
	let loadedEnd = $state(0);
	let playheadIdx = $state(0);
	let playing = $state(false);
	let playInterval = null;
	let streamCounter = $state(0);
	let loadingChunk = $state(false);

	// Direct Query is stateless — no sessions, no setup. "ready" means we've loaded
	// the inference CSV chunk and projections; "classifying" appears transiently
	// during in-flight /api/classify calls.
	let sessionStatus = $state('idle'); // idle | ready | error
	let setupError = $state('');

	let stageStatuses = $state(Object.fromEntries(STAGE_IDS.map((s) => [s, 'idle'])));
	let stageLabels = $state(Object.fromEntries(STAGE_IDS.map((s) => [s, []])));
	let hasStartedPlayback = $state(false);
	let classifyInFlight = $state(false);
	// Stage IDs that failed to classify on the most recent window. Surfaced in the
	// menubar so a partial Omega failure reads as "degraded" rather than silently
	// leaving those stages parked on their pre-classification state.
	let classifyErrors = $state([]);

	// Per-stage library coords (loaded once from /api/projections) and live-cursor
	// trail (filled as classify responses arrive). Both PCA and UMAP carried in
	// parallel so the embedding panel can toggle modes without re-fetching.
	let libraryCoords = $state(null); // { P1: { columns, library: { pca:[], umap:[] } }, ... }
	let liveTrail = $state(Object.fromEntries(STAGE_IDS.map((s) => [s, []])));
	let embeddingPanelOpen = $state(false);

	let liveRow = $derived(rows[playheadIdx] ?? null);

	// Gate P6 classification on activity. P6 is the backwash loop — when FIT601 ≈ 0
	// the stage is idle/standby and classification is essentially noise.
	const P6_ACTIVITY_THRESHOLD = 0.01;
	let aiSuggestions = $state(null);
	let suggestionSource = $state('loading');
	let suggestionSignature = $state('');
	let suggestionDebounce = null;
	let suggestionFetchInFlight = false;
	// Reasoning model for Suggested Actions: NVIDIA Nemotron (default) or Newton C 2.6.
	let reasoningModel = $state('nemotron');

	let effectiveStatuses = $derived.by(() => {
		const out = { ...stageStatuses };
		if (sessionStatus === 'ready' && hasStartedPlayback && liveRow) {
			const flow = parseFloat(liveRow.FIT601 ?? '0');
			if (!isNaN(flow) && flow < P6_ACTIVITY_THRESHOLD) {
				out.P6 = 'standby';
			}
		}
		return out;
	});

	async function loadInitialChunk() {
		loadingChunk = true;
		try {
			const data = await fetchChunk(startOffset, CHUNK_SIZE);
			rows = data.rows;
			total = data.total;
			loadedEnd = startOffset + data.rows.length;
		} catch (err) {
			console.error('Failed to load initial chunk:', err);
		} finally {
			loadingChunk = false;
		}
	}

	async function loadNextChunk() {
		if (loadingChunk || loadedEnd >= total) return;
		loadingChunk = true;
		try {
			const data = await fetchChunk(loadedEnd, CHUNK_SIZE);
			rows = [...rows, ...data.rows];
			loadedEnd += data.rows.length;
		} catch (err) {
			console.error('Failed to load chunk:', err);
		} finally {
			loadingChunk = false;
		}
	}

	onMount(() => {
		// Static library projection coords for the embedding panel. Background
		// scatter. Live cursor is appended per classify response.
		fetchProjections()
			.then((data) => {
				libraryCoords = data.stages ?? null;
			})
			.catch((err) => console.warn('[projections] failed:', err));
	});

	async function handleStart() {
		if (sessionStatus === 'ready') return;
		sessionStatus = 'ready';
		setupError = '';
		for (const s of MONITORED_STAGE_IDS) stageStatuses[s] = 'ready';
	}

	function handleStop() {
		handlePause();
		sessionStatus = 'idle';
		stageStatuses = Object.fromEntries(STAGE_IDS.map((s) => [s, 'idle']));
		stageLabels = Object.fromEntries(STAGE_IDS.map((s) => [s, []]));
		liveTrail = Object.fromEntries(STAGE_IDS.map((s) => [s, []]));
		hasStartedPlayback = false;
		classifyErrors = [];
	}

	async function classifyCurrentWindow() {
		if (sessionStatus !== 'ready') return;
		const windowEnd = (streamCounter + 1) * STEP_SIZE;
		const windowStart = windowEnd - WINDOW_SIZE;
		if (windowStart < 0 || windowEnd > rows.length) return;
		const counter = streamCounter;
		streamCounter++;
		const windowRows = rows.slice(windowStart, windowEnd);
		classifyInFlight = true;
		try {
			const result = await classifyWindow(windowRows);
			if (!hasStartedPlayback) return;
			for (const stageId of MONITORED_STAGE_IDS) {
				const stage = result.stages?.[stageId];
				if (!stage) continue;
				const upper = String(stage.label || '').toUpperCase();
				if (upper !== 'ATTACK' && upper !== 'NORMAL') continue;
				stageLabels[stageId] = [...stageLabels[stageId], upper].slice(-20);
				stageStatuses[stageId] = upper === 'ATTACK' ? 'attack' : 'normal';
				if (stage.coords) {
					const next = [...liveTrail[stageId], { ...stage.coords, label: upper, counter }];
					liveTrail[stageId] = next.slice(-TRAIL_LENGTH);
				}
			}
			classifyErrors = result.errors?.map((e) => e.stageId) ?? [];
			if (result.errors?.length) {
				console.warn('[classify] partial errors:', result.errors);
			}
		} catch (err) {
			classifyErrors = [...MONITORED_STAGE_IDS];
			console.error('[classify] failed:', err);
		} finally {
			classifyInFlight = false;
		}
	}

	function handlePlay() {
		if (!rows.length) return;
		playing = true;
		hasStartedPlayback = true;

		if (sessionStatus === 'ready' && rows.length >= WINDOW_SIZE) {
			classifyCurrentWindow();
		}

		playInterval = setInterval(() => {
			if (playheadIdx < rows.length - 1) playheadIdx += 1;

			if (playheadIdx > rows.length - 1000 && loadedEnd < total) loadNextChunk();

			if (sessionStatus === 'ready' && playheadIdx >= (streamCounter + 1) * STEP_SIZE) {
				classifyCurrentWindow();
			}

			if (playheadIdx >= rows.length - 1 && loadedEnd >= total) {
				playing = false;
				clearInterval(playInterval);
			}
		}, 1000 / REPLAY_SPEED);
	}

	function handlePause() {
		playing = false;
		if (playInterval) clearInterval(playInterval);
	}

	function handleReset() {
		handlePause();
		playheadIdx = 0;
		streamCounter = 0;
		stageLabels = Object.fromEntries(STAGE_IDS.map((s) => [s, []]));
		liveTrail = Object.fromEntries(STAGE_IDS.map((s) => [s, []]));
		hasStartedPlayback = false;
		classifyErrors = [];
		if (sessionStatus === 'ready') {
			stageStatuses = Object.fromEntries(STAGE_IDS.map((s) => [s, 'ready']));
		}
	}

	$effect(() => {
		loadInitialChunk();
	});

	const ANOMALY_DEBOUNCE_MS = 2000;
	let anomalySignature = $derived.by(() => {
		return ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']
			.filter((id) => effectiveStatuses[id] === 'attack')
			.sort()
			.join(',');
	});

	// What the current cards were generated for: model + anomaly set.
	let suggestionKey = $derived(anomalySignature ? `${reasoningModel}:${anomalySignature}` : '');

	function handleModelChange(model) {
		if (!model || model === reasoningModel) return;
		reasoningModel = model;
		aiSuggestions = null; // don't show the other model's cards under this model's label
		if (suggestionDebounce) {
			clearTimeout(suggestionDebounce);
			suggestionDebounce = null;
		}
		if (anomalySignature) runSuggestionsFetch();
	}

	async function runSuggestionsFetch() {
		if (suggestionFetchInFlight) return;
		const key = suggestionKey;
		const model = reasoningModel;
		if (!anomalySignature) return;
		suggestionFetchInFlight = true;
		suggestionSource = 'loading';

		const stageSensors = {};
		if (liveRow) {
			for (const stageId of STAGE_IDS) {
				if (effectiveStatuses[stageId] !== 'attack') continue;
				const sensors = {};
				for (const col of STAGE_COLUMNS[stageId]) sensors[col] = liveRow[col];
				stageSensors[stageId] = sensors;
			}
		}

		const timeoutPromise = new Promise((_, reject) =>
			setTimeout(() => reject(new Error('Client timeout: 150s exceeded')), 150000)
		);

		try {
			const result = await Promise.race([
				fetchSuggestions(effectiveStatuses, stageSensors, model),
				timeoutPromise
			]);
			aiSuggestions = result.suggestions ?? [];
			suggestionSource = result.source ?? 'error';
			suggestionSignature = key;
		} catch (err) {
			console.error('[suggestions] failed:', err);
			aiSuggestions = [];
			suggestionSource = 'error';
			suggestionSignature = key;
		} finally {
			suggestionFetchInFlight = false;
			if (suggestionKey && suggestionKey !== suggestionSignature) {
				runSuggestionsFetch();
			}
		}
	}

	$effect(() => {
		const sig = anomalySignature;
		const key = suggestionKey;
		if (!sig) {
			if (suggestionDebounce) {
				clearTimeout(suggestionDebounce);
				suggestionDebounce = null;
			}
			aiSuggestions = [];
			suggestionSource = reasoningModel;
			suggestionSignature = '';
			return;
		}
		if (key === suggestionSignature && aiSuggestions) return;
		suggestionSource = 'loading';
		if (suggestionDebounce || suggestionFetchInFlight) return;
		suggestionDebounce = setTimeout(() => {
			suggestionDebounce = null;
			runSuggestionsFetch();
		}, ANOMALY_DEBOUNCE_MS);
	});
</script>

<svelte:head><title>Newton Omega × NVIDIA Nemotron · SWaT</title></svelte:head>

<a
	href="#main-content"
	class="sr-only focus:not-sr-only focus:fixed focus:top-4 focus:left-4 focus:z-50 focus:rounded-md focus:bg-background focus:px-4 focus:py-2 focus:text-foreground focus:ring-2 focus:ring-ring"
>
	Skip to content
</a>

{#snippet partnerSnippet()}
	<span class="text-muted-foreground font-mono text-sm tracking-wider uppercase">
		SWaT · Omega × NVIDIA Nemotron · 6-stage water treatment
	</span>
{/snippet}

<div
	class="bg-background text-foreground flex min-h-screen flex-col"
>
	<Menubar partnerLogo={partnerSnippet}>
		{#if sessionStatus === 'ready'}
			<Badge
				variant="outline"
				class={cn('font-mono', classifyErrors.length ? 'text-atai-warning' : 'text-atai-good')}
			>
				Omega · Direct Query{classifyInFlight ? ' · classifying' : ' · ready'}
			</Badge>
			{#if classifyErrors.length}
				<Badge variant="outline" class="text-atai-warning font-mono">
					{classifyErrors.length} stage{classifyErrors.length > 1 ? 's' : ''} unavailable · {classifyErrors.join(
						' '
					)}
				</Badge>
			{/if}
			<Button variant="outline" size="sm" onclick={handleStop}>Stop</Button>
		{:else if sessionStatus === 'error'}
			<Badge variant="outline" class="text-atai-critical font-mono">Error</Badge>
			<Button variant="outline" size="sm" onclick={handleStart}>Retry</Button>
		{:else}
			<Button variant="default" size="sm" onclick={handleStart} disabled={!rows.length}>
				Start analysis
			</Button>
		{/if}
	</Menubar>

	<div class="border-border flex items-center gap-4 border-b px-4 py-2">
		<PlaybackControls
			{playing}
			current={startOffset + playheadIdx}
			{total}
			speed={REPLAY_SPEED}
			disabled={!rows.length || sessionStatus !== 'ready'}
			onplay={handlePlay}
			onpause={handlePause}
			onreset={handleReset}
		/>
	</div>

	<main id="main-content" class="grid min-h-0 grid-cols-[3fr_1fr] gap-4 overflow-hidden p-4" style="height: calc(100vh - 140px);">
		<h1 class="sr-only">SWaT per-stage anomaly dashboard (Direct Query)</h1>

		<div class="flex min-h-0 flex-col gap-3 overflow-hidden">
			<section aria-label="Plant process flow" class="shrink-0">
				<PlantSchematic stageStatuses={effectiveStatuses} class="max-h-44" />
			</section>

			<section
				class="grid min-h-0 flex-1 grid-cols-6 gap-3 overflow-hidden"
				aria-label="Process stages"
			>
				{#each STAGE_IDS as stageId}
					<StageCard
						{stageId}
						stageName={STAGE_META[stageId]}
						columns={STAGE_COLUMNS[stageId]}
						{liveRow}
						status={effectiveStatuses[stageId]}
						recentLabels={stageLabels[stageId]}
						class="min-h-0 overflow-hidden"
					/>
				{/each}
			</section>
		</div>

		<section class="min-h-0 overflow-hidden" aria-label="Suggested actions">
			<SuggestedActions
				stageStatuses={effectiveStatuses}
				stageNames={STAGE_META}
				{aiSuggestions}
				source={suggestionSource}
				model={reasoningModel}
				onModelChange={handleModelChange}
			/>
		</section>
	</main>

	<section class="border-border border-t" aria-label="Omega embedding visualization">
		<button
			type="button"
			class="text-muted-foreground hover:text-foreground hover:bg-muted/40 flex w-full items-center gap-2 px-4 py-2 font-mono text-xs uppercase tracking-wider transition-colors"
			onclick={() => (embeddingPanelOpen = !embeddingPanelOpen)}
			aria-expanded={embeddingPanelOpen}
		>
			<span>Omega embeddings · 6-stage 2D projection</span>
			{#if embeddingPanelOpen}
				<ChevronDownIcon class="size-3.5" aria-hidden="true" />
			{:else}
				<ChevronUpIcon class="size-3.5" aria-hidden="true" />
			{/if}
		</button>
		{#if embeddingPanelOpen}
			<EmbeddingPanel
				stageIds={MONITORED_STAGE_IDS}
				stageMeta={STAGE_META}
				library={libraryCoords}
				trails={liveTrail}
			/>
		{/if}
	</section>

	{#if sessionStatus === 'error'}
		<div
			class="bg-destructive text-destructive-foreground fixed right-4 bottom-4 max-w-md rounded-md px-4 py-3 font-mono text-xs"
			role="alert"
		>
			Setup error: {setupError}
		</div>
	{/if}
</div>
