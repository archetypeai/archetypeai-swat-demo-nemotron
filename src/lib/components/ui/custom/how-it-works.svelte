<script>
	import * as Dialog from '$lib/components/ui/primitives/dialog/index.js';
	import Button from '$lib/components/ui/primitives/button/index.js';

	// One diagram of the offline library build, per-window classification and suggested actions.
	// Columns are 168 apart; every box is 140 wide.
	const COL = [150, 318, 486, 654, 822];
</script>

{#snippet box(x, y, title, sub1, sub2 = '', kind = '', h = 58)}
	<g class="box {kind}">
		<rect {x} {y} width="140" height={h} rx="2" />
		<text class="title" x={x + 70} y={y + (sub2 ? 20 : 24)}>{title}</text>
		<text class="sub" x={x + 70} y={y + (sub2 ? 36 : 41)}>{sub1}</text>
		{#if sub2}<text class="sub" x={x + 70} y={y + 49}>{sub2}</text>{/if}
	</g>
{/snippet}

<Dialog.Root>
	<Dialog.Trigger>
		{#snippet child({ props })}
			<Button variant="outline" size="sm" {...props}>How it works</Button>
		{/snippet}
	</Dialog.Trigger>
	<Dialog.Content class="w-[min(1080px,calc(100vw-2rem))] gap-3 sm:max-w-[1080px]">
		<Dialog.Header>
			<Dialog.Title
				class="font-mono text-xs font-normal tracking-wider text-muted-foreground uppercase"
			>
				How it works
			</Dialog.Title>
		</Dialog.Header>
		<div class="overflow-x-auto">
			<svg
				class="diagram"
				viewBox="0 36 990 400"
				role="img"
				aria-label="Labelled windows are embedded once into a per-stage KNN library. Each live window is z-scored, embedded by Newton and classified per stage by KNN. When the set of attacked stages changes, Nemotron and Newton suggest operator actions in parallel, checked against the plant topology."
			>
				<defs>
					<marker
						id="hiw-arrow"
						viewBox="0 0 8 8"
						refX="7"
						refY="4"
						markerWidth="7"
						markerHeight="7"
						orient="auto-start-reverse"
					>
						<path d="M0 0 L8 4 L0 8 z" />
					</marker>
				</defs>

				<g class="lanes">
					<line x1="0" y1="140" x2="990" y2="140" />
					<line x1="0" y1="252" x2="990" y2="252" />
					<line x1="0" y1="396" x2="990" y2="396" />
				</g>
				<text class="lane" x="16" y="84">Offline</text>
				<text class="lane" x="16" y="98">once</text>
				<text class="lane" x="16" y="196">Every</text>
				<text class="lane" x="16" y="210">window</text>
				<text class="lane" x="16" y="314">On an</text>
				<text class="lane" x="16" y="328">attack</text>

				<!-- Lane 1: build the per-stage KNN library -->
				{@render box(COL[0], 60, 'Labelled windows', 'normal · attack', '128 rows, step 20')}
				{@render box(COL[1], 60, 'z-score', 'global scaler')}
				{@render box(COL[2], 60, 'Newton', 'one call per channel', '', 'newton')}
				{@render box(COL[3], 60, 'knn-library.json', '94 normal · 94 attack', 'per stage')}
				{@render box(COL[4], 60, 'Embedding panel', 'PCA / UMAP of library', '+ live windows')}

				<!-- Lane 2: classify each live window, stage by stage -->
				{@render box(COL[0], 168, 'Plant sensor data', '40 sensors · 6 stages', '128 rows ≈ 2 min')}
				{@render box(COL[1], 168, 'z-score', 'same scaler')}
				{@render box(COL[2], 168, 'Newton', '/api/lens/query', '', 'newton')}
				{@render box(COL[3], 168, 'KNN k=3', 'per stage P1–P6')}
				{@render box(COL[4], 168, 'Stage status', 'normal / attack', '+ embedding coords')}

				<!-- Lane 3: suggest operator actions -->
				{@render box(COL[0], 290, 'Attack set changes', '2 s debounce', '', 'attack')}
				{@render box(COL[1], 290, 'Live sensor values', 'attacked stages', 'vs normal baselines')}
				{@render box(COL[2], 264, 'NVIDIA Nemotron', 'NVIDIA hosted API', '', 'nemotron', 48)}
				{@render box(COL[2], 322, 'Newton', 'Archetype /query', '', 'newton-reason', 48)}
				{@render box(COL[3], 290, 'Topology check', 'targets must match', 'P1 → P6 flow')}
				{@render box(COL[4], 290, 'Suggested actions', 'toggle picks model')}

				<g class="arrows">
					<path d="M290 89 H318" /><path d="M458 89 H486" /><path d="M626 89 H654" /><path
						d="M794 89 H822"
					/>
					<path d="M290 197 H318" /><path d="M458 197 H486" /><path d="M626 197 H654" /><path
						d="M794 197 H822"
					/>
					<path d="M724 118 V168" />
					<path d="M892 168 V118" />
					<path d="M892 226 V242 H220 V290" />
					<path d="M290 319 H318" />
					<path d="M458 312 L486 288" /><path d="M458 326 L486 346" />
					<path d="M626 288 L654 312" /><path d="M626 346 L654 326" />
					<path d="M794 319 H822" />
				</g>
				<text class="note" x="756" y="147">library</text>
				<text class="note" x="922" y="147">coords</text>
				<text class="note" x="556" y="386">both in parallel</text>

				<text class="foot" x="495" y="422">
					The browser posts each window to /api/classify and each attack set to /api/suggestions ·
					no sessions, no streaming
				</text>
			</svg>
		</div>
	</Dialog.Content>
</Dialog.Root>

<style>
	.diagram {
		display: block;
		width: 100%;
		min-width: 820px;
		height: auto;
	}
	.diagram text {
		text-anchor: middle;
	}
	.box rect {
		fill: var(--muted);
		stroke: var(--input);
		stroke-width: 1;
	}
	.box.newton rect {
		stroke: var(--atai-neutral);
		fill: color-mix(in oklch, var(--atai-neutral) 10%, transparent);
	}
	.box.nemotron rect {
		stroke: var(--atai-good);
		fill: color-mix(in oklch, var(--atai-good) 10%, transparent);
	}
	.box.newton-reason rect {
		stroke: var(--chart-1);
		fill: color-mix(in oklch, var(--chart-1) 10%, transparent);
	}
	.box.attack rect {
		stroke: color-mix(in oklch, var(--atai-critical) 50%, transparent);
		fill: color-mix(in oklch, var(--atai-critical) 8%, transparent);
	}
	.title {
		fill: var(--foreground);
		font-family: var(--font-sans);
		font-size: 13px;
	}
	.sub,
	.note,
	.foot {
		fill: var(--muted-foreground);
		font-family: var(--font-mono);
		font-size: 10px;
	}
	.note {
		font-size: 9.5px;
	}
	.diagram .lane {
		text-anchor: start;
		fill: var(--muted-foreground);
		font-family: var(--font-mono);
		font-size: 10px;
		text-transform: uppercase;
		letter-spacing: 0.08em;
	}
	.lanes line {
		stroke: var(--border);
	}
	.arrows path {
		fill: none;
		stroke: var(--muted-foreground);
		stroke-width: 1.2;
		marker-end: url(#hiw-arrow);
	}
	#hiw-arrow path {
		fill: var(--muted-foreground);
	}
</style>
