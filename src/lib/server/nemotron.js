import { env } from '$env/dynamic/private';

// NVIDIA's hosted API (build.nvidia.com) — OpenAI-compatible chat completions.
const DEFAULT_ENDPOINT = 'https://integrate.api.nvidia.com/v1';
const DEFAULT_MODEL = 'nvidia/nemotron-3-super-120b-a12b';

export function nemotronModel() {
	return env.NEMOTRON_MODEL || DEFAULT_MODEL;
}

export async function queryNemotron({ query, systemPrompt = '', maxTokens = 1024 }) {
	if (!env.NVIDIA_API_KEY) throw new Error('Missing NVIDIA_API_KEY in .env');

	const controller = new AbortController();
	const timeoutId = setTimeout(() => controller.abort(), 120000);
	try {
		const endpoint = (env.NVIDIA_API_ENDPOINT || DEFAULT_ENDPOINT).replace(/\/$/, '');
		const res = await fetch(`${endpoint}/chat/completions`, {
			method: 'POST',
			headers: {
				Authorization: `Bearer ${env.NVIDIA_API_KEY}`,
				'Content-Type': 'application/json'
			},
			body: JSON.stringify({
				model: nemotronModel(),
				messages: [
					{ role: 'system', content: systemPrompt },
					{ role: 'user', content: query }
				],
				temperature: 0.2,
				max_tokens: maxTokens,
				// Turns off Nemotron 3's reasoning trace so the reply is just the JSON.
				chat_template_kwargs: { enable_thinking: false }
			}),
			signal: controller.signal
		});
		if (!res.ok) {
			const err = await res.text().catch(() => '');
			throw new Error(`Nemotron query failed: ${res.status} - ${err}`);
		}
		const data = await res.json();
		const content = data.choices?.[0]?.message?.content ?? '';
		// Strip a reasoning block if the model emits one anyway.
		return content.replace(/<think>[\s\S]*?<\/think>/g, '').trim();
	} finally {
		clearTimeout(timeoutId);
	}
}
