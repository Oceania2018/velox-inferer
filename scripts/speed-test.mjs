#!/usr/bin/env node
// Response-speed test at several concurrency levels, the way computer-autoplay's "Response
// speed test" measures one request: a fresh random-word prompt of about N tokens (a random nonce
// first, so no prefix cache can skip the prefill), temperature 0, thinking off, and a request to
// count far past the output cap so `max_tokens` ends the reply. Decode and prefill come from the
// server's llama.cpp-style `timings`; end-to-end is completion tokens over client wall time.
//
// usage: node scripts/speed-test.mjs [--endpoint URL] [--model ID] [--concurrency 1,2,4]
//        [--input 1024,4096,8192] [--output 32,128] [--rounds 5] [--json out.json]
//
// Each round starts `c` requests at once and waits for all of them; one warm-up round per
// configuration is discarded. Counting is easy to draft, so speculative decoding accepts more
// than on real agent steps: treat decode rates as an upper bound.

import { writeFileSync } from "node:fs";

const args = Object.fromEntries(
  process.argv.slice(2).reduce((acc, a, i, all) => {
    if (a.startsWith("--")) acc.push([a.slice(2), all[i + 1]]);
    return acc;
  }, []),
);
const endpoint = (args.endpoint ?? "http://127.0.0.1:8080/v1").replace(/\/$/, "");
const model = args.model ?? "qwen3.8-27b";
const list = (s, d) => (s ?? d).split(",").map(Number);
const levels = list(args.concurrency, "1,2,4");
const inputs = list(args.input, "1024,4096,8192");
const outputs = list(args.output, "32,128");
const rounds = Number(args.rounds ?? 5);

// Short common words: one token each (with the leading space) in mainstream BPE vocabularies.
const WORDS = (
  "the of and to in is it you that he was for on are as with his they at be this have from or one had by " +
  "word but not what all were we when your can said there use an each which she do how their if will up " +
  "other about out many then them these so some her would make like him into time has look two more write " +
  "go see number no way could people my than first water been call who oil its now find long down day did " +
  "get come made may part over new sound take only little work know place year live me back give most very"
).split(" ");
const OVERHEAD_TOKENS = 80;
const SYSTEM = "You are a benchmarking assistant. Follow the instruction exactly.";

function prompt(inputTokens, outputTokens) {
  const n = Math.max(0, inputTokens - OVERHEAD_TOKENS);
  const words = Array.from({ length: n }, () => WORDS[Math.floor(Math.random() * WORDS.length)]);
  const nonce = Math.floor(Math.random() * 1e9).toString(36);
  const countTo = Math.max(64, outputTokens * 4);
  return (
    `Run ${nonce}. The passage below is filler for a speed test; ignore its content.\n\n` +
    `<passage>\n${words.join(" ")}\n</passage>\n\n` +
    `Now write the numbers from 1 to ${countTo} separated by single spaces. Output only the numbers.`
  );
}

async function one(input, output) {
  const body = {
    model,
    messages: [
      { role: "system", content: SYSTEM },
      { role: "user", content: prompt(input, output) },
    ],
    max_tokens: output,
    temperature: 0,
    chat_template_kwargs: { enable_thinking: false },
  };
  const t0 = performance.now();
  const res = await fetch(`${endpoint}/chat/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const ms = performance.now() - t0;
  if (!res.ok) throw new Error(`HTTP ${res.status}: ${(await res.text()).slice(0, 200)}`);
  const j = await res.json();
  const t = j.timings ?? {};
  const completion = j.usage?.completion_tokens ?? 0;
  return {
    ms,
    promptTokens: j.usage?.prompt_tokens,
    completion,
    decode: t.predicted_per_second,
    prefillMs: t.prompt_ms,
    prefillRate: t.prompt_n && t.prompt_ms ? (t.prompt_n * 1000) / t.prompt_ms : undefined,
    serverMs: t.total_ms ?? (t.prompt_ms ?? 0) + (t.predicted_ms ?? 0),
    e2e: (completion * 1000) / ms,
  };
}

const median = (xs) => {
  const v = xs.filter((x) => x != null).sort((a, b) => a - b);
  if (!v.length) return NaN;
  const m = v.length >> 1;
  return v.length % 2 ? v[m] : (v[m - 1] + v[m]) / 2;
};

const results = [];
console.log(`endpoint ${endpoint}, model ${model}, ${rounds} rounds per configuration`);
console.log(
  "conc  input  output | decode tok/s  prefill s  prefill tok/s  latency s  max s  e2e tok/s | total tok/s  outside max s  errors",
);
for (const output of outputs) {
  for (const input of inputs) {
    for (const c of levels) {
      const reqs = [];
      let errors = 0;
      let walls = 0;
      let tokens = 0;
      for (let r = 0; r <= rounds; r++) {
        const t0 = performance.now();
        const got = await Promise.allSettled(Array.from({ length: c }, () => one(input, output)));
        const wall = performance.now() - t0;
        if (r === 0) continue; // warm-up
        for (const g of got) {
          if (g.status === "fulfilled") {
            reqs.push(g.value);
            tokens += g.value.completion;
          } else {
            errors++;
            console.error(`  c=${c} in=${input} out=${output}: ${g.reason?.message ?? g.reason}`);
          }
        }
        walls += wall;
      }
      const row = {
        concurrency: c,
        input,
        output,
        promptTokens: median(reqs.map((x) => x.promptTokens)),
        decode: median(reqs.map((x) => x.decode)),
        prefillS: median(reqs.map((x) => x.prefillMs)) / 1000,
        prefillRate: median(reqs.map((x) => x.prefillRate)),
        latencyS: median(reqs.map((x) => x.ms)) / 1000,
        latencyMaxS: Math.max(...reqs.map((x) => x.ms)) / 1000,
        // Client latency not spent in the server's timed work: queueing, proxy, network.
        outsideMaxS: Math.max(...reqs.map((x) => x.ms - x.serverMs)) / 1000,
        e2e: median(reqs.map((x) => x.e2e)),
        throughput: (tokens * 1000) / walls,
        requests: reqs.length,
        errors,
      };
      results.push(row);
      const f = (x, w, d = 0) => (Number.isFinite(x) ? x.toFixed(d) : "-").padStart(w);
      console.log(
        `${f(c, 4)}  ${f(input, 5)}  ${f(output, 6)} | ${f(row.decode, 12)}  ${f(row.prefillS, 9, 2)}  ` +
          `${f(row.prefillRate, 13)}  ${f(row.latencyS, 9, 2)}  ${f(row.latencyMaxS, 5, 2)}  ${f(row.e2e, 9, 1)} | ` +
          `${f(row.throughput, 11)}  ${f(row.outsideMaxS, 13, 2)}  ${f(errors, 6)}`,
      );
    }
  }
}
if (args.json) writeFileSync(args.json, JSON.stringify({ endpoint, model, rounds, results }, null, 2));
