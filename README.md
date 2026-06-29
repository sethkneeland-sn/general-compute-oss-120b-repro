# General Compute GPT-OSS 120B AI SDK Tool-Call Repro

> **Workaround available:** see [`workaround/`](workaround/README.md) for the
> root cause (with sources), why lowering temperature is not a reliable fix, and
> a validated workaround — declaring the routing helpers as real function tools
> — plus a runnable, dependency-free demo (`workaround/tools-mode-demo.mjs`).

## Hosted Repro

Worker: https://gc-oss120b-repro-20260628-0012.zshannon.workers.dev

Test steps:

1. Open the Worker URL.
2. Paste a General Compute API key into the form.
3. Keep the default prompt: `Make me a fortune cookie app`.
4. Submit the form.
5. Confirm the response summary shows `toolName: "exec"`, `toolChoice: "auto"`, `stream: true`, and a General Compute provider error.
6. In the General Compute dashboard, inspect the matching request. The expected raw model output is Harmony-style `to=functions.exec code` instead of OpenAI-compatible `tool_calls`.

Standalone RWSDK + Vercel AI SDK repro for the `gpt-oss-120b` tool-call parse failure against General Compute's OpenAI-compatible endpoint.

This repo intentionally does not import any application code. It defines one endpoint, `/`, that:

1. calls General Compute `gpt-oss-120b` through `@ai-sdk/openai-compatible`
2. runs the request through `ToolLoopAgent`, matching the production AI SDK pattern
3. gives the model one AI SDK tool named `exec`
4. captures the exact outbound request body that AI SDK sends
5. drains the agent stream and returns either parsed stream parts or the thrown parse error

## Observed Failure

With `toolName: "exec"`, General Compute returns HTTP 400:

```json
{"error":{"message":"Streaming request failed","type":"provider_error","code":"provider_error","param":null}}
```

The matching General Compute dashboard entry shows the provider failed while parsing GPT-OSS output. The model emitted Harmony-style tool syntax:

```text
assistant<|channel|>commentary to=functions.exec code<|message|>{
  await defineMetadataTool(...);
  await codingAgentTool(...);
  await finalize(...);
}
```

That is not OpenAI-compatible `tool_calls` JSON, so the provider reports `Invalid function calling output`.

## Versions

- `ai`: `7.0.2`
- `@ai-sdk/openai-compatible`: `3.0.0`
- `rwsdk`: `1.5.0`
- `vite`: `8.0.16`
- `wrangler`: `4.105.0`

## Run

```sh
npm install
npm run dev
```

Then open `http://localhost:8787/`, paste a General Compute API key, and submit the form.

Equivalent curl:

```sh
curl -sS http://localhost:8787/ \
  -H 'content-type: application/json' \
  -d '{"apiKey":"gc_your_key_here","prompt":"Make me a fortune cookie app"}' | jq .
```

The response includes `capturedRequests[0].bodyJson`, which is the exact OpenAI-compatible request payload produced by Vercel AI SDK.
