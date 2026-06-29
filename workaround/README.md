# Workaround: avoiding the gpt-oss-120b invalid-Harmony tool-call failure

This directory documents a reliable workaround for the failure this repo
reproduces, plus a runnable demo (`tools-mode-demo.mjs`).

## Symptom

With `gpt-oss-120b`, an agent that exposes a single `exec` tool and a system
prompt asking the model to *"write a program"* calling several helper functions
(the "code mode" pattern) intermittently fails. The OpenAI-compatible layer
returns:

```
400 ... "Failed to parse tool call from GPT OSS output:
         Expecting property name enclosed in double quotes: line 2 column 3"
```

…or the model gives up and returns the program as plain assistant text with
`finish_reason: "stop"` and no `tool_calls`. The same request can succeed or
fail run-to-run.

## Root cause

The model is emitting a tool call that is **not valid Harmony**.

In the [OpenAI Harmony format][harmony], a call to a **user-defined function**
tool is constrained to JSON and looks like:

```
<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{"city":"SF"}<|call|>
```

The `<|constrain|>json` token before the message body is required for function
tools, and the body after `<|message|>` is the JSON arguments. **Built-in**
tools (e.g. the `python` tool) are different — they do **not** use a
`<|constrain|>json` token and may carry raw code. This distinction is confirmed
by an OpenAI maintainer in the gpt-oss model discussion
([openai/gpt-oss-120b#69][hf69]): function calls use `<|constrain|>json`, while
built-in tools omit the constraint token.

Under the `exec` + "write a program" pattern, gpt-oss-120b instead emits a
**hybrid that matches neither**:

```
<|channel|>commentary to=functions.exec code<|message|>{
  await defineMetadataTool(...);
  await codingAgentTool(...);
  await finalize(...);
}
```

It uses a bare `code` token (built-in-tool style) for a user-defined function,
and puts a raw program — not JSON — after `<|message|>`. The serving layer
expects JSON arguments for a `functions.*` call, tries to parse the program as
JSON, and fails at the first non-JSON token (`await`, at line 2 column 3).

Our reading of *why* the model does this: an `exec(code: string)` tool whose
description tells the model to "write one program in the functions namespace"
closely resembles the built-in `python` tool gpt-oss was trained on, so the
model reaches for built-in-tool-style code emission instead of a JSON function
call. The relevant template/cookbook guidance was also being corrected upstream
in [#69][hf69].

## Why "just lower the temperature" is not a fix

Temperature reduces but does **not** eliminate the failure, and it is not
reliable. In our testing against a `gpt-oss-120b` OpenAI-compatible endpoint
with the `exec` + code-mode prompt:

- At `temperature: 1`, ~40–65% of requests failed with the parse error.
- At `temperature: 0`, **identical** requests for some prompts failed
  **15/15 (100%)**, while the *same request* minutes earlier had passed
  **8/8** — a 0%↔100% swing driven by server-side nondeterminism, not the
  sampling temperature.

So there is no temperature setting that makes the code-mode pattern safe.

## Recommended workaround: declare the routing helpers as real tools

Instead of one `exec` tool plus a "write a program" prompt, declare each routing
helper (`defineMetadataTool`, `schemaAgentTool`, `worldKnowledgeAgentTool`,
`codingAgentTool`, `designAgentTool`, `pricingAgentTool`, `finalize`) as a
**real function tool**, and instruct the model to **call the tools directly, one
step at a time** — never to write a program. The model then emits ordinary,
spec-compliant `<|constrain|>json` function calls.

In our testing this eliminated the failure entirely: **0 invalid-Harmony
failures across 120 multi-turn runs** (temperatures 0 and 1, multiple prompts),
versus up to 100% failure for the code-mode path. Typical runs produced clean
sequences such as
`defineMetadataTool → worldKnowledgeAgentTool → codingAgentTool → finalize`.

### Apply it with the Vercel AI SDK

The current repro builds the agent with a single `exec` tool:

```ts
// before — code mode (intermittently emits invalid Harmony)
new ToolLoopAgent({
  model: provider(MODEL_ID),
  instructions: productionRoutingSystemPrompt(prompt), // "write a single program ..."
  tools: {
    exec: tool({
      description: productionCodeModeDescription(),     // "write one program in functions namespace"
      inputSchema: z.object({ code: z.string() }),
      execute: runProgram,
    }),
  },
})
```

Declare the helpers as real tools instead:

```ts
// after — real tools (spec-compliant function calls)
import { tool, ToolLoopAgent } from 'ai'
import { z } from 'zod'

const agentArg = z.object({ prompt: z.string() })

new ToolLoopAgent({
  model: provider(MODEL_ID),
  instructions:
    'You are the routing agent. Satisfy the request by CALLING the tools in ' +
    'order, one step at a time. Do NOT write a program or emit code. Order: ' +
    'defineMetadataTool first; then schemaAgentTool if shared state is needed; ' +
    'then worldKnowledgeAgentTool if real-world data is needed; then ' +
    'codingAgentTool; optionally designAgentTool / pricingAgentTool; then ' +
    'finalize(message).',
  stopWhen: [/* stop when finalize is called */],
  tools: {
    defineMetadataTool: tool({
      description: 'Define app name, emoji, description, hue. Call first.',
      inputSchema: z.object({
        name: z.string(), emoji: z.string(), description: z.string(), hue: z.number().optional(),
      }),
      execute: defineMetadata,
    }),
    schemaAgentTool:        tool({ description: '…', inputSchema: agentArg, execute: runSchemaAgent }),
    worldKnowledgeAgentTool:tool({ description: '…', inputSchema: agentArg, execute: runWorldKnowledge }),
    codingAgentTool:        tool({ description: '…', inputSchema: agentArg, execute: runCodingAgent }),
    designAgentTool:        tool({ description: '…', inputSchema: agentArg, execute: runDesignAgent }),
    pricingAgentTool:       tool({ description: '…', inputSchema: agentArg, execute: runPricingAgent }),
    finalize: tool({
      description: 'End routing with a short message to the user.',
      inputSchema: z.object({ message: z.string() }),
      execute: async ({ message }) => ({ finalized: true, message }),
    }),
  },
})
```

## Run the demo

`tools-mode-demo.mjs` is dependency-free (Node 18+). It runs the real-tools
routing loop against any OpenAI-compatible `gpt-oss-120b` endpoint and reports
invalid-Harmony failures:

```sh
API_KEY=your_key node workaround/tools-mode-demo.mjs
# or against another endpoint:
API_KEY=your_key BASE_URL=https://your-endpoint/v1 MODEL=gpt-oss-120b \
  TEMPERATURE=0 ATTEMPTS=10 node workaround/tools-mode-demo.mjs
```

It prints a per-prompt verdict breakdown and a sample tool-call sequence, and
exits non-zero only if an invalid-Harmony parse failure occurs.

## Tradeoffs and notes

- This replaces the single-`exec` code-mode batching with per-step function
  calls — more round-trips, but spec-compliant output that parses reliably.
- **If you must keep code mode**, there is no reliable client-side fix today:
  temperature is not dependable (see above). The durable options are a
  provider-side parser that tolerates the non-spec output, or an upstream
  model/template fix (the cookbook/template guidance was being corrected in
  [#69][hf69]).
- Residual behavior with real tools: the model occasionally returns its final
  message as plain assistant text instead of a `finalize` tool call. Treat a
  turn with no tool calls as "done" rather than an error. Transient HTTP 5xx
  from the endpoint are unrelated to this issue.

[harmony]: https://cookbook.openai.com/articles/openai-harmony
[hf69]: https://huggingface.co/openai/gpt-oss-120b/discussions/69
