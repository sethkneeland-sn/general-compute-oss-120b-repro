// Workaround demo for the gpt-oss-120b invalid-Harmony tool-call failure.
//
// The original repro gives the model ONE `exec` tool plus a prompt that asks it
// to "write a program" calling several undeclared helper functions (code mode).
// Under that shape gpt-oss-120b intermittently emits a NON-spec Harmony tool
// call — `...to=functions.exec code<|message|>{ raw program }` — using a bare
// `code` token where the Harmony spec requires `<|constrain|>json`, so the
// OpenAI-compatible layer can't parse it and returns 400 "Failed to parse tool
// call from GPT OSS output" (or the model gives up and returns the program as
// plain text). See ../README.md → "Workaround" for the root cause and sources.
//
// This demo shows the workaround: declare the routing helpers as REAL function
// tools and let the model call them via normal function-calling, one step at a
// time. No code mode, no single exec tool.
//
// Dependency-free (Node 18+ global fetch). Works against any OpenAI-compatible
// gpt-oss-120b endpoint:
//
//   API_KEY=...                  (required)
//   BASE_URL=https://api.generalcompute.com/v1   (default)
//   MODEL=gpt-oss-120b           (default)
//   TEMPERATURE=0                (default)
//   ATTEMPTS=10                  (default)
//
//   API_KEY=*** node workaround/tools-mode-demo.mjs

const API_KEY = process.env.API_KEY || process.env.OPENAI_API_KEY;
const BASE_URL = (process.env.BASE_URL || "https://api.generalcompute.com/v1").replace(/\/$/, "");
const MODEL = process.env.MODEL || "gpt-oss-120b";
const TEMPERATURE = Number(process.env.TEMPERATURE ?? "0");
const ATTEMPTS = Number(process.env.ATTEMPTS ?? "10");
const MAX_TURNS = 8;

if (!API_KEY) {
  console.error("Set API_KEY (your gpt-oss-120b endpoint key).");
  process.exit(2);
}

// The routing helpers, declared as real OpenAI function tools (one per step).
const agentArg = {
  type: "object",
  properties: { prompt: { type: "string", description: "Task for the agent." } },
  required: ["prompt"],
};
const fn = (name, description, parameters) => ({ type: "function", function: { name, description, parameters } });

const TOOLS = [
  fn("defineMetadataTool", "Define app name, emoji, description, hue. Call this first.", {
    type: "object",
    properties: {
      name: { type: "string" },
      emoji: { type: "string" },
      description: { type: "string" },
      hue: { type: "integer" },
    },
    required: ["name", "emoji", "description"],
  }),
  fn("schemaAgentTool", "Define shared realtime state schema when the app needs it.", agentArg),
  fn("worldKnowledgeAgentTool", "Fetch real-world data (places, facts, statistics).", agentArg),
  fn("codingAgentTool", "Generate the app code after prerequisites are ready.", agentArg),
  fn("designAgentTool", "Apply an explicit design direction when the user asks for one.", agentArg),
  fn("pricingAgentTool", "Generate pricing/invoice after significant code changes.", agentArg),
  fn("finalize", "End routing with a short, friendly message to the user.", {
    type: "object",
    properties: { message: { type: "string" } },
    required: ["message"],
  }),
];

const SYSTEM = [
  "You are the routing agent for Flick. Satisfy the user's request by CALLING the",
  "available tools in order, one logical step at a time. Do NOT write a program or",
  "emit code — issue normal function calls.",
  "Routing order: defineMetadataTool first; then schemaAgentTool if the app needs",
  "shared realtime state; then worldKnowledgeAgentTool if it needs real-world data;",
  "then codingAgentTool; optionally designAgentTool for an explicit design direction",
  "and pricingAgentTool after significant code. When the request is satisfied, call",
  "finalize(message). The user typed an App Store-style search — build the most",
  "obvious, tightly scoped first result.",
].join(" ");

// Canned tool results so the routing loop can proceed (this is a demo harness;
// in production these are your real agent implementations).
function cannedResult(name) {
  if (name === "defineMetadataTool") return { defined: true, summary: "metadata defined" };
  return { valid: true };
}

async function chat(messages) {
  const res = await fetch(`${BASE_URL}/chat/completions`, {
    method: "POST",
    headers: { Authorization: `Bearer ${API_KEY}`, "Content-Type": "application/json" },
    body: JSON.stringify({ model: MODEL, temperature: TEMPERATURE, messages, tools: TOOLS, tool_choice: "auto" }),
  });
  const text = await res.text();
  return { status: res.status, text };
}

function isInvalidHarmony(text) {
  return /parse tool call from gpt oss/i.test(text) || /provider_error/i.test(text);
}

async function runOnce(prompt) {
  const messages = [
    { role: "system", content: SYSTEM },
    { role: "user", content: prompt },
  ];
  const called = [];
  for (let turn = 1; turn <= MAX_TURNS; turn++) {
    const { status, text } = await chat(messages);
    if (status >= 400) {
      return { verdict: isInvalidHarmony(text) ? "INVALID_HARMONY_400" : `HTTP_${status}`, called };
    }
    const msg = JSON.parse(text).choices?.[0]?.message ?? {};
    const toolCalls = msg.tool_calls ?? [];
    if (toolCalls.length === 0) return { verdict: "PLAINTEXT_NO_TOOLCALL", called };
    messages.push({ role: "assistant", content: msg.content ?? null, tool_calls: toolCalls });
    for (const tc of toolCalls) {
      const name = tc.function.name;
      called.push(name);
      if (name === "finalize") return { verdict: "FINALIZED", called };
      messages.push({ role: "tool", tool_call_id: tc.id, content: JSON.stringify(cannedResult(name)) });
    }
  }
  return { verdict: "MAX_TURNS", called };
}

const PROMPTS = ["Make me a fortune cookie app", "best burritos in sf", "tip calculator"];

(async () => {
  console.log(`endpoint=${BASE_URL} model=${MODEL} temperature=${TEMPERATURE} attempts=${ATTEMPTS}\n`);
  let invalid = 0;
  let total = 0;
  for (const prompt of PROMPTS) {
    const verdicts = [];
    let example = [];
    for (let i = 0; i < ATTEMPTS; i++) {
      const { verdict, called } = await runOnce(prompt);
      verdicts.push(verdict);
      total++;
      if (verdict === "INVALID_HARMONY_400") invalid++;
      if (called.length > example.length) example = called;
    }
    const bad = verdicts.filter((v) => v === "INVALID_HARMONY_400").length;
    const counts = verdicts.reduce((m, v) => ((m[v] = (m[v] || 0) + 1), m), {});
    console.log(`  ${JSON.stringify(prompt).padEnd(34)} invalid-harmony=${bad}/${ATTEMPTS}`);
    console.log(`      verdicts: ${JSON.stringify(counts)}`);
    console.log(`      sample tool sequence: ${example.join(" -> ") || "(none)"}`);
  }
  console.log(`\nTOTAL invalid-Harmony failures: ${invalid}/${total}`);
  console.log(invalid === 0
    ? "PASS — real-tools routing produced no invalid-Harmony tool calls (the target failure)."
    : "Some invalid-Harmony failures occurred — see ../README.md.");
  console.log("(PLAINTEXT_NO_TOOLCALL / MAX_TURNS / transient HTTP 5xx are not the target");
  console.log(" failure — they are benign degradation or server hiccups, not the parse crash.)");
  process.exit(invalid === 0 ? 0 : 1);
})();
