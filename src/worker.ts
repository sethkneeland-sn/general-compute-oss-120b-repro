import { createOpenAICompatible } from '@ai-sdk/openai-compatible'
import { isStepCount, tool, ToolLoopAgent, type StopCondition } from 'ai'
import { route } from 'rwsdk/router'
import { defineApp } from 'rwsdk/worker'
import { z } from 'zod'

const MODEL_ID = 'gpt-oss-120b'

export default defineApp([
  route('/', async ({ request }) => {
    if (request.method === 'GET') {
      return html(renderPage())
    }

    if (request.method !== 'POST') {
      return wantsJson(request) ? json({ error: 'Use POST /' }, 405) : html(renderPage({
        error: 'Use POST /',
      }), 405)
    }

    const input = await readInput(request)
    if (!input.apiKey) {
      const error = 'General Compute API key is required.'
      return wantsJson(request) ? json({ error }, 400) : html(renderPage({ error }), 400)
    }

    const capturedRequests: CapturedRequest[] = []
    const responseCapturePromises: Promise<void>[] = []
    const provider = createOpenAICompatible({
      baseURL: 'https://api.generalcompute.com/v1',
      fetch: async (url, init) => {
        const captured = await captureRequest(url, init)
        capturedRequests.push(captured)

        const response = await fetch(url, init)
        const responseMetadata = captureResponseHeaders(response)
        captured.response = responseMetadata
        responseCapturePromises.push(
          response
            .clone()
            .text()
            .then(text => {
              captured.response = {
                ...responseMetadata,
                bodyText: text,
              }
            })
            .catch(error => {
              captured.response = {
                ...responseMetadata,
                bodyReadError: stringifyError(error),
              }
            }),
        )
        return response
      },
      headers: {
        Authorization: `Bearer ${input.apiKey}`,
      },
      name: 'generalcompute',
    })

    const parts: unknown[] = []
    let error: SerializedError | undefined

    try {
      const agent = new ToolLoopAgent({
        model: provider(MODEL_ID),
        instructions: productionRoutingSystemPrompt(input.prompt),
        stopWhen: [execFinalized, isStepCount(10)],
        temperature: 0,
        tools: {
          exec: tool({
            description: productionCodeModeDescription(),
            inputSchema: z.object({
              code: z
                .string()
                .describe('TypeScript code to execute. Use the available functions as async globals.'),
            }),
            execute: async ({ code }) => ({
              plannedTools: plannedToolsFromCode(code),
              success: true,
              toolCalls: toolCallsFromCode(code),
            }),
          }),
        },
      })

      const result = await agent.stream({ prompt: input.prompt })
      for await (const part of result.fullStream) {
        parts.push(part)
      }
    } catch (caught) {
      error = serializeError(caught)
    }

    await Promise.allSettled(responseCapturePromises)

    const output = {
      capturedRequests,
      error,
      model: {
        id: MODEL_ID,
        provider: 'generalcompute',
      },
      prompt: input.prompt,
      summary: summarizeRequest(capturedRequests[0]),
      streamParts: parts,
      versions: {
        '@ai-sdk/openai-compatible': '3.0.0',
        ai: '7.0.2',
        rwsdk: '1.5.0',
      },
    }

    return wantsJson(request)
      ? json(output)
      : html(renderPage({ output, prompt: input.prompt }))
  }),
])

type CapturedRequest = {
  bodyJson?: unknown
  bodyText?: string
  headers: Record<string, string>
  method: string
  response?: CapturedResponse
  url: string
}

type CapturedResponse = {
  bodyReadError?: string
  bodyText?: string
  headers: Record<string, string>
  status: number
  statusText: string
}

type ReproInput = {
  apiKey: string
  prompt: string
}

type SerializedError = {
  cause?: unknown
  message: string
  name: string
  stack?: string
}

const execFinalized: StopCondition<any, any> = ({ steps }) =>
  steps.some(step =>
    step.toolResults.some(
      result =>
        result.toolName === 'exec' &&
        hasFinalizeToolCall(result.output),
    ),
  )

type ReproToolCall = {
  input: unknown
  name: string
  output: unknown
}

function hasFinalizeToolCall(output: unknown): boolean {
  return (
    typeof output === 'object' &&
    output !== null &&
    Array.isArray((output as { toolCalls?: unknown }).toolCalls) &&
    (output as { toolCalls: ReproToolCall[] }).toolCalls.some(
      call =>
        call.name === 'finalize' &&
        typeof call.output === 'object' &&
        call.output !== null &&
        typeof (call.output as { message?: unknown }).message === 'string',
    )
  )
}

function plannedToolsFromCode(code: string): string[] {
  const tools = [
    'defineMetadataTool',
    'schemaAgentTool',
    'worldKnowledgeAgentTool',
    'codingAgentTool',
    'designAgentTool',
    'pricingAgentTool',
    'finalize',
  ]
  return tools.filter(name => code.includes(`${name}(`))
}

function toolCallsFromCode(code: string): ReproToolCall[] {
  return plannedToolsFromCode(code).map(name => {
    if (name === 'defineMetadataTool') {
      return {
        input: {},
        name,
        output: { defined: true, summary: 'repro mock metadata defined' },
      }
    }

    if (name === 'finalize') {
      return {
        input: extractFinalizeMessage(code),
        name,
        output: {
          finalized: true,
          message: extractFinalizeMessage(code),
        },
      }
    }

    return {
      input: {},
      name,
      output: { valid: true },
    }
  })
}

function extractFinalizeMessage(code: string): string {
  const match = /finalize\(\s*["'`]([^"'`]*)["'`]\s*\)/.exec(code)
  return match?.[1] ?? 'Created your app.'
}

async function captureRequest(
  input: RequestInfo | URL,
  init: RequestInit | undefined,
): Promise<CapturedRequest> {
  const request = input instanceof Request ? input : undefined
  const bodyText = await bodyToText(init?.body)
  const captured: CapturedRequest = {
    bodyText,
    headers: redactHeaders(new Headers(init?.headers ?? request?.headers)),
    method: init?.method ?? request?.method ?? 'GET',
    url: request?.url ?? input.toString(),
  }

  if (bodyText) {
    try {
      captured.bodyJson = JSON.parse(bodyText)
    } catch {}
  }

  return captured
}

function captureResponseHeaders(response: Response): CapturedResponse {
  return {
    headers: headersToObject(response.headers),
    status: response.status,
    statusText: response.statusText,
  }
}

async function bodyToText(body: BodyInit | null | undefined): Promise<string | undefined> {
  if (body == null) return undefined
  if (typeof body === 'string') return body
  if (body instanceof URLSearchParams) return body.toString()
  if (body instanceof Blob) return body.text()
  if (body instanceof ArrayBuffer) return new TextDecoder().decode(body)
  if (ArrayBuffer.isView(body)) {
    return new TextDecoder().decode(body)
  }
  return `[unread ${Object.prototype.toString.call(body)}]`
}

function headersToObject(headers: Headers): Record<string, string> {
  return Object.fromEntries(headers.entries())
}

function summarizeRequest(captured: CapturedRequest | undefined): Record<string, unknown> {
  const body = isRecord(captured?.bodyJson) ? captured.bodyJson : {}
  const tools = Array.isArray(body.tools) ? body.tools : []
  const firstTool = isRecord(tools[0]) ? tools[0] : {}
  const functionSpec = isRecord(firstTool.function) ? firstTool.function : {}
  const responseBody = parseJsonObject(captured?.response?.bodyText)

  return {
    providerError: responseBody?.error,
    providerRequestId: captured?.response?.headers['x-request-id'],
    responseStatus: captured?.response?.status,
    stream: body.stream,
    toolChoice: body.tool_choice,
    toolName: functionSpec.name,
  }
}

function json(value: unknown, status = 200): Response {
  return new Response(JSON.stringify(value, null, 2), {
    headers: { 'content-type': 'application/json; charset=utf-8' },
    status,
  })
}

function html(value: string, status = 200): Response {
  return new Response(value, {
    headers: { 'content-type': 'text/html; charset=utf-8' },
    status,
  })
}

function redactHeaders(headers: Headers): Record<string, string> {
  const result = headersToObject(headers)
  if (result.authorization) {
    result.authorization = 'Bearer <redacted>'
  }
  return result
}

async function readInput(request: Request): Promise<ReproInput> {
  const fallback = 'Make me a fortune cookie app'
  if (!request.body) return { apiKey: '', prompt: fallback }

  const contentType = request.headers.get('content-type') ?? ''
  if (contentType.includes('application/x-www-form-urlencoded') || contentType.includes('multipart/form-data')) {
    const form = await request.formData()
    return {
      apiKey: stringFromForm(form, 'apiKey'),
      prompt: stringFromForm(form, 'prompt') || fallback,
    }
  }

  const body = (await request.json().catch(() => ({}))) as Partial<ReproInput>
  return {
    apiKey: typeof body.apiKey === 'string' ? body.apiKey : '',
    prompt: typeof body.prompt === 'string' && body.prompt.length > 0 ? body.prompt : fallback,
  }
}

function productionRoutingSystemPrompt(prompt: string): string {
  return `# ROLE

You are the routing agent for Flick. Choose the required tool calls, in order,
until the user's request is satisfied.

Make the required routing calls in a single program passed to the exec tool.

## Scope

You can route to tools for UI/code generation, local state, shared realtime
state, static web/reference data, metadata, design, and pricing.

### Out Of Scope

Flick can create tiny, working versions of many App Store-style ideas, but it cannot create apps that depend on external accounts, regulated transactions, large backend networks, or platform ownership.

Hard out of scope:

- Authenticated external APIs: bank accounts, brokerages, delivery platforms, ride-share platforms, calendars, email, social accounts, App Store accounts, vendor dashboards, or private company systems.
- Real money movement: payments, purchases, trading, banking, checkout, escrow, invoices that charge users, crypto transfers, gambling, or insurance.
- User identity systems: login, signup, account recovery, OAuth, password storage, user profiles backed by a server, or permissions/admin systems.
- Server-side product logic: marketplaces, feeds, recommendation networks, moderation queues, creator platforms, booking platforms, multiplayer servers, SaaS admin panels, or anything requiring a custom backend.
- Regulated professional workflows: diagnosis, treatment, legal advice, tax filing, investment decisions, prescription changes, emergency dispatch, or compliance systems.
- Device/system integrations not provided by Flick tools: contacts, inboxes, calendars, HealthKit, payment terminals, Bluetooth devices, HomeKit devices, background location fleets, or push-notification infrastructure unless a dedicated tool already exists for that capability.

## Flick Definition

A flick is pixel-sized software: a native iOS app generated on demand from
natural language and running on the user's phone in about one second. It is
scaled to the specific situation, not to an App Store category.

A flick does one real thing directly. It has no onboarding, login, setup funnel,
subscription pitch, newsletter, content feed, settings maze, or platform
ambition.

For routing, faithfully and minimally interpret the user's request. Prefer the
smallest complete app or change that is correct for the request over a broad
product concept.

# ROUTING POLICY

## Routing Order

Follow this routing order:

1. \`defineMetadataTool\` - define app name, emoji, description, and hue before code generation.
2. \`schemaAgentTool\` - define quantum schema if the requested flick needs shared realtime state and no matching store exists.
3. \`worldKnowledgeAgentTool\` - fetch real-world data if the request needs facts, places, current info, or statistics and no matching artifact exists.
4. \`codingAgentTool\` - generate code after required schema and world knowledge are ready.

Do not skip schema or world-knowledge prerequisites. Code generated without required stores or data will be wrong.

## Tool Selection

- Metadata creation or metadata property changes -> \`defineMetadataTool\`.
- Specific aesthetic/vibe/mood requests -> \`designAgentTool\`, then \`codingAgentTool\` to apply. Only call for explicit design directions such as "make it brutalist", "editorial feel", or "playful and colorful". Do not call for generic app requests; the default design is applied automatically.
- Shared state structure -> \`schemaAgentTool\`.
- Real-world data needs -> \`worldKnowledgeAgentTool\`.
- Coding/app code generation -> \`codingAgentTool\` after required schema and world knowledge are gathered. See REQUEST CONTEXT for task-string rules.
- Feature pricing/invoice -> \`pricingAgentTool\` after significant code changes.

## Using schemaAgentTool

Push notifications and geolocation extensions provide their own backend state management; don't create stores for device tokens, subscriptions, or location data.

Schema tools define structure only. Coding initializes data through generated hooks.

After \`schemaAgentTool\` succeeds, \`codingAgentTool\` receives generated store hooks from \`./stores\`.

## Using worldKnowledgeAgentTool

Requests that need world knowledge call \`worldKnowledgeAgentTool\` before coding:

- "make me an app with the best po'boys in new orleans" -> restaurant data
- "show me the last 10 Super Bowl winners" -> sports history
- "list the tallest buildings in NYC" -> architecture data
- "track my favorite stocks" -> stock ticker data
- "best burritos in sf" -> restaurant data

Pass a description of what data is needed. The world knowledge agent determines the query and schema.

The tool saves the data as an artifact that \`codingAgentTool\` can import.

# EXECUTION POLICY

## Tool Result Meanings

- \`defineMetadataTool\` with \`"defined": true\` = Metadata is stored. Proceed to the next required routing step.
- \`schemaAgentTool\` with \`"valid": true\` = Store created or updated. Proceed to coding after all prerequisites are satisfied.
- \`worldKnowledgeAgentTool\` with \`"valid": true\` = World knowledge gathered. Proceed to coding after all prerequisites are satisfied.
- \`codingAgentTool\` with \`"valid": true\` = Code is generated and secured. Move on only for remaining requested work.
- \`designAgentTool\` with \`"valid": true\` = Design vibe applied. Proceed to coding or finish.

## Execution

Call exec once with a single program that executes all required routing steps in order.
Check ROUTING POLICY before choosing tools.

For straightforward requests, call \`finalize("Created your app!")\` at the
end to skip review. Use a brief, friendly message describing what you built.

## Prevention Rules

- DO NOT call the same tool twice for the same task.
- DO NOT loop or ignore successful results.
- DO NOT skip \`defineMetadataTool\` before metadata exists.

# CODE-MODE API

Call the exec tool exactly once. Its code argument holds a single TypeScript program. The functions below are async globals inside that program. They are not tools, so call them only from inside the program.

declare function defineMetadataTool(input: {
  name: string;
  emoji: string;
  description: string;
  hue?: number;
}): Promise<{
  defined: boolean;
  summary: string;
}>;

declare function codingAgentTool(input: {
  prompt?: string;
  request?: string;
  task?: string;
}): Promise<{
  valid: boolean;
  error?: string;
}>;

declare function designAgentTool(input: {
  prompt?: string;
  request?: string;
  task?: string;
}): Promise<{
  valid: boolean;
  error?: string;
}>;

declare function pricingAgentTool(input: {
  prompt?: string;
  request?: string;
  task?: string;
}): Promise<{
  valid: boolean;
  error?: string;
}>;

declare function schemaAgentTool(input: {
  prompt?: string;
  request?: string;
  task?: string;
}): Promise<{
  valid: boolean;
  error?: string;
}>;

declare function worldKnowledgeAgentTool(input: {
  prompt?: string;
  request?: string;
  task?: string;
}): Promise<{
  valid: boolean;
  error?: string;
}>;

declare function finalize(input: string): Promise<{
  finalized: boolean;
  message: string;
}>;

# CURRENT STATE

## APP METADATA

**No metadata exists.**

## GENERATED CODE

**No code generated yet.**

## COLOR PALETTE

**No color palette set.**

## QUANTUM STORES

**No stores defined yet.**

## WORLD KNOWLEDGE

**No world knowledge gathered yet.**

## INVOICE

**No invoice generated yet.**

# REQUEST CONTEXT

## CURRENT TURN

In an App Store search style interface, the user input:

${prompt}

The user did not select a suggestion. Generate the most obvious, normal, tightly scoped first result for this search input.

Example expansions:

- "bingo card maker" -> a bingo card maker for creating and playing custom bingo boards
- "tip calculator" -> a simple bill tip calculator with totals people can use immediately
- "baby chores" -> a baby chores tracker for logging feedings, diapers, naps, and whose turn is next
- "packing list" -> a packing list builder for trips with checkable items
- "shot tracker" -> a basketball shot tracker for makes, misses, and quick shooting stats

## codingAgentTool Task String

For this new app, expand the App Store search input from CURRENT TURN into the obvious first-result app request. The task should be short, normal, and complete enough for the coding agent to build the flick. Do not pass an ambiguous bare keyword; use a small expansion for clarity. Do not mention frameworks, UI structure, storage mechanics, or extra features beyond the normal first result.

Examples:

- "bingo card maker" -> \`{ "prompt": "Create a bingo card maker for making and playing custom bingo cards" }\`
- "tip calculator" -> \`{ "prompt": "Build a simple tip calculator for splitting a bill and tip" }\`
- "baby chores" -> \`{ "prompt": "Create a baby chores tracker for feedings, diapers, naps, and turns" }\`
- "packing list" -> \`{ "prompt": "Create a packing list builder with checkable trip items" }\`
- "shot tracker" -> \`{ "prompt": "Create a basketball shot tracker for makes, misses, and quick shooting stats" }\``
}

function productionCodeModeDescription(): string {
  return `Runs one TypeScript program that makes all the required routing calls in order. Put the whole program in the code argument as a JSON string. Inside that string, write each line break as \\n, and do not end a line with a backslash. Write bare statements with top-level await. Do not define functions. Call finalize(message) as the last statement when you do not need to inspect any result. The routing functions the program can call are listed under CODE-MODE API in the instructions.`
}

function renderPage(options: {
  error?: string
  output?: unknown
  prompt?: string
} = {}): string {
  const prompt = options.prompt ?? 'Make me a fortune cookie app'
  return `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>General Compute codemode repro</title>
  <style>
    body { color: #111827; font-family: system-ui, sans-serif; line-height: 1.4; margin: 32px; max-width: 960px; }
    form { display: grid; gap: 12px; margin-bottom: 24px; }
    input, textarea { border: 1px solid #d1d5db; border-radius: 6px; font: inherit; padding: 8px 10px; }
    textarea { min-height: 92px; }
    button { align-self: start; background: #111827; border: 0; border-radius: 6px; color: white; cursor: pointer; font: inherit; padding: 8px 12px; }
    pre { background: #f3f4f6; border-radius: 6px; overflow: auto; padding: 12px; }
    .error { color: #b91c1c; font-weight: 600; }
  </style>
</head>
<body>
  <h1>General Compute codemode repro</h1>
  <form method="post">
    <label>
      General Compute API key
      <input name="apiKey" type="password" autocomplete="off" required>
    </label>
    <label>
      Prompt
      <textarea name="prompt">${escapeHtml(prompt)}</textarea>
    </label>
    <button type="submit">Run repro</button>
  </form>
  ${options.error ? `<p class="error">${escapeHtml(options.error)}</p>` : ''}
  ${options.output ? `<h2>Result</h2><pre>${escapeHtml(JSON.stringify(options.output, null, 2))}</pre>` : ''}
</body>
</html>`
}

function escapeHtml(value: string): string {
  return value
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;')
}

function stringFromForm(form: FormData, name: string): string {
  const value = form.get(name)
  return typeof value === 'string' ? value.trim() : ''
}

function wantsJson(request: Request): boolean {
  const accept = request.headers.get('accept') ?? ''
  const contentType = request.headers.get('content-type') ?? ''
  return accept.includes('application/json') || contentType.includes('application/json')
}

function serializeError(error: unknown): SerializedError {
  if (error instanceof Error) {
    return {
      cause: error.cause,
      message: error.message,
      name: error.name,
      stack: error.stack,
    }
  }
  return {
    message: String(error),
    name: typeof error,
  }
}

function stringifyError(error: unknown): string {
  return error instanceof Error ? `${error.name}: ${error.message}` : String(error)
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function parseJsonObject(value: string | undefined): Record<string, unknown> | undefined {
  if (!value) return undefined
  try {
    const parsed: unknown = JSON.parse(value)
    return isRecord(parsed) ? parsed : undefined
  } catch {
    return undefined
  }
}
