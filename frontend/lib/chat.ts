import type { Citation, Message } from "@/app/types";
import type { ThreadMessageLike } from "@assistant-ui/react";
import type { components } from "@/lib/api-types";
import { conversationApi } from "@/lib/api/conversations";

type TextPart = { type: string; text?: string };

export function messageText(parts: readonly TextPart[]): string {
  return parts
    .filter((part) => part.type === "text" && part.text)
    .map((part) => part.text)
    .join("");
}

function sourcePart(c: Citation) {
  // 用有序列表，1. 2. 3. 正好和答案里引用的 [1] [2] 对上
  return `1. [${c.title}](${c.source})`;
}

/** Text part + citation source parts, for an assistant message body. */
function assistantContent(
  answer: string,
  citations: Citation[],
): ThreadMessageLike["content"] {
  const sources = citations.map(sourcePart).filter(Boolean).join("\n");

  const text =
    answer && sources
      ? `${answer}\n\n**Sources:**\n\n${sources}`
      : answer;
  return [{ type: "text", text }];
}

/** Convert a backend Message into assistant-ui's loose message shape. */
export function toThreadMessageLike(m: Message, id: string): ThreadMessageLike {
  return {
    id,
    role: m.role,
    content:
      m.role === "assistant"
        ? assistantContent(m.content, m.citations)
        : [{ type: "text", text: m.content }],
    // assistant-ui only allows `status` on assistant messages.
    ...(m.role === "assistant"
      ? { status: { type: "complete", reason: "stop" } as const }
      : {}),
  };
}

export function userMessage(id: string, text: string): ThreadMessageLike {
  return { id, role: "user", content: [{ type: "text", text }] };
}

export function runningAssistantMessage(id: string): ThreadMessageLike {
  return {
    id,
    role: "assistant",
    content: [{ type: "text", text: "" }],
    status: { type: "running" },
  };
}

export function completeAssistantMessage(
  m: ThreadMessageLike,
  answer: string,
  citations: Citation[],
): ThreadMessageLike {
  return {
    ...m,
    content: assistantContent(answer, citations),
    status: { type: "complete", reason: "stop" },
  };
}

export type StreamEvent =
  | components["schemas"]["MetaEvent"]
  | components["schemas"]["TokenEvent"]
  | components["schemas"]["DoneEvent"]
  | components["schemas"]["RefusedEvent"]
  | components["schemas"]["ConversationEvent"]
  | components["schemas"]["ErrorEvent"];

/** Event discriminator strings, kept in sync with the generated union. */
export const EventType = {
  Meta: "meta",
  Token: "token",
  Done: "done",
  Refused: "refused",
  Conversation: "conversation",
  Error: "error",
} as const satisfies Record<string, StreamEvent["type"]>;

function parseLine(line: string): StreamEvent | null {
  const trimmed = line.trim();
  if (!trimmed) return null;
  let data = trimmed;
  if (data.startsWith("data:")) data = data.slice(5).trimStart();
  if (data === "[DONE]") return null;
  try {
    return JSON.parse(data) as StreamEvent;
  } catch {
    return null;
  }
}

/** Parse the backend's SSE (`data: {...}` lines) into typed events. */
export async function* parseSse(
  body: ReadableStream<Uint8Array>,
): AsyncGenerator<StreamEvent> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) {
        buffer += decoder.decode();
        break;
      }
      buffer += decoder.decode(value, { stream: true });
      let idx: number;
      while ((idx = buffer.indexOf("\n")) !== -1) {
        const line = buffer.slice(0, idx);
        buffer = buffer.slice(idx + 1);
        const ev = parseLine(line);
        if (ev) yield ev;
      }
    }
    for (const line of buffer.split("\n")) {
      const ev = parseLine(line);
      if (ev) yield ev;
    }
  } finally {
    reader.releaseLock();
  }
}

export type StreamAnswerResult = {
  answer: string;
  citations: Citation[];
  conversationId: number | null;
};

/**
 * Stream an answer from the backend. Calls `onText` with the full accumulated
 * text whenever it changes (each token, and once more for `refused` / `error`
 * answers), and returns the final answer, citations, and the (possibly new)
 * backend conversation id.
 */
export async function streamAnswer(
  question: string,
  conversationId: string | null,
  onAnswerUpdate: (text: string) => void,
): Promise<StreamAnswerResult> {
  let answer = "";
  const citations: Citation[] = [];
  let nextConversationId: number | null = null;

  try {
    const body = await conversationApi.stream(question, conversationId);
    for await (const ev of parseSse(body)) {
      if (ev.type === EventType.Token) {
        answer += ev.text;
        onAnswerUpdate(answer);
      } else if (ev.type === EventType.Meta) {
        citations.push(...ev.citations);
      } else if (ev.type === EventType.Refused) {
        answer = ev.answer;
        onAnswerUpdate(answer);
      } else if (ev.type === EventType.Conversation) {
        nextConversationId = ev.id;
      } else if (ev.type === EventType.Error) {
        answer = ev.message || "Something went wrong.";
        onAnswerUpdate(answer);
      }
    }
  } catch {
    if (!answer) answer = "Something went wrong.";
  }

  return { answer, citations, conversationId: nextConversationId };
}

export type MessageSetter = (
  updater: (prev: ThreadMessageLike[]) => ThreadMessageLike[],
) => void;

/**
 * Run a full user → assistant exchange: append the user message and an empty
 * "running" assistant bubble, stream the answer into it, then finalize with
 * citations and a complete status. Returns the (possibly new) backend
 * conversation id.
 */
export async function runExchange(
  question: string,
  conversationId: string | null,
  setMessages: MessageSetter,
): Promise<number | null> {
  const userId = crypto.randomUUID();
  const assistantId = crypto.randomUUID();

  setMessages((prev) => [
    ...prev,
    userMessage(userId, question),
    runningAssistantMessage(assistantId),
  ]);

  const patch = (text: string) =>
    setMessages((prev) =>
      prev.map((m) =>
        m.id === assistantId ? { ...m, content: [{ type: "text", text }] } : m,
      ),
    );

  const {
    answer,
    citations,
    conversationId: nextId,
  } = await streamAnswer(question, conversationId, patch);

  setMessages((prev) =>
    prev.map((m) =>
      m.id === assistantId ? completeAssistantMessage(m, answer, citations) : m,
    ),
  );

  return nextId;
}
