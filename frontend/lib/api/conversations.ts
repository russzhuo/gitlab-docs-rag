import type { Conversation, Message } from "@/app/types";
import { api, apiStream } from "../http";

const keys = {
  // Root scope: invalidate this to refresh the list and every conversation's
  // messages at once.
  all: ["conversations"] as const,
  list: () => ["conversations"] as const,
  messages: (id: string) => [...keys.all, id, "messages"] as const,
};

export const conversationApi = {
  keys,
  list: () => api<Conversation[]>("/conversations"),
  messages: (id: string) => api<Message[]>(`/conversations/${id}/messages`),
  stream: (question: string, conversationId: string | null, signal?: AbortSignal) =>
    apiStream("/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, conversation_id: conversationId }),
      signal,
    }),
};
