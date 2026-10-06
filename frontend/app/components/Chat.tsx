"use client";

import { useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import type { User } from "@/app/types";
import { authApi } from "@/lib/api/auth";
import { conversationApi } from "@/lib/api/conversations";
import { messageText, runExchange, toThreadMessageLike } from "@/lib/chat";
import {
  AssistantRuntimeProvider,
  useExternalStoreRuntime,
  type ThreadMessageLike,
} from "@assistant-ui/react";
import { Thread } from "@/components/assistant-ui/elements/thread.aui";
import { Button } from "@/components/ui/button";

export default function Chat({ user }: { user: User }) {
  const router = useRouter();
  const queryClient = useQueryClient();

  const [activeId, setActiveId] = useState<number | null>(null);
  const [messages, setMessages] = useState<ThreadMessageLike[]>([]);
  const activeIdRef = useRef<number | null>(null);
  activeIdRef.current = activeId;

  const conversationsQuery = useQuery({
    queryKey: conversationApi.keys.list(),
    queryFn: conversationApi.list,
  });

  useEffect(() => {
    if (activeId === null) {
      setMessages([]);
      return;
    }
    let alive = true;
    setMessages([]);
    conversationApi
      .messages(String(activeId))
      .then((msgs) => {
        if (!alive) return;
        setMessages(
          msgs.map((m, i) => toThreadMessageLike(m, `${activeId}-${i}`)),
        );
      })
      .catch(() => {});
    return () => {
      alive = false;
    };
  }, [activeId]);

  // Open the most recent conversation once the list has loaded.
  const didInitActive = useRef(false);
  useEffect(() => {
    if (didInitActive.current || !conversationsQuery.data) return;
    didInitActive.current = true;
    if (conversationsQuery.data[0]) setActiveId(conversationsQuery.data[0].id);
  }, [conversationsQuery.data]);

  const runtime = useExternalStoreRuntime({
    messages,
    setMessages: (msgs) => setMessages([...msgs]),
    convertMessage: (m) => m,
    onNew: async (message) => {
      const question = messageText(message.content);
      const conversationId = await runExchange(
        question,
        activeIdRef.current === null ? null : String(activeIdRef.current),
        setMessages,
      );

      if (conversationId !== null && activeIdRef.current === null) {
        setActiveId(conversationId);
      }
      queryClient.invalidateQueries({ queryKey: conversationApi.keys.all });
    },
  });

  async function logout() {
    try {
      await authApi.logout();
      router.refresh();
    } catch {
      /* ignore */
    }
  }

  const conversations = conversationsQuery.data ?? [];

  const main = (
    <main className="min-w-0 flex-1">
      <AssistantRuntimeProvider runtime={runtime}>
        <Thread />
      </AssistantRuntimeProvider>
    </main>
  );

  const header = (
    <header className="flex items-center gap-3 border-b px-4 py-2.5">
      <div className="text-sm font-semibold">Handbook Assistant</div>
      <div className="flex-1" />
      <Button variant="outline" size="sm" onClick={() => setActiveId(null)}>
        New chat
      </Button>
      <span className="text-sm text-muted-foreground">{user.username}</span>
      <Button variant="ghost" size="sm" onClick={logout}>
        Log out
      </Button>
    </header>
  );

  const aside = (
    <aside className="bg-[#f9fafb] flex w-64 shrink-0 flex-col gap-1 overflow-y-auto border-r p-2">
      <button
        type="button"
        onClick={() => setActiveId(null)}
        className={`rounded-md px-3 py-2 text-left text-sm ${
          activeId === null ? "bg-muted font-medium" : "hover:bg-muted/50"
        }`}
      >
        New chat
      </button>
      {conversations.map((c) => (
        <button
          key={c.id}
          type="button"
          onClick={() => setActiveId(c.id)}
          className={`truncate rounded-md px-3 py-2 text-left text-sm ${
            activeId === c.id ? "bg-muted font-medium" : "hover:bg-muted/50"
          }`}
        >
          {c.title}
        </button>
      ))}
    </aside>
  );

  return (
    <div className="flex h-dvh flex-col">
      {header}

      <div className="flex min-h-0 flex-1">
        {aside}
        {main}
      </div>
    </div>
  );
}
