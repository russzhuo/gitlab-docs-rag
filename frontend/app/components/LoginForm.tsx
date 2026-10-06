"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { useMutation } from "@tanstack/react-query";
import { api } from "../../lib/http";
import type { User } from "../types";

export default function LoginForm() {
  const router = useRouter();
  const [mode, setMode] = useState<"login" | "register">("login");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");

  const authMutation = useMutation({
    mutationFn: () =>
      api<User>(`/auth/${mode}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, password }),
      }),
    onSuccess: () => router.refresh(),
  });

  function submit(e: React.FormEvent) {
    e.preventDefault();
    authMutation.mutate();
  }

  return (
    <main className="login">
      <h1>Internal handbook assistant</h1>
      <p className="sub">Log in to ask policy questions.</p>
      <form className="auth" onSubmit={submit}>
        <input
          placeholder="Username"
          value={username}
          onChange={(e) => setUsername(e.target.value)}
          autoFocus
        />
        <input
          type="password"
          placeholder="Password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
        />
        {authMutation.isError && <div className="auth-error">{authMutation.error?.message}</div>}
        <button type="submit" disabled={authMutation.isPending || !username || !password}>
          {authMutation.isPending ? "…" : mode === "login" ? "Log in" : "Sign up"}
        </button>
      </form>
      <p className="auth-toggle">
        {mode === "login" ? "No account? " : "Have an account? "}
        <button
          className="link"
          onClick={() => setMode(mode === "login" ? "register" : "login")}
        >
          {mode === "login" ? "Sign up" : "Log in"}
        </button>
      </p>
    </main>
  );
}
