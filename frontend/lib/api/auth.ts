import type { User } from "@/app/types";
import { api, serverApi } from "../http";

const keys = {
  me: ["auth", "me"] as const,
};

export const authApi = {
  keys,
  logout: () => api("/auth/logout", { method: "POST" }),
  me: (session: string | undefined) => serverApi<User>("/auth/me", session),
};
