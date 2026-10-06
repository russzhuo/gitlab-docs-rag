import { cookies } from "next/headers";
import type { User } from "@/app/types";
import { authApi } from "@/lib/api/auth";

// Server component helper: check the session cookie against FastAPI /auth/me.
export async function getServerUser(): Promise<User | null> {
  const cookieStore = await cookies();
  const session = cookieStore.get("session")?.value;
  return authApi.me(session);
}
