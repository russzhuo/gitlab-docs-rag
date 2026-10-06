import LoginForm from "./components/LoginForm";
import Chat from "./components/Chat";
import { getServerUser } from "../lib/auth";

// Server component: check auth on the server (via the session cookie forwarded
// to FastAPI) so there is no client-side "loading" flash.
export default async function Home() {
  const user = await getServerUser();
  if (!user) return <LoginForm />;
  return <Chat user={user} />;
}
