import type { Metadata } from "next";
import "./globals.css";
import Providers from "./providers";
import { Noto_Sans } from "next/font/google";
import { cn } from "@/lib/utils";

const notoSans = Noto_Sans({ subsets: ["latin"], variable: "--font-sans" });

export const metadata: Metadata = {
  title: "Handbook Assistant",
  description: "Answers only from the indexed GitLab Handbook subset.",
  icons: { icon: "/static/favicon.png" },
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className={cn("font-sans", notoSans.variable)}>
      <body>
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
