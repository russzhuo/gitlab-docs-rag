// Next.js ships type declarations for CSS Modules (*.module.css) but not for
// plain global CSS side-effect imports, so `import "./globals.css"` would
// otherwise fail type-checking in the editor with ts(2882).
declare module "*.css";
declare module "*.scss";
declare module "*.sass";
