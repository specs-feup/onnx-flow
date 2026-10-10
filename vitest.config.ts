import { defineConfig, configDefaults } from "vitest/config";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  resolve: {
    alias: {
      "@specs-feup/onnx-flow": path.resolve(__dirname, "src"),
    },
  },
  test: {
    globals: true,
    environment: "node",
    testTimeout: 30000, // 30s timeout for ORT WASM & CLI runs
    hookTimeout: 30000,
    setupFiles: ["test/setup.ts"], // Global proto resolution hook
    include: ["test/unit/**/*.test.ts", "test/integration/**/*.test.ts"],
    exclude: [...configDefaults.exclude, "out/**"],
    coverage: {
      provider: "v8",
      reporter: ["text", "json", "html"],
      include: ["src/**/*.ts"],
      exclude: [
        "src/ExplorerAPI/**",
        "**/cgra-decomposition-example/**",
      ],
    },
    benchmark: {
      include: ["test/benchmark/**/*.bench.ts"],
    },
  },
});