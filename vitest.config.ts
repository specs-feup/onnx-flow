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
    include: ["test/unit/**/*.test.ts", "test/integration/**/*.test.ts"],
    exclude: [...configDefaults.exclude, "out/**"],
    coverage: {
      provider: "v8",
      reporter: ["text", "json", "html"],
      include: ["src/**/*.ts"],
      exclude: ["src/index.ts", "src/ExplorerAPI/**"],
    },
    benchmark: {
      include: ["test/benchmark/**/*.bench.ts"],
    },
  },
});