import globals from "globals";
import pluginJs from "@eslint/js";
import tseslint from "typescript-eslint";
import pluginPrettierRecommended from "eslint-plugin-prettier/recommended";

/** @type {import('eslint').Linter.Config[]} */
export default [
  // 1. Target all files
  { files: ["**/*.{js,mjs,cjs,ts}"] },

  // 2. Define environments (Node.js for your CLI tool)
  { languageOptions: { globals: { ...globals.node, ...globals.browser } } },

  // 3. Base JS rules
  pluginJs.configs.recommended,

  // 4. TypeScript rules
  ...tseslint.configs.recommended,

  // 5. Prettier integration (must be last to override others)
  pluginPrettierRecommended,

  // 6. Custom Overrides
  {
    rules: {
      "@typescript-eslint/no-explicit-any": "off",
      "@typescript-eslint/explicit-module-boundary-types": "off",
      "prettier/prettier": "error",
    },
  },
  {
    // Ignore specific folders (replacing .eslintignore)
    ignores: ["dist/", "out/", "node_modules/", "*.min.js"],
  },
];