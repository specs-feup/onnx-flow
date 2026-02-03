import pluginJs from '@eslint/js';
import pluginPrettierRecommended from 'eslint-plugin-prettier/recommended';
import globals from 'globals';
import tseslint from 'typescript-eslint';

/** @type {import('eslint').Linter.Config[]} */
export default [
  // 1. Target all files
  {files: ['**/*.{js,mjs,cjs,ts}']},

  // 2. Define environments (Node.js for your CLI tool)
  {languageOptions: {globals: {...globals.node, ...globals.browser}}},

  // 3. Base JS rules
  pluginJs.configs.recommended,

  // 4. TypeScript rules
  ...tseslint.configs.recommended,

  // 5. Prettier integration (must be last to override others)
  pluginPrettierRecommended,

  // 6. Custom Overrides
  {
    rules: {
      // Allow explicit "any" for now
      '@typescript-eslint/no-explicit-any': 'off',

      '@typescript-eslint/explicit-module-boundary-types': 'off',
      'prettier/prettier': 'error',

      // Configure unused variables to ignore those starting with an underscore
      '@typescript-eslint/no-unused-vars': [
        'error', {
          'argsIgnorePattern': '^_',
          'varsIgnorePattern': '^_',
          'caughtErrorsIgnorePattern': '^_'
        }
      ],

      // Allow namespaces and empty interfaces because of flow
      '@typescript-eslint/no-namespace': 'off',
      '@typescript-eslint/no-empty-object-type': 'off',
    },
  },
  {
    // Ignore specific folders (replacing .eslintignore)
    ignores: ['dist/', 'out/', 'node_modules/', '*.min.js'],
  },
];