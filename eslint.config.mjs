import pluginJs from '@eslint/js';
import pluginPrettierRecommended from 'eslint-plugin-prettier/recommended';
import globals from 'globals';
import tseslint from 'typescript-eslint';

/** @type {import('eslint').Linter.Config[]} */
export default [
  // 1. Target all files
  {files: ['**/*.{js,mjs,cjs,ts}']},

  // 2. Define environments and Type-Aware Parser Options
  {
    languageOptions: {
      globals: {...globals.node, ...globals.browser},
      parserOptions: {
        projectService: true, // Recommended for typescript-eslint v8+
        tsconfigRootDir: import.meta.dirname, // Ensures it finds your tsconfig.json
      },
    }
  },

  // 3. Base JS rules
  pluginJs.configs.recommended,

  // 4. TypeScript rules
  ...tseslint.configs.recommended,

  // 5. Prettier integration (must be last to override others)
  pluginPrettierRecommended,

  // 6. Custom Overrides
  {
    rules: {
      '@typescript-eslint/consistent-type-imports': 'error',
      '@typescript-eslint/ban-ts-comment': ['error', {
          'ts-expect-error': 'allow-with-description',
          'ts-ignore': true, // or 'allow-with-description'
          'ts-nocheck': true,
          'ts-check': false,
      }],

      // Allow explicit "any" for now
      '@typescript-eslint/no-explicit-any': 'error',

      '@typescript-eslint/explicit-module-boundary-types': 'error',
      'prettier/prettier': 'error',

      // Flags unnecessary optional chaining (?.) and conditions that are always true/false
      '@typescript-eslint/no-unnecessary-condition': 'warn', 
      
      // Flags risky boolean checks (like checking an array instead of array.length)
      '@typescript-eslint/strict-boolean-expressions': 'warn',

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