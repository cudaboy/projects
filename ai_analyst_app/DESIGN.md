# AI Analyst App DESIGN.md

Reference selected from `VoltAgent/awesome-design-md`: **Coinbase** (`https://getdesign.md/coinbase/design-md`).

## Style Direction

Use a Toss-like blue fintech surface inspired by Coinbase's clean blue identity: trustworthy, minimal, institutional, fast, and legible.

## Color Tokens

- `--blue-700`: #0052ff — primary CTA / active navigation
- `--blue-600`: #0a66ff — secondary action / focus
- `--blue-100`: #e8f1ff — pale blue background panels
- `--ink-900`: #0a0f1f — primary text
- `--ink-600`: #4b587c — secondary text
- `--line`: #dce6f5 — subtle borders
- `--surface`: #ffffff — cards
- `--canvas`: #f5f8ff — app background
- `--positive`: #16a34a
- `--warning`: #f59e0b
- `--danger`: #ef4444

## Typography

- Prefer system sans: Inter, Pretendard, SF Pro, Segoe UI, Arial, sans-serif.
- Hero/title: 44-64px desktop, 34-42px mobile, weight 800.
- Section heading: 24-32px, weight 750.
- Body: 15-17px, line-height 1.65.
- Numeric dashboard values: tabular nums, 24-36px, weight 800.

## Components

- Cards use white surface, 1px blue-gray border, 24-32px radius, soft shadow.
- Buttons are pill-shaped. Primary buttons use blue fill and white text.
- Inputs are large, rounded, white, with blue focus ring.
- Dashboard panels should feel like a banking/fintech product, not a developer console.
- Use spacious vertical rhythm and strong hierarchy.

## Layout Rules

- Landing hero: blue gradient background with a large rounded analysis panel.
- Main workspace: two-column responsive grid — left control panel, right result/report area.
- History/performance: table/cards with blue accent chips.
- Mobile: stack all columns, keep primary CTA sticky-feeling and easy to tap.

## Interaction Rules

- Show clear loading states for long agent analysis.
- Keep error messages concise with retry guidance.
- Parse score JSON when available and render score cards before markdown reports.
- Always include the investment disclaimer near final reports.
