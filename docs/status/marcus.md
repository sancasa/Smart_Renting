# Marcus: status

**Lane:** frontend + product UX (`web/`)
**Today's lane goal:** Milestone 1 design checkpoint: scaffold, lease overview, deposit review (sample mode only)
**Branch:** marcus/frontend

## Done (with evidence)
- `web/` scaffolded: Vite + React + TypeScript, plain CSS tokens, Source Sans 3 (self-hosted). The old Flask/Strata code is untouched.
- Lease overview and deposit review work end to end against the **sample adapter** (simulated, not real).
  Evidence: `npm run typecheck`, `npm test` (33 tests), `npm run build` all pass; flows exercised in a browser at 1440, 390 and 320 px.
- Money accounting: integer base units (bigint), each unit in exactly one bucket (confirmed returned, confirmed paid, processing, held); refuses to display amounts that do not reconcile.
- Landlord draft (edit/remove charges, history kept apart from edits), submit, zero-deduction path, tenant accept (exact version) / dispute, stale-version handling, duplicate-submit protection (idempotency keys).

- Lease overview redesign (checkpoint 2, overview only): one deposit summary panel with the headline amount (e.g. "$880.00 returned to you", "$120.00 needs your review") and receipt-style supporting facts; compact header and demo bar; slim progress line; Evidence and Recent activity as disclosure rows. Review layout unchanged apart from the shared compact header/demo bar.
  Evidence: `npm run typecheck`, `npm test` (43 tests), `npm run build` pass; awaiting-response and settled states inspected at 1440, 390 and 320 px (no overflow, targets at least 40px).

- Visual refinement of overview and deposit review (checkpoint 3): cool page `#F6F7F9`, text `#17212B`, primary `#0F5C63`, `#EDF5F3` for financial breakdowns. Overview: labeled 56px amount ("Deductions under review"), tighter content-height panel, top-aligned breakdown, one non-duplicating sentence per state, settlement shown "in progress" (outlined) until funds are resolved. Review: compact labeled comparison regions (about 266x100px), finding/amount/evidence in one short card, fewer bold labels, shorter copy. Demo notice is visible but quiet, controls collapsed.
  Held-funds wording corrected everywhere: disputing does not release money; funds stay held until accepted or the dispute is resolved (regression test covers every state and role).
  Evidence: typecheck, 45 tests, build pass; 41 rendered text/background pairs measured in the browser, lowest 5.55:1; no overflow and no target under 40px at 320px; no new dependencies.

- Interaction polish pass (checkpoint 4, no new features): one shared content container; compact sticky top bar (notice + header, height published as `--top-h` so scroll padding keeps focus clear of it); connected lifecycle timeline under the property heading; refund breakdown sticky on desktop only when the window is at least 960x640; shared `Disclosure` (chevrons) for evidence, activity, history, proposal details, and demo assumptions; versions and exact timestamps moved behind "Proposal details"; demo controls moved into a native modal dialog (closed by default); after navigation the page scrolls to top and the page heading takes focus (back/forward keeps browser scroll restoration).
  Evidence: typecheck, 49 tests, build pass; measured in the running app at 1440, 1100, 800, 390 and 320 px: focus never hidden by the bar in either scroll direction, opening a disclosure moves nothing above it, reduced motion cuts transitions to about 0, no horizontal overflow, no target under 40px, no new dependencies.

- Layout by task and lease stage (checkpoint 5, no new features): active leases keep a compact connected timeline (59px) beside the property heading; once settled it becomes "Settlement complete" plus a "View timeline" disclosure. Deduction-review and settlement pages lose the financial sidebar: a compact summary (`reviewFinancials`, tested) sits directly under the heading, then findings in one centered 960px column with before/after photos side by side and the notes beside them. Detailed activity, proposal details and history stay collapsed; evidence needed to judge a deduction stays visible. "Saved" now shows on the button instead of a reserved line, so nothing reserves empty space.
  Evidence: typecheck, 59 tests, build pass; measured in the running app at 1440, 390 and 320 px: column is exactly 960px and centered, no sidebar or sticky summary, photos side by side, returned/processing/held labels checked per state, no overflow, no target under 40px, no nested scroll outside dialogs, no new dependencies.

- Two component improvements (checkpoint 6): (1) header account button (initials avatar, name, smaller role label, chevron) opening a disclosure-style account panel: identity details plus the one real action, "Lease overview". Escape closes and returns focus to the button, outside click closes without stealing focus, tabbing out closes, choosing the current page just returns focus. No sign-out or settings (no session service exists yet) and no persona switching (that stays in Demo controls). (2) the deposit summary adapts to its content: when "Original deposit" (or a $0.00 row) is the only supporting value it becomes a quiet line under the amount and the panel sizes to its content (686px instead of 1,072px); fuller breakdowns are unchanged.
  Evidence: typecheck, 67 tests, build pass; checked tenant and landlord headers, sparse (active, move-out, zero-deduction processing and settled) and full (awaiting response, confirmed refund, settled with deductions) summaries at 1440, 390 and 320 px; keyboard open/Escape/Tab-out/outside-click/current-page behavior traced with real key presses; no new dependencies.

- Overview grid normalized (checkpoint 7): one centered 960px content container (`--content-max`) shared by the demo notice, header, overview and review, so every left edge sits at 240px and every right edge at 1200px at 1440px. The deposit panel fills the container; the panel and the Evidence/Activity disclosures respond to the container's width (CSS container queries), so they stack when two columns would be cramped (disclosures below 46rem, panel below 44rem). Participant info reduced to the counterparty only (your own identity lives in the account button); development-style wording removed (visible demo hint, "Placeholder…" evidence text, "Demo data…" activity note).
  Evidence: typecheck, 69 tests, build pass; active and waiting-for-landlord states at 1440, 900, 780, 700, 390 and 320 px; focus rings unchanged (3px teal, unclipped); no new dependencies.

## Still simulated / not built
- Everything runs on `web/src/services/sample/`. No backend, wallet, contract, AI or storage is connected.
- The live adapter (`web/src/services/live/`) intentionally throws `integration_blocked` for every call.
- Photos are labeled placeholders. No AI output and no transaction evidence are shown or implied.
- Not in this milestone: inspection, sign-in, invitations, lease creation, funding, optional photo evidence on disputes.

## Sample assumptions (pending team agreement; also listed in the demo controls)
1. Inspections are treated as complete (real completion rules are undecided).
2. Either participant may request move-out (the UI follows `allowedActions` from the server).
3. The landlord can edit charge amounts and reasons.
4. Submitting a proposal starts the undisputed refund; whether posting and refunding are atomic is undecided.
5. No deadlines, no automatic payouts. Sample transfers confirm only when the demo control is pressed.

## Next (max 3)
1. Review the design checkpoint (overview + review) before expanding it.
2. Reusable inspection screens (move-in / move-out) and the entry flows.
3. Replace sample methods with live ones as capabilities are agreed and proven.

## Blocked by / waiting on
- Team agreement on API payloads, wallet model, escrow transitions (see requests below). None of this blocks the sample-mode work.

## Integration requests (proposed types live in `web/src/services/types.ts`; nothing is in `docs/API.md` or `shared/`)
For the backend owner:
- Lease get/list, with `phase`, `stage`, **`allowedActions` for the viewer**, `moveOutAuthorizedRoles`, `waitingOn`, and only policy-supplied deadlines.
- `funds` as separate buckets (confirmed returned, confirmed paid, processing, held) that add up to the original deposit, as integer base-unit strings.
- Proposal draft/save/submit/get with the original AI output kept separate from human edits; server validates totals; tenant must never receive a landlord draft.
- Accept/dispute an exact proposal version; reject stale versions with `stale_proposal`. Idempotency key on every money action.
- One error shape: unauthorized, session_expired, incomplete_evidence, stale_proposal, duplicate_operation, insufficient_funds, service_failure.
- Operation status (pending / succeeded / failed / unknown) with verified transaction data.
For the contract owner:
- Token representation and decimals (frontend assumes the 6-decimal ERC-20 view), allowed transitions and signers, whether posting a proposal and refunding are atomic, how a timeout is triggered (needs a permitted call), ABI + addresses in `shared/`.

## Requests from others (teammates write here instead of editing my files)
-
