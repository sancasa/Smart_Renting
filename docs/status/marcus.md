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
