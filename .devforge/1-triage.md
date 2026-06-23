# Triage — PR #35: feat: modernize langchain integration connector

**Problem**: Large umbrella PR modernising the apify/langchain-apify package. Squash-merges four
feature sub-branches (core tools, LangChain-native components, search/crawling, social media)
plus umbrella-level cleanup (package restructure, shared constants, clamp docs, bug fixes).

**Decision**: PROCEED

**Complexity**: large
- 37 files changed; 7 374 insertions, 338 deletions
- Public API changes: new tool classes, renamed env vars (APIFY_TOKEN → APIFY_API_TOKEN alias),
  new convenience exports
- Core/shared code touched: _constants.py, _types.py, tools/base.py, __init__.py
- New package layout (tools/ sub-package replacing flat modules)
- 322 + 30 new unit tests

**Review-only**: yes — task is to review PR #35, not to build new changes.

**Approach sketch**:
1. Verify PR claims against actual diff (fact-check)
2. Design a review scope (what invariants to check)
3. Build iter-1/diff.patch (done: 8 384 lines)
4. Run panel reviewers: staff-review + thermonuclear + code-review
5. Synthesise findings and present summary

**Open questions**:
- Are the sub-PR reviews (the 4 sub-branches) in scope, or only umbrella-level changes?
  → For this run: review the full diff (umbrella + sub-branch content as merged).
- Integration tests require a live APIFY_API_TOKEN; no oracle available, so oracle = skipped (review-only run).
