Review PR #35: "feat: modernize langchain integration connector" on apify/langchain-apify.

This is a large umbrella PR that squash-merged four feature sub-branches:
- Core generic tools (#28): 6 tools, APIFY_TOKEN env-var rename
- LangChain-native components (#29): ApifySearchRetriever and ApifyCrawlLoader
- Search & crawling tools (#31): 6 tools
- Social-media tools (#30): 7 social tools

Plus umbrella-level changes:
- Package split: collapsed _actor_tools.py + tools.py into a tools/ package
- Single-source constants: shared _constants.py, _types.py, _error_messages.py
- Schema-level clamp documentation with new tests
- Bug fixes (integration tests, apify_token forwarding, _prune_actor_input_schema falsy defaults)
- Minor perf fix in ApifyActorsTool.__init__
- ApifyDatasetLoader.apify_client changed to PrivateAttr
- README rewrite, DEVELOPMENT.md, CONTRIBUTING.md (new)

PR URL: https://github.com/apify/langchain-apify/pull/35
Branch: feat/modernize-langchain-integration
Base: main
