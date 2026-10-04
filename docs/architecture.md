# Architecture decisions

## Access before retrieval

Filter at the beginning so inaccessible text is not embedded, ranked, or included in a model prompt. Filtering only the final answer would leak information upstream.

## Transparent lexical baseline

BM25 works without a model download. Dense embeddings are an explicit option, allowing lexical-only and fused retrieval to be compared on the same question set.

## Evidence IDs are necessary but insufficient

The validator catches fabricated IDs. A citation-aware entailment evaluation is still needed to measure whether each statement follows from its evidence.

## Next engineering step

Persist indexed embeddings with source revisions and ACL versions; add an independent held-out question set, reranker comparisons and claim-support evaluation.
