# Cutting Embeddings to a Quarter: How Matryoshka Representations Made My Search System Cheaper to Run

*I dropped 75% of every vector in a procurement search system. Here's why the cost fell sharply, why I expect relevance to hold up, and how I'm checking.*

--

When you start building semantic search, embedding dimensions seem like a detail. You pick a model, use the size it gives you, and move on. That changes once the corpus grows. Every document carries a vector, every vector takes memory, and in most vector databases memory is what you pay for.

I learned this building a procurement search system: a tender search platform that indexes government tender notices from many countries. My test set holds about 88,000 notices. The live system will take in roughly 85,000–100,000 new notices a day, so it will reach millions of documents quickly. At that scale, the size of each vector stops being a detail and becomes a big part of the hosting bill.

This article covers how I used **Matryoshka Representation Learning (MRL)** to shrink my embeddings from 4,096 dimensions to 1,024, why the search architecture around it matters as much as the technique, and how I'm measuring whether it actually holds up.

## The problem: storage grows with documents × dimensions

A dense embedding is a list of floating-point numbers. Stored as float32, each number takes 4 bytes. So the raw size of your vector index is roughly:

```
documents × dimensions × 4 bytes
```

That figure grows linearly on both axes. Doubling the corpus doubles it. Doubling the dimensions doubles it again. My embedding model, Qwen3-Embedding-8B, produces 4,096-dimensional vectors by default. That's a high-quality representation, but a large one: each tender notice costs 16 KB of vector data before you add any index structures.

For a few thousand documents, nobody cares. For millions, those 16 KB per document set what size of machine you need.

## What MRL actually is

Matryoshka Representation Learning was introduced by Kusupati et al. in the 2022 paper *"Matryoshka Representation Learning"* (NeurIPS 2022). The name comes from Russian nesting dolls, and the idea fits the image well.

A normal embedding model spreads information across all its dimensions in no particular order. If you chop off the second half of the vector, you lose information at random, and quality usually drops badly.

An MRL-trained model is trained differently. During training, the loss is calculated not only on the full vector but also on several prefixes of it, for example the first 64, 128, 256, 512 dimensions and so on. The model learns to pack the most important information into the earliest dimensions. The result is that **the first N values of the vector are already a usable embedding on their own.**

```
Full vector (4,096 dims)
┌────┬────┬────────┬────────────────┬──────────────────────────────────────┐
│256 │512 │ 1,024  │                │                                      │
└────┴────┴────────┴────────────────┴──────────────────────────────────────┘
 ◄─256─►
 ◄───512───►
 ◄──────1,024──────►
 ◄─────────────────────────────── 4,096 ──────────────────────────────────►

Each prefix is a smaller, self-contained embedding nested inside the larger one.
```

> **Diagram suggestion:** draw a single horizontal bar with nested, progressively lighter segments labelled 256 / 512 / 1,024 / 4,096, with the 1,024 segment highlighted as "what I store."

To use a smaller size, you take the first N values and **re-normalize** the vector to unit length, so cosine similarity and dot product still behave correctly. Many embedding APIs do this for you through a `dimensions` parameter. The API I use returns the first 1,024 values, re-normalized.

Two caveats are worth stating clearly:

- **It only works well for models trained this way.** Truncating a model that wasn't trained with MRL loses information unpredictably. Check the model card before you rely on it.
- **It isn't the only way to shrink vectors.** *PCA* (principal component analysis) can reduce dimensions after the fact, but it's an extra fitted step that has to be maintained as your data changes, and it isn't free in quality either. *Quantization* (storing each number as int8 or even a single bit instead of float32) cuts the bytes per dimension rather than the number of dimensions. Quantization and MRL work on different axes, so they can be combined.

The paper reports that MRL embeddings can be much smaller than full-size ones while staying close in accuracy on the tasks it evaluated. How close depends on the model, the task and how far you cut, which is why I treat it as something to measure, not assume.

## How I used it

The change itself was a single API parameter: I request **1,024 dimensions instead of the native 4,096.** No retraining, no extra pipeline step, no PCA model to keep updated.

The storage effect, for raw float32 vectors:

| Corpus size | 4,096 dims | 1,024 dims | Saving |
|---|---|---|---|
| 88K tenders (test set) | ~1.45 GB | ~362 MB | 4× |
| 1M tenders | ~16 GB | ~4 GB | 4× |
| 5M tenders | ~82 GB | ~20 GB | 4× |

These are raw vector sizes. Real indexes such as HNSW add graph overhead on top, but that overhead also shrinks or stays flat as dimensions drop, so the ratio still points the right way. Because vector database pricing is driven largely by memory, a 4× smaller footprint means a smaller hosting tier. Shorter vectors also make each distance calculation cheaper, which speeds up search.

### Why the quality risk is smaller in my design

The more important point is architectural. In my system, the embedding doesn't decide the final ranking. The search pipeline is hybrid and **filter-first**:

1. **Hard filters run first:** country, publication dates, contract value, notice type. This shrinks the search space before any similarity scoring.
2. **Keyword search (BM25) and vector search run inside the filtered set**, and their results are blended.
3. **A reranker re-scores the top 50 candidates**, reading the full text of each notice, and decides the final order.

That changes what truncation can actually damage. The vectors' job is to get relevant tenders *into the top 50*. They don't decide the order within it. So the realistic failure mode of shorter vectors is a relevant notice falling just outside the candidate set, not a worse ordering of results the user sees.

The keyword half of the hybrid search also limits the damage. Tender searches are full of exact terms: CPV codes, product names, specific services. BM25 catches many of those matches regardless of what the vectors do. The vectors mainly add value on paraphrase and concept-level matches, and those are exactly the cases where the early, most important dimensions carry the signal.

**[RESULT: candidate recall@50 at 1,024 vs 4,096 dims]**
**[RESULT: top-10 relevance at 1,024 vs 4,096 dims]**

### Does it hold across models?

One model working well under truncation doesn't prove the approach is reliable. So I'm also testing a second open embedding model, F2LLM-v2-4B, which produces 2,560-dimensional vectors natively, at full size and truncated to 1,024. If the gap is small for both, I can be more confident that the choice of 1,024 isn't just luck with one model.

**[RESULT: F2LLM-v2-4B 2,560 vs 1,024 dims — recall@50 and top-10 relevance]**

## Trade-offs, and when not to cut aggressively

MRL isn't free compression. A few cases where I'd be more careful:

- **Very small dimensions.** Going from 4,096 to 1,024 keeps a lot of capacity. Going to 64 or 128 is a different decision. Quality tends to drop faster at the small end, and fine-grained distinctions such as similar tenders in neighbouring categories are usually the first to blur.
- **Models not trained for MRL.** If the model card doesn't say it supports flexible dimensions, don't truncate. Use quantization or PCA instead, and measure.
- **Systems without a reranker.** If your vector similarity *is* your final ranking, every bit of lost precision shows up directly in what users see. My design tolerates truncation largely because the reranker gets the final say.
- **Pure semantic search with no keyword fallback.** Without BM25 alongside, there's nothing to catch the exact-match cases the shorter vectors might miss.
- **Changing your mind later costs a re-embed.** Vectors at different sizes aren't comparable. Moving from 1,024 to 2,048 later means re-embedding the corpus, which matters at millions of documents. If you keep the full vectors in cheap cold storage, you can re-truncate without calling the model again.

## How to evaluate it yourself

Don't trust a general benchmark for your specific corpus. A quick, honest evaluation looks like this:

1. **Collect real queries.** A few dozen to a few hundred, taken from logs or written by people who know the domain. Include short keyword queries and longer, descriptive ones.
2. **Run the full pipeline twice**, once at full dimensions and once truncated. Keep everything else identical: filters, BM25 weights, reranker.
3. **Measure candidate recall.** For each query, how much of the full-dimension top 50 also appears in the truncated top 50? This isolates the step truncation actually touches.
4. **Measure final relevance.** Grade the top 10 results from each run, ideally by people who know the domain, or by an LLM judge with a clear rubric that you've spot-checked against human grades.
5. **Look at the disagreements**, not just the averages. The queries where the two runs differ will tell you *what kind* of relevance you're losing, which matters more than a single score.

If candidate recall stays high and final top-10 relevance barely moves, you have your answer. If it does move, try the next size up before giving up on the idea.

## Takeaways

- **Vector storage scales with documents × dimensions**, and at millions of documents it becomes a real share of your infrastructure cost.
- **MRL lets you use a prefix of the embedding as a smaller embedding**, as long as the model was trained for it. In my case it was one API parameter and a 4× cut in raw vector storage.
- **Architecture decides how much truncation hurts.** Filters, hybrid keyword search and a reranker mean the vectors only need to get good candidates into the top 50, not rank them perfectly.
- **Measure on your own data.** Candidate recall and top-10 relevance at full vs truncated dimensions will tell you more than any published benchmark.

I'll update this article with the measured results once the comparison runs are complete.

---

### Reference

Kusupati, A., Bhatt, G., Rege, A., Wallingford, M., Sinha, A., Ramanujan, V., Howard-Snyder, W., Chen, K., Kakade, S., Jain, P., & Farhadi, A. (2022). *Matryoshka Representation Learning.* Advances in Neural Information Processing Systems (NeurIPS 2022). arXiv:2205.13147.

---

### Alternative titles

1. **The Nesting-Doll Trick: Storing a Quarter of Every Embedding Without Losing the Plot**
2. **4,096 Dimensions Was Too Many: Matryoshka Embeddings in a Production Procurement Search**
3. **Smaller Vectors, Same Search: Using MRL When Your Corpus Grows by 100K Documents a Day**
