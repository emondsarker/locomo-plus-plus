"""
Step 4: Semantic filtering — remove trigger-cue pairs with high surface similarity
to ensure models can't shortcut via lexical/semantic retrieval.

Uses BM25 (lexical) and MPNet (neural) similarity scoring.

Usage:
    python scripts/04_semantic_filter.py \
        --cues data/cues/cues.json \
        --triggers data/triggers/triggers.json \
        --output data/filtered/
"""

import argparse
import numpy as np
from utils import load_json, save_json

# Thresholds — pairs above either threshold are removed
DEFAULT_BM25_THRESHOLD = 15.0
DEFAULT_COSINE_THRESHOLD = 0.65


def compute_bm25_scores(trigger_texts, cue_texts):
    """Compute BM25 scores between each trigger and its related cues."""
    from rank_bm25 import BM25Okapi

    tokenized_cues = [text.lower().split() for text in cue_texts]
    bm25 = BM25Okapi(tokenized_cues)

    scores = []
    for trigger_text in trigger_texts:
        tokenized_query = trigger_text.lower().split()
        doc_scores = bm25.get_scores(tokenized_query)
        scores.append(float(np.max(doc_scores)))
    return scores


def compute_mpnet_scores(trigger_texts, cue_texts):
    """Compute cosine similarity between triggers and cues using MPNet."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-mpnet-base-v2")

    trigger_embs = model.encode(trigger_texts, normalize_embeddings=True)
    cue_embs = model.encode(cue_texts, normalize_embeddings=True)

    # For each trigger, compute max cosine similarity against all cues
    # (we only care about its related cues, but computing against all is simpler
    #  and more conservative)
    sim_matrix = trigger_embs @ cue_embs.T  # (num_triggers, num_cues)
    max_sims = np.max(sim_matrix, axis=1)
    return max_sims.tolist()


def compute_per_trigger_scores(triggers, cues):
    """Compute similarity scores for each trigger against its related cues only."""
    from sentence_transformers import SentenceTransformer
    from rank_bm25 import BM25Okapi

    # Build cue index
    cue_by_id = {c["cue_id"]: c for c in cues}

    # Load MPNet
    print("Loading MPNet model...")
    mpnet = SentenceTransformer("all-mpnet-base-v2")

    results = []
    for trigger in triggers:
        related_cue_ids = trigger.get("related_cue_ids", [])
        related_cues = [cue_by_id[cid] for cid in related_cue_ids if cid in cue_by_id]

        if not related_cues:
            results.append({"bm25": 0.0, "cosine": 0.0})
            continue

        trigger_text = trigger["trigger_text"]
        cue_texts = [c["dialogue"] for c in related_cues]

        # BM25
        tokenized_cues = [text.lower().split() for text in cue_texts]
        bm25 = BM25Okapi(tokenized_cues)
        bm25_scores = bm25.get_scores(trigger_text.lower().split())
        max_bm25 = float(np.max(bm25_scores))

        # MPNet cosine
        trigger_emb = mpnet.encode([trigger_text], normalize_embeddings=True)
        cue_embs = mpnet.encode(cue_texts, normalize_embeddings=True)
        cosine_scores = (trigger_emb @ cue_embs.T)[0]
        max_cosine = float(np.max(cosine_scores))

        results.append({"bm25": max_bm25, "cosine": max_cosine})

    return results


def main():
    parser = argparse.ArgumentParser(description="Semantic filtering of cue-trigger pairs")
    parser.add_argument("--cues", type=str, default="data/cues/cues.json")
    parser.add_argument("--triggers", type=str, default="data/triggers/triggers.json")
    parser.add_argument("--output", type=str, default="data/filtered/")
    parser.add_argument("--bm25-threshold", type=float, default=DEFAULT_BM25_THRESHOLD)
    parser.add_argument("--cosine-threshold", type=float, default=DEFAULT_COSINE_THRESHOLD)
    args = parser.parse_args()

    cues = load_json(args.cues)
    triggers = load_json(args.triggers)
    print(f"Loaded {len(cues)} cues, {len(triggers)} triggers")

    print("Computing similarity scores...")
    scores = compute_per_trigger_scores(triggers, cues)

    # Filter
    kept = []
    removed = []
    for trigger, score in zip(triggers, scores):
        trigger["_bm25_score"] = round(score["bm25"], 3)
        trigger["_cosine_score"] = round(score["cosine"], 3)

        if score["bm25"] > args.bm25_threshold or score["cosine"] > args.cosine_threshold:
            removed.append(trigger)
        else:
            kept.append(trigger)

    print(f"\nFiltering results:")
    print(f"  Kept:    {len(kept)} triggers")
    print(f"  Removed: {len(removed)} triggers")
    print(f"  BM25 threshold: {args.bm25_threshold}")
    print(f"  Cosine threshold: {args.cosine_threshold}")

    if removed:
        avg_bm25 = np.mean([t["_bm25_score"] for t in removed])
        avg_cos = np.mean([t["_cosine_score"] for t in removed])
        print(f"  Removed avg BM25: {avg_bm25:.3f}, avg cosine: {avg_cos:.3f}")

    # Save filtered triggers and the related cues
    filtered_cue_ids = set()
    for t in kept:
        filtered_cue_ids.update(t.get("related_cue_ids", []))
    filtered_cues = [c for c in cues if c["cue_id"] in filtered_cue_ids]

    save_json(kept, f"{args.output}/triggers_filtered.json")
    save_json(filtered_cues, f"{args.output}/cues_filtered.json")
    save_json(removed, f"{args.output}/triggers_removed.json")

    print(f"\nSaved to {args.output}/")
    print(f"  triggers_filtered.json: {len(kept)} triggers")
    print(f"  cues_filtered.json: {len(filtered_cues)} cues")
    print(f"  triggers_removed.json: {len(removed)} removed (for inspection)")


if __name__ == "__main__":
    main()
