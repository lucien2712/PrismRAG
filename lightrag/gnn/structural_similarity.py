"""
FastRP Embedding Similarity Calculator
Uses precomputed FastRP embeddings for true structural similarity computation
"""

import numpy as np
from typing import Dict, Iterable, List, Optional
from ..utils import logger


def compute_adaptive_fastrp_similarity(
    target_entity: str,
    seed_entities: List[str],
    node_embedding
) -> float:
    """
    Content-aware FastRP similarity calculation using precomputed 128-dimensional embeddings
    Computes cosine similarity between FastRP embedding vectors

    Args:
        target_entity: Target entity name
        seed_entities: List of seed entity names
        node_embedding: NodeEmbeddingEnhancer instance with fastrp_embeddings

    Returns:
        Average cosine similarity between target and seed entity embeddings
    """
    try:
        if not node_embedding or not hasattr(node_embedding, 'fastrp_embeddings'):
            logger.warning("FastRP embeddings not available")
            return 0.0

        fastrp_embeddings = node_embedding.fastrp_embeddings
        if not fastrp_embeddings:
            logger.warning("FastRP embeddings dictionary is empty")
            return 0.0

        # Get target entity embedding
        if target_entity not in fastrp_embeddings:
            logger.debug(f"Target entity '{target_entity}' not found in FastRP embeddings")
            return 0.0

        target_embedding = fastrp_embeddings[target_entity]
        similarities = []

        for seed_entity in seed_entities:
            if seed_entity not in fastrp_embeddings:
                logger.debug(f"Seed entity '{seed_entity}' not found in FastRP embeddings")
                continue

            seed_embedding = fastrp_embeddings[seed_entity]

            # Compute cosine similarity between embeddings
            similarity = cosine_similarity(target_embedding, seed_embedding)
            similarities.append(similarity)

            logger.debug(f"FastRP similarity between '{target_entity}' and '{seed_entity}': {similarity:.4f}")

        result = float(np.mean(similarities)) if similarities else 0.0
        logger.debug(f"Average FastRP similarity for '{target_entity}': {result:.4f} (from {len(similarities)} seed entities)")

        return result

    except Exception as e:
        logger.error(f"Error computing FastRP embedding similarity: {e}")
        return 0.0


def compute_adaptive_fastrp_batch(
    seed_entities: List[str],
    node_embedding,
    candidate_entities: Optional[Iterable[str]] = None,
) -> Dict[str, float]:
    """
    Vectorized Content-aware FastRP similarity calculation for multiple candidates.

    Args:
        seed_entities: Seed entity names
        node_embedding: NodeEmbeddingEnhancer instance with fastrp_embeddings
        candidate_entities: Optional iterable of candidate entity names.
            If None, use all entities except the seeds.

    Returns:
        Dictionary mapping candidate entity -> similarity score
    """
    try:
        if not node_embedding or not hasattr(node_embedding, "fastrp_embeddings"):
            logger.warning("FastRP embeddings not available for batch computation")
            return {}

        fastrp_embeddings = node_embedding.fastrp_embeddings or {}
        if not fastrp_embeddings:
            logger.warning("FastRP embeddings dictionary is empty for batch computation")
            return {}

        seed_entities = [seed for seed in seed_entities if seed in fastrp_embeddings]
        if not seed_entities:
            logger.debug("No valid seed entities found in FastRP embeddings")
            return {}

        if candidate_entities is None:
            candidate_entities = [
                name for name in fastrp_embeddings.keys() if name not in seed_entities
            ]
        else:
            candidate_entities = [
                name for name in candidate_entities
                if name not in seed_entities and name in fastrp_embeddings
            ]

        if not candidate_entities:
            logger.debug("No candidate entities available for FastRP batch computation")
            return {}

        # Normalize seed embeddings once
        seed_vectors = []
        for seed_entity in seed_entities:
            seed_vec = np.asarray(fastrp_embeddings[seed_entity])
            seed_norm = np.linalg.norm(seed_vec)
            if seed_norm == 0:
                continue
            seed_vectors.append(seed_vec / seed_norm)

        if not seed_vectors:
            logger.debug("All seed embeddings have zero norm; returning empty FastRP batch result")
            return {}

        seed_matrix = np.vstack(seed_vectors)

        candidate_vectors = []
        candidate_names = []
        zero_score_entities = []

        for entity_name in candidate_entities:
            candidate_vec = np.asarray(fastrp_embeddings[entity_name])
            candidate_norm = np.linalg.norm(candidate_vec)
            if candidate_norm == 0:
                zero_score_entities.append(entity_name)
                continue

            candidate_vectors.append(candidate_vec / candidate_norm)
            candidate_names.append(entity_name)

        scores: Dict[str, float] = {}

        if candidate_vectors:
            candidate_matrix = np.vstack(candidate_vectors)
            similarity_matrix = candidate_matrix @ seed_matrix.T
            mean_scores = similarity_matrix.mean(axis=1)

            logger.info(f"FastRP: Computed similarities for {len(candidate_names)} candidates")

            for entity_name, score in zip(candidate_names, mean_scores):
                scores[entity_name] = float(score)

        for entity_name in zero_score_entities:
            scores[entity_name] = 0.0

        return scores

    except Exception as e:
        logger.debug(f"Vectorized FastRP computation failed, falling back to sequential mode: {e}")

        # Fallback to sequential calculation
        scores: Dict[str, float] = {}
        for candidate in candidate_entities or []:
            score = compute_adaptive_fastrp_similarity(candidate, seed_entities, node_embedding)
            scores[candidate] = score
        return scores


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Cosine similarity score between 0 and 1
    """
    try:
        # Normalize vectors to unit length
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        # Compute cosine similarity
        similarity = np.dot(vec1, vec2) / (norm1 * norm2)
        return float(similarity)  

    except Exception as e:
        logger.error(f"Error computing cosine similarity: {e}")
        return 0.0
