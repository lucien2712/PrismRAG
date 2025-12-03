from __future__ import annotations

"""Node embedding enhancer using FastRP embeddings and query-aware Personalized PageRank."""

import numpy as np
import networkx as nx
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass, field
import os
import pickle
import json
from scipy import sparse

from ..utils import cosine_similarity
from .structural_similarity import compute_adaptive_fastrp_batch

logger = logging.getLogger(__name__)


def resolve_positive_int(value: int | str | None, default: int) -> int:
    """Return a positive integer from value or fallback to default."""
    try:
        candidate = int(value)  # type: ignore[arg-type]
        if candidate > 0:
            return candidate
    except (TypeError, ValueError):
        pass
    return default


async def compute_embeddings_in_batches(
    items: List[str],
    embedding_func,
    batch_size: int,
) -> List:
    """Compute embeddings for the given items in batches while preserving order."""
    if batch_size <= 0:
        batch_size = 1

    embeddings: List = []
    for start in range(0, len(items), batch_size):
        batch = items[start:start + batch_size]
        if not batch:
            continue
        batch_embeddings = await embedding_func.func(batch)

        # Handle different return types
        if isinstance(batch_embeddings, np.ndarray):
            # If numpy array, convert to list of individual embeddings
            if batch_embeddings.ndim == 2:
                # Shape: (batch_size, embedding_dim) -> list of (embedding_dim,)
                embeddings.extend([emb for emb in batch_embeddings])
            elif batch_embeddings.ndim == 1:
                # Single embedding
                embeddings.append(batch_embeddings)
            else:
                # Unexpected shape
                embeddings.extend(batch_embeddings)
        elif isinstance(batch_embeddings, list):
            embeddings.extend(batch_embeddings)
        else:
            embeddings.append(batch_embeddings)
    return embeddings


@dataclass
class NodeEmbeddingConfig:
    """Configuration for FastRP embeddings and query-aware Personalized PageRank."""

    # FastRP parameters
    embedding_dimension: int = 256
    normalization_strength: float = -0.1
    iteration_weights: List[float] = field(default_factory=lambda: [1.0, 1.0, 0.5, 0.25])
    random_seed: Optional[int] = 42

    # Personalized PageRank parameters (used at query time)
    pagerank_alpha: float = 0.85  # Damping parameter
    pagerank_max_iter: int = 1000
    pagerank_tol: float = 1e-03


class NodeEmbeddingEnhancer:
    """Enhances entity retrieval using FastRP embeddings and query-aware Personalized PageRank."""
    
    def __init__(self, config: NodeEmbeddingConfig, working_dir: str):
        self.config = config
        self.working_dir = working_dir

        self.fastrp_embeddings: Optional[Dict[str, np.ndarray]] = None
        self.graph: Optional[nx.Graph] = None
        self._relations_cache: Optional[List[Dict]] = None  # Cache for all relations
        self._relation_embeddings_cache: Optional[Dict[str, np.ndarray]] = None  # Cache for relation embeddings
        self._entity_embeddings_cache: Optional[Dict[str, np.ndarray]] = None  # Cache for entity embeddings (entity_name + description)

        # File paths for persistence
        self.graph_path = os.path.join(working_dir, "node_embedding_graph.pkl")
        self.fastrp_path = os.path.join(working_dir, "fastrp_embeddings.pkl")
        self.relations_cache_path = os.path.join(working_dir, "relations_cache.pkl")
        self.relation_embeddings_cache_path = os.path.join(working_dir, "relation_embeddings_cache.pkl")
        self.entity_embeddings_cache_path = os.path.join(working_dir, "entity_embeddings_cache.pkl")

        # Try to load existing data
        self._load_persisted_data()
        
    async def compute_node_embeddings(
        self, 
        entities: List[Dict],
        relations: List[Dict],
        text_embeddings: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Compute enhanced embeddings during document insertion.
        
        Args:
            entities: List of entity dictionaries
            relations: List of relation dictionaries  
            text_embeddings: Dict mapping entity_id -> text_embedding
            
        Returns:
            Dict mapping entity_id -> enhanced_embedding
        """
        if not entities or not relations:
            logger.warning("No entities or relations to process for node embedding")
            return {}
            
        logger.info(f"Computing enhanced embeddings for {len(entities)} entities, {len(relations)} relations")
        
        # 1. Build/update NetworkX graph
        new_graph = self._build_networkx_graph(entities, relations)
        
        # Merge with existing graph if present
        if self.graph is not None:
            # Merge nodes and edges from new graph
            self.graph.add_nodes_from(new_graph.nodes(data=True))
            self.graph.add_edges_from(new_graph.edges(data=True))
            logger.info(f"Updated existing graph: now has {len(self.graph.nodes())} nodes, {len(self.graph.edges())} edges")
        else:
            self.graph = new_graph
        
        if len(self.graph.nodes()) == 0:
            logger.warning("Empty graph, skipping node embedding computation")
            return {}
            
        # 2. Compute FastRP embeddings
        self.fastrp_embeddings = await self._compute_fastrp_embeddings()

        # 3. Save computed data to disk for query-time reuse
        self._save_persisted_data()

        logger.info(f"FastRP embeddings computed for {len(self.fastrp_embeddings)} entities")
        # Note: FastRP embeddings are stored separately for query-time similarity computation
        # Personalized PageRank is computed dynamically at query time with query-aware weighting
        return {}
        
    def _build_networkx_graph(self, entities: List[Dict], relations: List[Dict]) -> nx.Graph:
        """Build NetworkX graph from entities and relations."""
        graph = nx.Graph()
        
        # Add nodes
        for entity in entities:
            entity_id = str(entity.get('entity_name', entity.get('id', '')))
            if entity_id:
                graph.add_node(entity_id)
                
        # Add edges
        for relation in relations:
            src = str(relation.get('src_id', ''))
            tgt = str(relation.get('tgt_id', ''))
            
            if src and tgt and src in graph.nodes() and tgt in graph.nodes():
                # Use weight if available, otherwise default to 1.0
                weight = relation.get('weight', 1.0)
                keywords = relation.get('keywords', '')
                graph.add_edge(src, tgt, weight=weight, keywords=keywords)
                
        logger.info(f"Built graph with {len(graph.nodes())} nodes, {len(graph.edges())} edges")
        return graph
        
    async def _compute_fastrp_embeddings(self) -> Dict[str, np.ndarray]:
        """Compute FastRP embeddings using proper degree normalization and self-loops."""
        if len(self.graph.nodes()) < 2:
            logger.warning("Graph too small for FastRP, returning empty embeddings")
            return {}
            
        try:
            # Set random seed for reproducibility
            if self.config.random_seed is not None:
                np.random.seed(self.config.random_seed)
            
            # Convert NetworkX graph to adjacency matrix
            node_list = list(self.graph.nodes())
            n_nodes = len(node_list)
            embedding_dim = self.config.embedding_dimension

            # Create adjacency matrix with edges and self-loops
            A = nx.adjacency_matrix(self.graph, nodelist=node_list, weight="weight").astype(np.float32)
            A = A + sparse.eye(A.shape[0], dtype=np.float32)  # Add self-loops
            
            # Degree normalization
            deg = np.asarray(A.sum(axis=1)).ravel()
            deg[deg == 0] = 1.0  # Avoid division by zero
            
            # Create normalized adjacency matrix S = D^{-1/2} (A+I) D^{-1/2}
            D_inv_sqrt = sparse.diags(1.0 / np.sqrt(deg))
            S = D_inv_sqrt @ A @ D_inv_sqrt
            
            # Degree-based normalization strength r
            r = self.config.normalization_strength  # e.g., -0.1

            # Initialize random projection matrix R (n x d)
            R = np.random.choice([-1.0, 1.0], size=(n_nodes, embedding_dim)).astype(np.float32)
            # Row normalize R
            R_norms = np.linalg.norm(R, axis=1, keepdims=True)
            R_norms = np.where(R_norms > 1e-9, R_norms, 1e-9)
            R = R / R_norms

            # FastRP multi-order aggregation: X = Σ(k=0 to K) w_k * D^r * S^k * R
            X = np.zeros((n_nodes, embedding_dim), dtype=np.float32)
            Z = R.copy()

            for k, weight in enumerate(self.config.iteration_weights):
                # Apply single-sided degree normalization: D^r * Z
                term = (deg ** r)[:, np.newaxis] * Z

                # Accumulate weighted term
                X += weight * term

                # Propagate for next iteration: Z = S @ Z
                if k < len(self.config.iteration_weights) - 1:  # Don't compute for last iteration
                    Z = S @ Z
                    if sparse.issparse(Z):
                        Z = Z.toarray()
            
            # Optional: final row normalization for stability
            final_norms = np.linalg.norm(X, axis=1, keepdims=True)
            final_norms = np.where(final_norms > 1e-9, final_norms, 1e-9)
            X = X / final_norms
            
            # Convert back to dictionary format
            embeddings = {}
            for idx, node in enumerate(node_list):
                embeddings[str(node)] = X[idx]
                    
            logger.info(f"FastRP embeddings computed for {len(embeddings)} nodes")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error computing FastRP embeddings: {e}")
            return {}
                   
    async def get_all_relations_cached(self, knowledge_graph_inst) -> List[Dict]:
        """Get all relations with caching to avoid repeated full-graph queries."""
        if self._relations_cache is None:
            logger.debug("Relations cache miss, fetching from knowledge graph")
            self._relations_cache = await knowledge_graph_inst.get_all_edges()
            # Optionally persist to disk with gzip compression
            try:
                import gzip
                os.makedirs(self.working_dir, exist_ok=True)
                cache_path_gz = self.relations_cache_path + '.gz'
                with gzip.open(cache_path_gz, 'wb', compresslevel=1) as f:
                    pickle.dump(self._relations_cache, f, protocol=4)
                logger.debug(f"Saved {len(self._relations_cache)} relations to compressed cache")
            except Exception as e:
                logger.debug(f"Failed to save relations cache: {e}")
        return self._relations_cache

    def invalidate_relations_cache(self):
        """Invalidate all caches when graph structure changes."""
        self._relations_cache = None
        self._relation_embeddings_cache = None
        self._entity_embeddings_cache = None

        # Remove both .pkl and .gz versions for backward compatibility
        for cache_path in [self.relations_cache_path, self.relations_cache_path + '.gz']:
            if os.path.exists(cache_path):
                try:
                    os.remove(cache_path)
                    logger.debug(f"Relations cache invalidated: {cache_path}")
                except Exception as e:
                    logger.debug(f"Failed to remove relations cache file: {e}")
        if os.path.exists(self.relation_embeddings_cache_path):
            try:
                os.remove(self.relation_embeddings_cache_path)
                logger.debug("Relation embeddings cache invalidated")
            except Exception as e:
                logger.debug(f"Failed to remove relation embeddings cache file: {e}")
        if os.path.exists(self.entity_embeddings_cache_path):
            try:
                os.remove(self.entity_embeddings_cache_path)
                logger.debug("Entity embeddings cache invalidated")
            except Exception as e:
                logger.debug(f"Failed to remove entity embeddings cache file: {e}")

        logger.debug("All caches invalidated")

    async def build_relations_cache(self, knowledge_graph_inst):
        """
        Pre-build relation cache during insert to speed up first query.

        This method fetches all relations from the knowledge graph and saves them
        to disk with gzip compression. By building the cache during insert rather
        than on first query, we eliminate the 30-60 second delay users would
        otherwise experience on their first query.

        Args:
            knowledge_graph_inst: Knowledge graph storage instance to fetch relations from
        """
        logger.info("Building relation cache after insert to speed up first query...")
        try:
            await self.get_all_relations_cached(knowledge_graph_inst)
            logger.info(f"Relation cache built successfully with {len(self._relations_cache)} relations")
        except Exception as e:
            logger.warning(f"Failed to build relation cache during insert: {e}. Cache will be built on first query.")
            # Non-fatal: cache will be built on first query if this fails

    async def _get_relation_embeddings_from_vdb(
        self,
        knowledge_graph_inst,
        global_config: Dict[str, Any]
    ) -> Dict[str, np.ndarray]:
        """
        Get relation embeddings directly from Vector Storage (same format as insert).
        This reuses embeddings computed during insert, avoiding redundant API calls.

        Args:
            knowledge_graph_inst: Knowledge graph storage instance
            global_config: Global configuration with relationships_vdb

        Returns:
            Dict mapping relation_keywords -> embedding vector
        """
        try:
            relationships_vdb = global_config.get("relationships_vdb")
            if not relationships_vdb or not hasattr(relationships_vdb, '_get_client'):
                logger.debug("Relationship vector storage doesn't support direct vector access")
                return {}

            # Get all relations
            all_relations = await self.get_all_relations_cached(knowledge_graph_inst)
            if not all_relations:
                logger.debug("No relations found from knowledge graph")
                return {}

            logger.debug(f"Retrieved {len(all_relations)} relations from cache")

            # Generate vector database IDs for all relationships
            from ..utils import compute_mdhash_id
            rel_vdb_ids = []

            for relation in all_relations:
                # Try different field names (NetworkX uses 'source'/'target', others may use 'source_id'/'target_id')
                src_id = relation.get('source') or relation.get('source_id', '')
                tgt_id = relation.get('target') or relation.get('target_id', '')
                keywords = relation.get('keywords', '')

                if src_id and tgt_id and keywords.strip():
                    # IMPORTANT: Must match insert format - no separator between src and tgt
                    # For undirected graphs, try BOTH directions (src+tgt and tgt+src)
                    rel_key_1 = src_id + tgt_id
                    rel_key_2 = tgt_id + src_id
                    vdb_id_1 = compute_mdhash_id(rel_key_1, prefix="rel-")
                    vdb_id_2 = compute_mdhash_id(rel_key_2, prefix="rel-")
                    # Add both possible IDs (VDB lookup will filter to existing ones)
                    rel_vdb_ids.append(vdb_id_1)
                    if vdb_id_2 != vdb_id_1:  # Avoid duplicates
                        rel_vdb_ids.append(vdb_id_2)

            if not rel_vdb_ids:
                logger.debug("No valid relation IDs to lookup in VDB")
                return {}

            # Direct access to NanoVectorDB storage to bypass client.get() limitation
            # NanoVectorDB's client.get() has a bug that only returns ~139 results
            # So we access the underlying storage directly
            client = await relationships_vdb._get_client()
            storage = getattr(client, '_NanoVectorDB__storage', None)

            if storage is None:
                # Fallback to client.get() if storage access fails
                logger.warning("Cannot access NanoVectorDB storage directly, using client.get()")
                vdb_results = client.get(rel_vdb_ids)
            else:
                # Build ID lookup map from storage
                storage_map = {item['__id__']: item for item in storage.get('data', [])}
                # Get results for requested IDs
                vdb_results = [storage_map.get(vdb_id) for vdb_id in rel_vdb_ids]

            relation_vectors = {}
            missing_count = 0
            none_count = 0
            no_vector_count = 0
            no_id_count = 0
            decode_error_count = 0

            for vdb_data in vdb_results:
                if vdb_data is None:
                    none_count += 1
                    missing_count += 1
                    continue

                if 'vector' not in vdb_data:
                    no_vector_count += 1
                    missing_count += 1
                    continue

                # Get relation info directly from VDB data
                src_id = vdb_data.get('src_id')
                tgt_id = vdb_data.get('tgt_id')
                keywords = vdb_data.get('keywords', '')

                if not src_id or not tgt_id or not keywords:
                    no_id_count += 1
                    missing_count += 1
                    continue

                # For undirected graphs, normalize to (src, tgt) tuple using lexicographic order
                # This ensures consistent key regardless of edge direction in graph
                if src_id > tgt_id:
                    edge_key = (tgt_id, src_id)
                else:
                    edge_key = (src_id, tgt_id)
                try:
                    # Decode NanoVectorDB format: base64 + zlib + float16
                    import base64
                    import zlib

                    compressed_vector = base64.b64decode(vdb_data['vector'])
                    vector_bytes = zlib.decompress(compressed_vector)
                    vector_array = np.frombuffer(vector_bytes, dtype=np.float16).astype(np.float32)

                    # Use normalized edge_key to handle undirected graphs
                    # This prevents different edges with same keywords from being overwritten
                    relation_vectors[edge_key] = {
                        'vector': vector_array,
                        'keywords': keywords
                    }
                except Exception as e:
                    logger.debug(f"Error decoding vector for relation '{keywords}': {e}")
                    decode_error_count += 1
                    missing_count += 1

            logger.info(f"Retrieved {len(relation_vectors)} relation embeddings from Vector Storage")
            return relation_vectors

        except Exception as e:
            logger.debug(f"Error getting relation embeddings from VDB: {e}")
            return {}

    async def get_relation_embeddings_cached(
        self,
        knowledge_graph_inst,
        embedding_func
    ) -> Dict[str, np.ndarray]:
        """
        Get relation embeddings with caching.

        Returns:
            Dict mapping relation_keywords -> embedding vector
        """
        if self._relation_embeddings_cache is not None:
            logger.debug(f"Relation embeddings cache hit: {len(self._relation_embeddings_cache)} relations")
            return self._relation_embeddings_cache

        logger.debug("Relation embeddings cache miss, computing embeddings...")

        # Get all relations
        all_relations = await self.get_all_relations_cached(knowledge_graph_inst)

        # Prepare relation strings
        rel_strs: List[str] = []
        rel_keywords_order: List[str] = []

        for relation in all_relations:
            rel_keywords = relation.get("keywords", "")
            if rel_keywords.strip():
                rel_str = (
                    f"{rel_keywords}\t{relation.get('source_id', '')}\n"
                    f"{relation.get('target_id', '')}\n{relation.get('description', '')}"
                )
                rel_strs.append(rel_str)
                rel_keywords_order.append(rel_keywords)

        if not rel_strs:
            logger.warning("No relations with keywords found")
            return {}

        # Compute embeddings in batches
        batch_size = 16  # Fixed batch size for relation embeddings
        rel_embeddings = await compute_embeddings_in_batches(
            rel_strs, embedding_func, batch_size
        )

        # Build cache dictionary
        self._relation_embeddings_cache = {}
        for idx, rel_embedding in enumerate(rel_embeddings):
            rel_keywords = rel_keywords_order[idx]
            self._relation_embeddings_cache[rel_keywords] = np.array(rel_embedding)

        # Persist to disk
        try:
            os.makedirs(self.working_dir, exist_ok=True)
            with open(self.relation_embeddings_cache_path, 'wb') as f:
                pickle.dump(self._relation_embeddings_cache, f)
            logger.info(f"Cached {len(self._relation_embeddings_cache)} relation embeddings to disk")
        except Exception as e:
            logger.warning(f"Failed to save relation embeddings cache: {e}")

        return self._relation_embeddings_cache

    async def _get_entity_embeddings_from_vdb(
        self,
        seed_nodes: List[Dict],
        global_config: Dict[str, Any]
    ) -> Dict[str, np.ndarray]:
        """
        Get entity embeddings directly from Vector Storage (same format as insert).
        This reuses embeddings computed during insert, avoiding redundant API calls.

        Args:
            seed_nodes: List of seed node dictionaries
            global_config: Global configuration with entities_vdb

        Returns:
            Dict mapping entity_name -> embedding vector
        """
        try:
            entities_vdb = global_config.get("entities_vdb")
            if not entities_vdb or not hasattr(entities_vdb, '_get_client'):
                logger.debug("Vector storage doesn't support direct vector access")
                return {}

            # Collect entity names
            entity_names = [node.get("entity_name") for node in seed_nodes if node.get("entity_name")]
            if not entity_names:
                return {}

            # Generate vector database IDs
            from ..utils import compute_mdhash_id
            entity_vdb_ids = [compute_mdhash_id(name, prefix="ent-") for name in entity_names]

            # Direct access for NanoVectorDB
            client = await entities_vdb._get_client()
            vdb_results = client.get(entity_vdb_ids)

            entity_vectors = {}
            for i, vdb_data in enumerate(vdb_results):
                if vdb_data and 'vector' in vdb_data:
                    entity_name = entity_names[i]
                    try:
                        # Decode NanoVectorDB format: base64 + zlib + float16
                        import base64
                        import zlib

                        compressed_vector = base64.b64decode(vdb_data['vector'])
                        vector_bytes = zlib.decompress(compressed_vector)
                        vector_array = np.frombuffer(vector_bytes, dtype=np.float16).astype(np.float32)
                        entity_vectors[entity_name] = vector_array
                    except Exception as e:
                        logger.debug(f"Error decoding vector for {entity_name}: {e}")

            logger.debug(f"Retrieved {len(entity_vectors)} entity embeddings from Vector Storage")
            return entity_vectors

        except Exception as e:
            logger.debug(f"Error getting entity embeddings from VDB: {e}")
            return {}

    async def get_entity_embeddings_cached(
        self,
        seed_nodes: List[Dict],
        embedding_func
    ) -> Dict[str, np.ndarray]:
        """
        Get entity embeddings with caching.

        Args:
            seed_nodes: List of seed node dictionaries with entity_name and description
            embedding_func: Embedding function to use

        Returns:
            Dict mapping entity_name -> embedding vector
        """
        if self._entity_embeddings_cache is None:
            self._entity_embeddings_cache = {}

        # Collect entities that need embedding computation
        entities_to_compute = []
        entity_names_to_compute = []

        for node in seed_nodes:
            entity_name = node.get("entity_name")
            if not entity_name:
                continue

            # Check if already cached
            if entity_name in self._entity_embeddings_cache:
                continue

            # Get description
            description = node.get("description") or ""
            if description:
                entity_str = f"{entity_name} {description}"
                entities_to_compute.append(entity_str)
                entity_names_to_compute.append(entity_name)

        # Compute embeddings for new entities
        if entities_to_compute:
            logger.debug(f"Computing embeddings for {len(entities_to_compute)} new entities")
            batch_size = 16  # Fixed batch size
            new_embeddings = await compute_embeddings_in_batches(
                entities_to_compute, embedding_func, batch_size
            )

            # Add to cache
            for idx, entity_name in enumerate(entity_names_to_compute):
                self._entity_embeddings_cache[entity_name] = np.array(new_embeddings[idx])

            # Persist to disk
            try:
                os.makedirs(self.working_dir, exist_ok=True)
                with open(self.entity_embeddings_cache_path, 'wb') as f:
                    pickle.dump(self._entity_embeddings_cache, f)
                logger.debug(f"Cached {len(self._entity_embeddings_cache)} entity embeddings to disk")
            except Exception as e:
                logger.warning(f"Failed to save entity embeddings cache: {e}")

        # Return only the embeddings for requested entities
        result = {}
        for node in seed_nodes:
            entity_name = node.get("entity_name")
            if entity_name and entity_name in self._entity_embeddings_cache:
                result[entity_name] = self._entity_embeddings_cache[entity_name]

        return result

    def compute_personalized_pagerank(
        self,
        seed_entities: List[str],
        entity_similarities: Dict[str, float] = None,
        relation_similarities: Dict[str, float] = None,
        tau: float = 0.1,
        alpha: float = 0.3
    ) -> Dict[str, float]:
        """
        Compute personalized PageRank scores with query-aware weighting.

        Args:
            seed_entities: List of entity IDs to use as personalization seeds
            entity_similarities: Optional dict mapping entity_id -> similarity_score for query-aware seed weighting (Phase 1)
            relation_similarities: Optional dict mapping relation_keywords -> similarity_score for direct edge reweighting (Phase 2)
            tau: Temperature parameter for softmax weighting (lower = more focused)
            alpha: Not used in simplified implementation (kept for compatibility)

        Returns:
            Dict mapping entity_id -> personalized_pagerank_score
        """
        if not self.graph or not seed_entities:
            return {}

        try:
            # Create personalization vector
            personalization = {}
            valid_seeds = [e for e in seed_entities if e in self.graph.nodes()]

            if not valid_seeds:
                logger.warning("No valid seed entities found in graph")
                return {}

            # Compute query-aware seed weights if similarity scores provided
            if entity_similarities:
                seed_weights = self._compute_query_aware_weights(valid_seeds, entity_similarities, tau)
                logger.debug(f"Query-aware seed weights computed: {len(seed_weights)} weighted seeds")
            else:
                # Fallback to uniform weights
                seed_weights = {seed: 1.0/len(valid_seeds) for seed in valid_seeds}
                logger.warning(f"Phase 1: No entity similarities available, using uniform seed weights for {len(valid_seeds)} seeds")

            # Build personalization vector for all nodes
            for node in self.graph.nodes():
                personalization[node] = seed_weights.get(node, 0.0)

            # Ensure normalization (should already be normalized, but double-check)
            total = sum(personalization.values())
            if total > 0:
                personalization = {k: v/total for k, v in personalization.items()}

            # Phase 2: Apply direct edge reweighting if relation similarities provided
            if relation_similarities and alpha > 0.0:
                # Temporarily set edge weights based on query-relation similarity
                reweighted_edges = 0
                skipped_edges = 0
                for u, v, edge_data in self.graph.edges(data=True):
                    # Normalize edge key for undirected graphs (lexicographic order)
                    normalized_key = (v, u) if u > v else (u, v)

                    # Use similarity as weight if available, otherwise use fallback weight
                    if normalized_key in relation_similarities:
                        similarity = relation_similarities[normalized_key]
                        self.graph[u][v]['temp_weight'] = similarity
                        reweighted_edges += 1
                    else:
                        # Fallback: use fixed small weight (not LLM-generated weight)
                        self.graph[u][v]['temp_weight'] = 0.1
                        skipped_edges += 1

                logger.info(f"Phase 2: Applied query-aware edge reweighting to {reweighted_edges}/{self.graph.number_of_edges()} edges")
                if skipped_edges > 0:
                    logger.warning(f"Phase 2: {skipped_edges} edges without relation similarities, using fallback weight 0.1")

                # Compute PageRank with temporary weights
                ppr_scores = nx.pagerank(
                    self.graph,
                    alpha=self.config.pagerank_alpha,
                    personalization=personalization,
                    max_iter=self.config.pagerank_max_iter,
                    tol=self.config.pagerank_tol,
                    weight='temp_weight'
                )

                # Clean up temporary weights
                for u, v in self.graph.edges():
                    if 'temp_weight' in self.graph[u][v]:
                        del self.graph[u][v]['temp_weight']
            else:
                # Compute PageRank with original weights
                ppr_scores = nx.pagerank(
                    self.graph,
                    alpha=self.config.pagerank_alpha,
                    personalization=personalization,
                    max_iter=self.config.pagerank_max_iter,
                    tol=self.config.pagerank_tol,
                    weight='weight'
                )

            logger.info(f"Personalized PageRank computed for {len(ppr_scores)} nodes with {len(valid_seeds)} seeds")
            return ppr_scores

        except Exception as e:
            logger.error(f"Error computing Personalized PageRank: {e}")
            return {}
        
    def extract_reasoning_paths(
        self, 
        source_entities: List[str], 
        target_entities: List[str],
        max_paths: int = 5,
        max_length: int = 4
    ) -> List[List[str]]:
        """Extract reasoning paths between source and target entities."""
        if not self.graph:
            return []
            
        paths = []
        
        for source in source_entities:
            for target in target_entities:
                if source == target:
                    continue
                    
                if source in self.graph.nodes() and target in self.graph.nodes():
                    try:
                        # Find shortest paths
                        all_paths = list(nx.all_simple_paths(
                            self.graph, source, target, cutoff=max_length
                        ))
                        
                        # Sort by length and take top paths
                        all_paths.sort(key=len)
                        paths.extend(all_paths[:max_paths])
                        
                    except nx.NetworkXNoPath:
                        continue
                        
        # Remove duplicates and sort by length  
        unique_paths = []
        for path in paths:
            if path not in unique_paths:
                unique_paths.append(path)
                
        unique_paths.sort(key=len)
        return unique_paths[:max_paths]

    async def compute_query_aware_ppr(
        self,
        seed_nodes: List[Dict],
        top_ppr_nodes: int,
        knowledge_graph_inst,
        global_config: Dict[str, Any],
        ll_keywords: str = "",
        hl_keywords: str = "",
        query_ll_embedding: List[float] | None = None,
        query_hl_embedding: List[float] | None = None,
    ) -> List[Dict]:
        """Compute top PPR entities with query-aware weighting.

        Args:
            query_ll_embedding: Pre-computed ll_keywords embedding (optional, for caching)
            query_hl_embedding: Pre-computed hl_keywords embedding (optional, for caching)
        """
        if top_ppr_nodes <= 0:
            return []

        seed_entity_names = [
            node.get("entity_name", "")
            for node in seed_nodes
            if node.get("entity_name")
        ]
        if not seed_entity_names:
            logger.warning("No seed entities for PPR analysis")
            return []

        logger.info(f"PPR analysis from {len(seed_entity_names)} seed entities")

        entity_similarities: Dict[str, float] = {}
        if ll_keywords.strip():
            try:
                entities_vdb = global_config.get("entities_vdb")
                if entities_vdb and hasattr(entities_vdb, "embedding_func"):
                    embedding_func = entities_vdb.embedding_func
                    if embedding_func and embedding_func.func:
                        # Use pre-computed embedding if available, otherwise compute it
                        if query_ll_embedding is not None:
                            query_vec = np.array(query_ll_embedding)
                            logger.debug("Using pre-computed ll_keywords embedding for PPR Phase 1")
                        else:
                            query_embedding = await embedding_func.func([ll_keywords])
                            query_vec = np.array(query_embedding[0])
                            logger.debug("Computed ll_keywords embedding for PPR Phase 1")

                        # Try to get entity embeddings from Vector Storage first (same format as insert)
                        # This avoids recomputing embeddings that were already computed during insert
                        entity_embeddings_dict = await self._get_entity_embeddings_from_vdb(
                            seed_nodes, global_config
                        )

                        if entity_embeddings_dict:
                            # Vectorized similarity computation for all entities at once
                            entity_names_list = list(entity_embeddings_dict.keys())
                            entity_embeddings_matrix = np.vstack([
                                entity_embeddings_dict[name] for name in entity_names_list
                            ])

                            # Normalize query vector
                            query_norm = np.linalg.norm(query_vec)
                            if query_norm > 0:
                                query_normalized = query_vec / query_norm

                                # Normalize entity embeddings matrix (row-wise)
                                entity_norms = np.linalg.norm(entity_embeddings_matrix, axis=1, keepdims=True)
                                entity_norms[entity_norms == 0] = 1  # Avoid division by zero
                                entity_embeddings_normalized = entity_embeddings_matrix / entity_norms

                                # Compute all similarities with single matrix multiplication
                                similarities = entity_embeddings_normalized @ query_normalized

                                # Build result dictionary
                                for idx, entity_name in enumerate(entity_names_list):
                                    entity_similarities[entity_name] = float(similarities[idx])

                        logger.info(f"Phase 1: Computed {len(entity_similarities)} entity similarities for query-aware seed weighting")

            except Exception as exc:
                logger.warning(f"Could not compute query-aware similarities: {exc}")

        relation_similarities: Dict[str, float] = {}
        if hl_keywords.strip():
            try:
                relationships_vdb = global_config.get("relationships_vdb")
                if relationships_vdb and hasattr(relationships_vdb, "embedding_func"):
                    embedding_func = relationships_vdb.embedding_func
                    if embedding_func and embedding_func.func:
                        # Use pre-computed embedding if available, otherwise compute it
                        if query_hl_embedding is not None:
                            hl_query_vec = np.array(query_hl_embedding)
                            logger.debug("Using pre-computed hl_keywords embedding for PPR Phase 2")
                        else:
                            hl_query_embedding_result = await embedding_func.func([hl_keywords])
                            hl_query_vec = np.array(hl_query_embedding_result[0])
                            logger.debug("Computed hl_keywords embedding for PPR Phase 2")

                        # Try to get relation embeddings from Vector Storage first (same format as insert)
                        # This avoids recomputing embeddings that were already computed during insert
                        relation_embeddings_dict = await self._get_relation_embeddings_from_vdb(
                            knowledge_graph_inst, global_config
                        )

                        if relation_embeddings_dict:
                            # Vectorized similarity computation for all relations at once
                            edge_keys = list(relation_embeddings_dict.keys())  # List of (src_id, tgt_id) tuples
                            rel_embeddings_matrix = np.vstack([
                                relation_embeddings_dict[k]['vector'] for k in edge_keys
                            ])

                            # Normalize query vector
                            query_norm = np.linalg.norm(hl_query_vec)
                            if query_norm > 0:
                                hl_query_normalized = hl_query_vec / query_norm

                                # Normalize relation embeddings matrix (row-wise)
                                rel_norms = np.linalg.norm(rel_embeddings_matrix, axis=1, keepdims=True)
                                rel_norms[rel_norms == 0] = 1  # Avoid division by zero
                                rel_embeddings_normalized = rel_embeddings_matrix / rel_norms

                                # Compute all similarities with single matrix multiplication
                                similarities = rel_embeddings_normalized @ hl_query_normalized

                                # Build result dictionary - now keyed by (src_id, tgt_id) instead of keywords
                                for idx, edge_key in enumerate(edge_keys):
                                    relation_similarities[edge_key] = float(similarities[idx])

                        logger.info(f"Phase 2: Computed {len(relation_similarities)} relation similarities for query-aware edge reweighting")

            except Exception as exc:
                logger.warning(f"Could not compute relation similarities: {exc}")

        ppr_scores = self.compute_personalized_pagerank(
            seed_entity_names,
            entity_similarities if entity_similarities else None,
            relation_similarities if relation_similarities else None,
            tau=0.1,
            alpha=0.3
        )

        if not ppr_scores:
            logger.warning("No Personalized PageRank scores computed")
            return []

        all_entities = [
            (entity_name, score)
            for entity_name, score in ppr_scores.items()
            if entity_name not in seed_entity_names
        ]
        if not all_entities:
            logger.warning("No entities found for PPR ranking")
            return []

        all_entities.sort(key=lambda x: x[1], reverse=True)
        top_entities = all_entities[:top_ppr_nodes]

        logger.info(f"PPR selected top {len(top_entities)} entities from {len(all_entities)} total entities")

        top_entity_names = [entity_name for entity_name, _ in top_entities]
        nodes_dict = await knowledge_graph_inst.get_nodes_batch(top_entity_names)

        ppr_entities: List[Dict] = []
        for entity_name, ppr_score in top_entities:
            entity_data = (nodes_dict or {}).get(entity_name)
            if entity_data:
                entity_copy = dict(entity_data)
                entity_copy.update({
                    "entity_name": entity_name,
                    "pagerank_score": ppr_score,
                    "discovery_method": "ppr_analysis",
                    "rank": entity_data.get("degree", 0),
                })
                ppr_entities.append(entity_copy)

        avg_ppr = (sum(e["pagerank_score"] for e in ppr_entities) / len(ppr_entities)) if ppr_entities else 0.0
        logger.info(
            "PPR analysis completed: %d entities with avg PageRank: %.4f",
            len(ppr_entities),
            avg_ppr,
        )


        return ppr_entities

    async def compute_adaptive_fastrp(
        self,
        seed_nodes: List[Dict],
        top_fastrp_nodes: int,
        knowledge_graph_inst
    ) -> List[Dict]:
        """Compute top FastRP entities using vectorized similarities."""
        if top_fastrp_nodes <= 0:
            return []

        seed_entity_names = [
            node.get("entity_name", "")
            for node in seed_nodes
            if node.get("entity_name")
        ]
        if not seed_entity_names:
            logger.warning("No seed entities for FastRP analysis")
            return []

        if not hasattr(self, "fastrp_embeddings") or not self.fastrp_embeddings:
            logger.warning("FastRP embeddings not available for FastRP analysis")
            return []

        logger.info(f"FastRP analysis from {len(seed_entity_names)} seed entities")

        candidate_scores = compute_adaptive_fastrp_batch(
            seed_entity_names,
            self,
        )

        if not candidate_scores:
            logger.warning("No valid FastRP candidates found")
            return []

        sorted_candidates = sorted(
            candidate_scores.items(),
            key=lambda item: item[1],
            reverse=True
        )
        top_candidates = sorted_candidates[:top_fastrp_nodes]

        top_entity_names = [entity_name for entity_name, _ in top_candidates]
        nodes_dict = await knowledge_graph_inst.get_nodes_batch(top_entity_names)

        top_smart_entities: List[Dict] = []
        for entity_name, similarity in top_candidates:
            entity_data = (nodes_dict or {}).get(entity_name)
            if entity_data:
                entity_copy = dict(entity_data)
                entity_copy.update({
                    "entity_name": entity_name,
                    "adaptive_fastrp_similarity": similarity,
                    "discovery_method": "fastrp_analysis",
                    "rank": entity_data.get("degree", 0),
                })
                top_smart_entities.append(entity_copy)

        logger.info(
            "Content-aware FastRP selected top %d entities with avg similarity: %.4f",
            len(top_smart_entities),
            (sum(e["adaptive_fastrp_similarity"] for e in top_smart_entities) / len(top_smart_entities))
            if top_smart_entities else 0.0,
        )

        return top_smart_entities
    
    def _save_persisted_data(self):
        """Save graph structure and embeddings to disk for query-time reuse."""
        try:
            # Ensure working directory exists
            os.makedirs(self.working_dir, exist_ok=True)
            
            # Save graph structure with gzip compression
            if self.graph:
                import gzip
                graph_path_gz = self.graph_path + '.gz'
                with gzip.open(graph_path_gz, 'wb', compresslevel=1) as f:
                    pickle.dump(self.graph, f, protocol=4)
                logger.info(f"Saved compressed graph with {len(self.graph.nodes())} nodes")
            
            # Save FastRP embeddings with gzip compression
            if self.fastrp_embeddings:
                import gzip
                fastrp_path_gz = self.fastrp_path + '.gz'
                with gzip.open(fastrp_path_gz, 'wb', compresslevel=1) as f:
                    pickle.dump(self.fastrp_embeddings, f, protocol=4)
                logger.info(f"Saved compressed FastRP embeddings for {len(self.fastrp_embeddings)} entities")

        except Exception as e:
            logger.error(f"Error saving node embedding data: {e}")
    
    def _load_persisted_data(self):
        """Load previously saved graph structure and embeddings."""
        try:
            # Load graph structure (support both .pkl and .pkl.gz for backward compatibility)
            graph_path_gz = self.graph_path + '.gz'
            if os.path.exists(graph_path_gz):
                import gzip
                with gzip.open(graph_path_gz, 'rb') as f:
                    self.graph = pickle.load(f)
                logger.info(f"Loaded compressed graph with {len(self.graph.nodes())} nodes")
            elif os.path.exists(self.graph_path):
                with open(self.graph_path, 'rb') as f:
                    self.graph = pickle.load(f)
                logger.info(f"Loaded graph with {len(self.graph.nodes())} nodes from {self.graph_path}")

            # Load FastRP embeddings (support both .pkl and .pkl.gz for backward compatibility)
            fastrp_path_gz = self.fastrp_path + '.gz'
            if os.path.exists(fastrp_path_gz):
                import gzip
                with gzip.open(fastrp_path_gz, 'rb') as f:
                    self.fastrp_embeddings = pickle.load(f)
                logger.info(f"Loaded compressed FastRP embeddings for {len(self.fastrp_embeddings)} entities")
            elif os.path.exists(self.fastrp_path):
                with open(self.fastrp_path, 'rb') as f:
                    self.fastrp_embeddings = pickle.load(f)
                logger.info(f"Loaded FastRP embeddings for {len(self.fastrp_embeddings)} entities")

            # Load Relations cache (support both .pkl and .pkl.gz for backward compatibility)
            cache_path_gz = self.relations_cache_path + '.gz'
            if os.path.exists(cache_path_gz):
                import gzip
                with gzip.open(cache_path_gz, 'rb') as f:
                    self._relations_cache = pickle.load(f)
                logger.info(f"Loaded compressed relations cache with {len(self._relations_cache)} relations")
            elif os.path.exists(self.relations_cache_path):
                with open(self.relations_cache_path, 'rb') as f:
                    self._relations_cache = pickle.load(f)
                logger.info(f"Loaded relations cache with {len(self._relations_cache)} relations")

            # Load Relation embeddings cache
            if os.path.exists(self.relation_embeddings_cache_path):
                with open(self.relation_embeddings_cache_path, 'rb') as f:
                    self._relation_embeddings_cache = pickle.load(f)
                logger.info(f"Loaded relation embeddings cache for {len(self._relation_embeddings_cache)} relations")

            # Load Entity embeddings cache
            if os.path.exists(self.entity_embeddings_cache_path):
                with open(self.entity_embeddings_cache_path, 'rb') as f:
                    self._entity_embeddings_cache = pickle.load(f)
                logger.info(f"Loaded entity embeddings cache for {len(self._entity_embeddings_cache)} entities")

        except Exception as e:
            logger.warning(f"Could not load persisted node embedding data: {e}")
            # Reset to empty state
            self.graph = None
            self.fastrp_embeddings = None
            self._relations_cache = None
            self._relation_embeddings_cache = None
            self._entity_embeddings_cache = None

    def _compute_query_aware_weights(
        self,
        valid_seeds: List[str],
        entity_similarities: Dict[str, float],
        tau: float = 0.01
    ) -> Dict[str, float]:
        """
        Compute query-aware seed weights using softmax on similarity scores.

        Args:
            valid_seeds: List of valid seed entities in the graph
            entity_similarities: Dict mapping entity_id -> similarity_score
            tau: Temperature parameter for softmax (lower = more focused)

        Returns:
            Dict mapping seed_entity -> normalized_weight
        """
        import numpy as np

        # Extract similarity scores for valid seeds
        scores = []
        seeds_with_scores = []

        for seed in valid_seeds:
            if seed in entity_similarities:
                scores.append(entity_similarities[seed])
                seeds_with_scores.append(seed)
            else:
                # Fallback score for seeds without similarity data
                scores.append(0.0)
                seeds_with_scores.append(seed)

        if not seeds_with_scores:
            logger.warning("No seeds with similarity scores, falling back to uniform weights")
            return {seed: 1.0/len(valid_seeds) for seed in valid_seeds}

        # Apply softmax with temperature
        scores = np.array(scores)
        exp_scores = np.exp(scores / tau)
        weights = exp_scores / np.sum(exp_scores)

        # Create weight dictionary
        seed_weights = {seed: float(weight) for seed, weight in zip(seeds_with_scores, weights)}

        logger.debug(f"Query-aware weights: max={max(weights):.3f}, min={min(weights):.3f}, "
                    f"std={np.std(weights):.3f}")

        return seed_weights

