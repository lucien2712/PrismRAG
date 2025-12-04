import json
import asyncio
import os
import hashlib
from typing import Union
from .utils import logger
from .prompt import PROMPTS


def load_existing_entity_types(working_dir: str) -> list:
    """Load existing entity types from working_dir."""
    entity_type_path = os.path.join(working_dir, "entity_types.json")
    if os.path.exists(entity_type_path):
        try:
            with open(entity_type_path, "r", encoding="utf-8") as f:
                entity_types = json.load(f)
            logger.info(f"Loaded {len(entity_types)} entity types from {entity_type_path}")
            return entity_types
        except Exception as e:
            logger.warning(f"Failed to load entity types from {entity_type_path}: {e}")

    # Return default entity types if file doesn't exist or fails to load
    default_entity_types = [
        {
            "entity_type": "Person",
            "explanation": "An entity representing individual persons."
        },
        {
            "entity_type": "Creature",
            "explanation": "An entity representing living beings, animals, or organisms."
        },
        {
            "entity_type": "Organization",
            "explanation": "An entity representing organizations, companies, or institutions."
        },
        {
            "entity_type": "Location",
            "explanation": "An entity representing geographical locations."
        },
        {
            "entity_type": "Event",
            "explanation": "An entity representing events or activities."
        },
        {
            "entity_type": "Concept",
            "explanation": "An entity representing abstract concepts, ideas, or theories."
        },
        {
            "entity_type": "Method",
            "explanation": "An entity representing methods, techniques, or procedures."
        },
        {
            "entity_type": "Content",
            "explanation": "An entity representing content, information, or media."
        },
        {
            "entity_type": "Data",
            "explanation": "An entity representing data, datasets, or information structures."
        },
        {
            "entity_type": "Artifact",
            "explanation": "An entity representing artifacts, objects, or manufactured items."
        },
        {
            "entity_type": "NaturalObject",
            "explanation": "An entity representing natural objects or phenomena."
        }
    ]
    logger.info(f"Using default entity types: {len(default_entity_types)} types")
    return default_entity_types


def save_entity_types(working_dir: str, entity_types: list):
    """Save entity types to working_dir."""
    entity_type_path = os.path.join(working_dir, "entity_types.json")
    try:
        with open(entity_type_path, "w", encoding="utf-8") as f:
            json.dump(entity_types, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved {len(entity_types)} entity types to {entity_type_path}")
    except Exception as e:
        logger.error(f"Failed to save entity types to {entity_type_path}: {e}")
        raise


def load_processing_status(working_dir: str) -> dict:
    """Load file processing status from working_dir."""
    status_path = os.path.join(working_dir, "entity_check_status.json")
    if os.path.exists(status_path):
        try:
            with open(status_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load processing status from {status_path}: {e}")
    return {}


def save_processing_status(working_dir: str, status: dict):
    """Save file processing status to working_dir."""
    status_path = os.path.join(working_dir, "entity_check_status.json")
    try:
        with open(status_path, "w", encoding="utf-8") as f:
            json.dump(status, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved processing status for {len(status)} files to {status_path}")
    except Exception as e:
        logger.error(f"Failed to save processing status to {status_path}: {e}")
        raise


def update_processing_status(working_dir: str, file_text_pairs: list):
    """Update processing status for processed files."""
    status = load_processing_status(working_dir)
    for file_name, _ in file_text_pairs:
        status[file_name] = True
    save_processing_status(working_dir, status)


def process_texts(working_dir: str, texts: Union[str, list[str]], force_refresh: bool = False) -> list:
    """Process text content directly without file I/O.

    Args:
        working_dir: Working directory for status tracking
        texts: String or list of strings to analyze.
        force_refresh: Whether to reprocess already-processed texts.

    Returns:
        list: List of (identifier, text_content) tuples.
    """
    # Load processing status to check for already-processed content
    processed_texts = load_processing_status(working_dir) if not force_refresh else {}
    file_text_pairs = []

    # Convert single string to list for uniform processing
    if isinstance(texts, str):
        texts = [texts]

    for idx, text_content in enumerate(texts):
        if text_content.strip():  # Only add non-empty texts
            # Use content hash as identifier for deduplication
            content_hash = hashlib.md5(text_content.encode('utf-8')).hexdigest()[:16]
            identifier = f"text_{idx+1}_{content_hash}"

            # Check if this content hash was already processed
            if not force_refresh and identifier in processed_texts:
                logger.debug(f"Skipped: {identifier} (already processed)")
                continue

            file_text_pairs.append((identifier, text_content))
            logger.info(f"Added text content: {identifier}")
        else:
            logger.warning(f"Skipped empty text at index {idx}")

    logger.info(f"Selected {len(file_text_pairs)} text items for processing (skipped {len(texts) - len(file_text_pairs)} already-processed)")
    return file_text_pairs


def process_files_from_folder(working_dir: str, folder_path: str, force_refresh: bool = False) -> list:
    """Process files from a specified folder."""
    if not os.path.exists(folder_path):
        logger.warning(f"Folder {folder_path} does not exist")
        return []

    processed_files = load_processing_status(working_dir) if not force_refresh else {}
    file_text_pairs = []

    try:
        all_files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]
        logger.info(f"Found {len(all_files)} .txt files in {folder_path}")

        for file_name in all_files:
            if not force_refresh and file_name in processed_files:
                logger.debug(f"Skipped: {file_name} (already processed)")
                continue

            file_path = os.path.join(folder_path, file_name)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    text_content = f.read()

                if text_content.strip():  # Only add non-empty files
                    file_text_pairs.append((file_name, text_content))
                    logger.info(f"Added file: {file_name}")
                else:
                    logger.warning(f"Skipped empty file: {file_name}")

            except Exception as e:
                logger.error(f"Failed to read {file_name}: {e}")

    except Exception as e:
        logger.error(f"Error processing folder {folder_path}: {e}")

    logger.info(f"Selected {len(file_text_pairs)} files for processing")
    return file_text_pairs


async def process_files_from_chunks_db(text_chunks_db, working_dir: str, force_refresh: bool = False) -> list:
    """Process files from the text_chunks_db storage."""
    try:
        # Get all chunk data from storage
        logger.info("Retrieving documents from text_chunks_db")

        # Access the storage to get all documents
        all_chunk_keys = []
        try:
            # Try to get all keys from the storage
            if hasattr(text_chunks_db, 'get_all_keys'):
                all_chunk_keys = await text_chunks_db.get_all_keys()
            elif hasattr(text_chunks_db, '_client'):
                # For some storage implementations, we might need to access the client directly
                client = await text_chunks_db._get_client()
                if hasattr(client, 'keys'):
                    all_chunk_keys = list(client.keys())
            else:
                logger.warning("Cannot retrieve all keys from text_chunks_db - method not supported")
                return []

        except Exception as e:
            logger.warning(f"Could not retrieve all keys from text_chunks_db: {e}")
            return []

        if not all_chunk_keys:
            logger.info("No documents found in text_chunks_db")
            return []

        logger.info(f"Found {len(all_chunk_keys)} chunks in text_chunks_db")

        processed_files = load_processing_status(working_dir) if not force_refresh else {}
        file_content_map = {}

        # Group chunks by file_path
        for chunk_key in all_chunk_keys:
            try:
                chunk_data = await text_chunks_db.get_by_id(chunk_key)
                if chunk_data and 'content' in chunk_data:
                    # Use full_doc_id or chunk_key as file identifier
                    file_id = chunk_data.get('full_doc_id', chunk_key)

                    if not force_refresh and file_id in processed_files:
                        continue

                    if file_id not in file_content_map:
                        file_content_map[file_id] = []

                    file_content_map[file_id].append(chunk_data['content'])

            except Exception as e:
                logger.warning(f"Failed to process chunk {chunk_key}: {e}")
                continue

        # Combine chunks for each file
        file_text_pairs = []
        for file_id, contents in file_content_map.items():
            combined_content = "\n\n".join(contents)
            if combined_content.strip():
                file_text_pairs.append((file_id, combined_content))

        logger.info(f"Prepared {len(file_text_pairs)} document groups for entity type analysis")
        return file_text_pairs

    except Exception as e:
        logger.error(f"Error processing files from chunks_db: {e}")
        return []


def create_openai_client(tool_llm_model_name: str = None):
    """Create OpenAI client using environment variable."""
    try:
        import openai
        return openai.OpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            base_url=os.getenv("OPENAI_API_BASE_URL", "https://api.openai.com/v1")
        )
    except ImportError:
        raise ImportError("Please install openai package: pip install openai")
    except KeyError:
        raise ValueError("OPENAI_API_KEY environment variable is not set")


async def process_file_with_llm(
    file_content: str,
    current_entity_types: list,
    tool_llm_model_name: str,
    tokenizer,
    _extract_json_callback,
    _process_large_file_callback,
    openai_client=None,  # 新增參數：共用的 OpenAI client
    token_tracker=None,  # 新增參數：token tracker
    tool_llm_model_kwargs: dict = None  # 新增參數：tool LLM kwargs
) -> list:
    """Process a file with LLM to suggest new entity types.

    Automatically chunks documents that exceed 30000 tokens to avoid token limit issues.
    """
    # Check token count and pre-chunk if needed
    try:
        token_count = len(tokenizer.encode(file_content))
        logger.debug(f"Document has {token_count} tokens")

        # If document exceeds 30000 tokens, use chunking approach
        if token_count > 30000:
            logger.info(f"Document has {token_count} tokens (>30000), using chunking approach")
            return await _process_large_file_callback(
                file_content, current_entity_types
            )
    except Exception as e:
        logger.warning(f"Could not count tokens, will attempt direct processing: {e}")

    # For documents under 30000 tokens, process directly with retry mechanism
    max_retries = 3
    # 使用傳入的 client，或 fallback 到建立新的
    if openai_client is None:
        client = create_openai_client(tool_llm_model_name)
    else:
        client = openai_client
    current_entity_types_json = json.dumps(
        [et["entity_type"] for et in current_entity_types]
    )

    if tool_llm_model_kwargs is None:
        tool_llm_model_kwargs = {}

    for attempt in range(max_retries):
        try:
            # Prepare API call parameters
            api_params = {
                "model": tool_llm_model_name,
                "messages": [
                    {
                        "role": "system",
                        "content": PROMPTS["entity_type_suggestion_system"],
                    },
                    {
                        "role": "user",
                        "content": PROMPTS["entity_type_suggestion_user"].format(
                            current_entity_types=current_entity_types_json,
                            file_content=file_content
                        ),
                    },
                ],
                **tool_llm_model_kwargs  # Merge additional kwargs
            }

            response = client.chat.completions.create(**api_params)

            # Track token usage if token_tracker is provided
            if token_tracker and hasattr(response, 'usage') and response.usage:
                token_counts = {
                    'prompt_tokens': getattr(response.usage, 'prompt_tokens', 0),
                    'completion_tokens': getattr(response.usage, 'completion_tokens', 0),
                    'total_tokens': getattr(response.usage, 'total_tokens', 0),
                }
                token_tracker.add_usage(token_counts)

            llm_response = response.choices[0].message.content
            suggested_types = _extract_json_callback(llm_response)

            # Check if we got valid results (non-empty list)
            if suggested_types and isinstance(suggested_types, list):
                logger.info(f"Successfully extracted {len(suggested_types)} entity types on attempt {attempt + 1}")
                return suggested_types
            else:
                logger.warning(f"Attempt {attempt + 1}/{max_retries}: No valid entity types extracted, retrying...")
                continue

        except Exception as e:
            # For any errors, retry
            logger.warning(f"Attempt {attempt + 1}/{max_retries}: Error processing file with LLM: {e}")
            if attempt < max_retries - 1:
                continue
            else:
                logger.error(f"Failed to process file after {max_retries} attempts")
                return []

    # If all retries failed
    logger.error(f"Failed to extract entity types after {max_retries} attempts, returning empty list")
    return []


async def process_large_file_with_chunking(
    file_content: str,
    current_entity_types: list,
    tool_llm_model_name: str,
    tokenizer,
    _extract_json_callback,
    openai_client=None,  # 新增參數：共用的 OpenAI client
    token_tracker=None,  # 新增參數：token tracker
    tool_llm_model_kwargs: dict = None  # 新增參數：tool LLM kwargs
) -> list:
    """Process a large file by splitting into chunks.

    This function splits large documents into manageable chunks,
    processes each chunk separately, and aggregates the results.

    Args:
        file_content: The full text content to process
        current_entity_types: List of existing entity types
        tool_llm_model_name: Model name to use
        tokenizer: Tokenizer for chunking
        _extract_json_callback: Function to extract JSON from response

    Returns:
        List of suggested entity types aggregated from all chunks
    """
    from .operate import chunking_by_token_size

    logger.info(f"Splitting large document into chunks for entity type extraction...")

    # Split document into chunks (30000 tokens per chunk, 500 token overlap)
    chunks = chunking_by_token_size(
        tokenizer=tokenizer,
        content=file_content,
        max_token_size=30000,
        overlap_token_size=500
    )

    logger.info(f"Split document into {len(chunks)} chunks")

    # Process each chunk with LLM
    all_suggested_types = []
    # 使用傳入的 client，或 fallback 到建立新的
    if openai_client is None:
        client = create_openai_client(tool_llm_model_name)
    else:
        client = openai_client
    current_entity_types_json = json.dumps(
        [et["entity_type"] for et in current_entity_types]
    )

    if tool_llm_model_kwargs is None:
        tool_llm_model_kwargs = {}

    for chunk_idx, chunk_data in enumerate(chunks, 1):
        chunk_content = chunk_data["content"]
        logger.info(f"Processing chunk {chunk_idx}/{len(chunks)}")

        try:
            # Prepare API call parameters
            api_params = {
                "model": tool_llm_model_name,
                "messages": [
                    {
                        "role": "system",
                        "content": PROMPTS["entity_type_suggestion_system"],
                    },
                    {
                        "role": "user",
                        "content": PROMPTS["entity_type_suggestion_user"].format(
                            current_entity_types=current_entity_types_json,
                            file_content=chunk_content
                        ),
                    },
                ],
                **tool_llm_model_kwargs  # Merge additional kwargs
            }

            response = client.chat.completions.create(**api_params)

            # Track token usage if token_tracker is provided
            if token_tracker and hasattr(response, 'usage') and response.usage:
                token_counts = {
                    'prompt_tokens': getattr(response.usage, 'prompt_tokens', 0),
                    'completion_tokens': getattr(response.usage, 'completion_tokens', 0),
                    'total_tokens': getattr(response.usage, 'total_tokens', 0),
                }
                token_tracker.add_usage(token_counts)

            llm_response = response.choices[0].message.content
            suggested_types = _extract_json_callback(llm_response)

            if suggested_types and isinstance(suggested_types, list):
                all_suggested_types.extend(suggested_types)
                logger.info(f"Chunk {chunk_idx}: Extracted {len(suggested_types)} entity types")
            else:
                logger.warning(f"Chunk {chunk_idx}: No valid entity types extracted")

        except Exception as e:
            logger.warning(f"Chunk {chunk_idx}: Error processing chunk: {e}")
            continue

    # Deduplicate entity types by entity_type field
    seen_types = set()
    deduplicated_types = []

    for entity_type_obj in all_suggested_types:
        entity_type_name = entity_type_obj.get("entity_type", "")
        if entity_type_name and entity_type_name not in seen_types:
            seen_types.add(entity_type_name)
            deduplicated_types.append(entity_type_obj)

    logger.info(f"Chunk processing complete: {len(all_suggested_types)} total types -> {len(deduplicated_types)} after deduplication")
    return deduplicated_types


async def refine_entity_types(entity_types: list, tool_llm_model_name: str, _extract_json_callback, openai_client=None, token_tracker=None, tool_llm_model_kwargs: dict = None) -> list:
    """Use LLM to remove duplicates and refine entity types with retry mechanism."""
    max_retries = 3
    # 使用傳入的 client，或 fallback 到建立新的
    if openai_client is None:
        client = create_openai_client(tool_llm_model_name)
    else:
        client = openai_client
    entity_types_json = json.dumps(entity_types, ensure_ascii=False, indent=2)

    if tool_llm_model_kwargs is None:
        tool_llm_model_kwargs = {}

    for attempt in range(max_retries):
        try:
            # Prepare API call parameters
            api_params = {
                "model": tool_llm_model_name,
                "messages": [
                    {
                        "role": "system",
                        "content": PROMPTS["entity_type_refinement_system"],
                    },
                    {
                        "role": "user",
                        "content": PROMPTS["entity_type_refinement_user"].format(
                            entity_types=entity_types_json
                        ),
                    },
                ],
                **tool_llm_model_kwargs  # Merge additional kwargs
            }

            response = client.chat.completions.create(**api_params)

            # Track token usage if token_tracker is provided
            if token_tracker and hasattr(response, 'usage') and response.usage:
                token_counts = {
                    'prompt_tokens': getattr(response.usage, 'prompt_tokens', 0),
                    'completion_tokens': getattr(response.usage, 'completion_tokens', 0),
                    'total_tokens': getattr(response.usage, 'total_tokens', 0),
                }
                token_tracker.add_usage(token_counts)

            response_content = response.choices[0].message.content
            refined_entity_types = _extract_json_callback(response_content)

            # Check if we got valid results (non-empty list)
            if refined_entity_types and isinstance(refined_entity_types, list):
                logger.info(f"Successfully refined {len(entity_types)} entity types down to {len(refined_entity_types)} on attempt {attempt + 1}")
                return refined_entity_types
            else:
                logger.warning(f"Attempt {attempt + 1}/{max_retries}: No valid refined types extracted, retrying...")
                continue

        except Exception as e:
            logger.warning(f"Attempt {attempt + 1}/{max_retries}: Error refining entity types: {e}")
            if attempt < max_retries - 1:
                continue
            else:
                logger.error(f"Failed to refine entity types after {max_retries} attempts, returning original")
                return entity_types

    # If all retries failed, return original entity types
    logger.warning(f"LLM refinement failed after {max_retries} attempts, returning original entity types")
    return entity_types


def extract_json_from_response(response_content: str) -> list:
    """Extract JSON from LLM response using json_repair for robustness."""
    import json_repair

    try:
        json_start = (
            response_content.find("[")
            if "[" in response_content
            else response_content.find("{")
        )
        json_end = (
            response_content.rfind("]") + 1
            if "]" in response_content
            else response_content.rfind("}") + 1
        )

        if json_start != -1 and json_end != -1:
            json_str = response_content[json_start:json_end]
            logger.debug(f"Extracted JSON string: {json_str[:200]}...")

            # Use json_repair for robust parsing
            result = json_repair.loads(json_str)

            # Ensure result is a list
            if isinstance(result, list):
                return result
            elif isinstance(result, dict):
                # If single dict returned, wrap in list
                return [result]
            else:
                logger.warning(f"Unexpected JSON type: {type(result)}")
                return []

        logger.warning("No valid JSON found in LLM response")
        return []

    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
        logger.error(f"Problematic response: {response_content[:500]}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error extracting JSON: {e}")
        return []


def update_entity_types_from_working_dir(rag_instance):
    """Update entity_types in addon_params from working_dir if available."""
    entity_type_path = os.path.join(rag_instance.working_dir, "entity_types.json")
    if os.path.exists(entity_type_path):
        try:
            with open(entity_type_path, "r", encoding="utf-8") as f:
                entity_types_data = json.load(f)

            # Extract just the entity_type names for addon_params
            if entity_types_data and isinstance(entity_types_data, list):
                entity_type_names = [et.get("entity_type", "") for et in entity_types_data if et.get("entity_type")]
                if entity_type_names:
                    # Update the addon_params with our entity types
                    rag_instance.addon_params["entity_types"] = entity_type_names
                    logger.info(f"Updated entity_types from working_dir: {len(entity_type_names)} types loaded")
                    logger.debug(f"Entity types: {entity_type_names}")
                    return

        except Exception as e:
            logger.warning(f"Failed to load entity types from {entity_type_path}: {e}")

    # If file doesn't exist or loading failed, keep the default entity types from constants.py
    logger.info("Using default entity types from constants.py")


async def entity_type_discovery(
    rag_instance,
    input_folder: str = None,
    texts: Union[str, list[str]] = None,
    force_refresh: bool = False,
    token_tracker=None,
    tool_llm_model_kwargs: dict = None
) -> dict:
    """Asynchronously augment entity types by analyzing documents.

    This method analyzes documents to discover new entity types using LLM,
    then automatically reloads the updated entity types for use in future operations.

    Args:
        rag_instance: LightRAG instance
        input_folder: Optional folder containing .txt files to analyze.
        texts: Optional text content (string or list of strings) to analyze.
              If provided, takes priority over input_folder and chunks_db.
        force_refresh: Whether to reprocess already processed files.

    Returns:
        dict: Results containing new and refined entity types.
    """
    logger.info("Starting entity type discovery process")

    try:
        # Step 1: Load existing entity types from working_dir
        current_entity_types = load_existing_entity_types(rag_instance.working_dir)
        logger.info(f"Loaded {len(current_entity_types)} existing entity types")

        # Step 2: Get documents to process (priority: texts > input_folder > chunks_db)
        if texts is not None:
            file_text_pairs = process_texts(rag_instance.working_dir, texts, force_refresh)
        elif input_folder:
            file_text_pairs = process_files_from_folder(rag_instance.working_dir, input_folder, force_refresh)
        else:
            file_text_pairs = await process_files_from_chunks_db(rag_instance.text_chunks_db, rag_instance.working_dir, force_refresh)

        if not file_text_pairs:
            logger.info("No new files to process")
            return {
                "status": "success",
                "message": "No new files to process",
                "existing_entity_types": len(current_entity_types),
                "new_entity_types": 0,
                "total_entity_types": len(current_entity_types)
            }

        logger.info(f"Processing {len(file_text_pairs)} documents for entity type discovery")

        # Step 2.5: Pre-chunk all documents into individual chunks for fine-grained processing
        # This allows us to perform refinement every N chunks instead of every N documents
        logger.info("Pre-chunking documents into individual chunks...")
        from .operate import chunking_by_token_size

        chunk_pairs = []  # [(chunk_id, chunk_content), ...]
        for file_name, file_content in file_text_pairs:
            try:
                token_count = len(rag_instance.tokenizer.encode(file_content))

                if token_count > 30000:
                    # Large document: split into chunks
                    chunks = chunking_by_token_size(
                        tokenizer=rag_instance.tokenizer,
                        content=file_content,
                        max_token_size=30000,
                        overlap_token_size=500
                    )
                    logger.info(f"Document '{file_name}': {token_count} tokens → {len(chunks)} chunks")

                    for chunk_idx, chunk_data in enumerate(chunks, 1):
                        chunk_id = f"{file_name}_chunk_{chunk_idx}"
                        chunk_pairs.append((chunk_id, chunk_data["content"]))
                else:
                    # Small document: treat as single chunk
                    logger.debug(f"Document '{file_name}': {token_count} tokens (single chunk)")
                    chunk_pairs.append((file_name, file_content))

            except Exception as e:
                logger.warning(f"Failed to chunk document '{file_name}': {e}, treating as single chunk")
                chunk_pairs.append((file_name, file_content))

        logger.info(f"Pre-chunking complete: {len(file_text_pairs)} documents → {len(chunk_pairs)} chunks")

        # Step 3: Process chunks with LLM to suggest new entity types (with periodic refinement)
        # Refinement interval: every 10 chunks (not documents!)
        max_async = rag_instance.llm_model_max_async
        REFINE_INTERVAL = 10  # Fixed: refine every 10 chunks
        all_new_entity_types = []

        # Use llm_model_max_async for concurrency control
        semaphore = asyncio.Semaphore(max_async)
        logger.info(f"Processing chunks with max_async={max_async}, refinement every {REFINE_INTERVAL} chunks")

        # Create callback wrappers for helper functions
        def _extract_json_callback(response_content):
            return extract_json_from_response(response_content)

        # Note: No need for _process_large_file_callback since chunking is done upfront

        async def process_chunk_with_limit(chunk_id: str, chunk_content: str, entity_types: list):
            """Process a single chunk (already chunked, no auto-chunking needed)"""
            async with semaphore:
                logger.debug(f"Processing chunk: {chunk_id}")
                # Directly process the chunk without auto-chunking logic
                # (token count already verified during pre-chunking)
                max_retries = 3
                client = rag_instance.openai_client
                current_entity_types_json = json.dumps(
                    [et["entity_type"] for et in entity_types]
                )

                if tool_llm_model_kwargs is None:
                    llm_kwargs = {}
                else:
                    llm_kwargs = tool_llm_model_kwargs

                for attempt in range(max_retries):
                    try:
                        api_params = {
                            "model": rag_instance.tool_llm_model_name,
                            "messages": [
                                {
                                    "role": "system",
                                    "content": PROMPTS["entity_type_suggestion_system"],
                                },
                                {
                                    "role": "user",
                                    "content": PROMPTS["entity_type_suggestion_user"].format(
                                        current_entity_types=current_entity_types_json,
                                        file_content=chunk_content
                                    ),
                                },
                            ],
                            **llm_kwargs
                        }

                        response = client.chat.completions.create(**api_params)

                        # Track tokens
                        if token_tracker and hasattr(response, 'usage') and response.usage:
                            token_counts = {
                                'prompt_tokens': getattr(response.usage, 'prompt_tokens', 0),
                                'completion_tokens': getattr(response.usage, 'completion_tokens', 0),
                                'total_tokens': getattr(response.usage, 'total_tokens', 0),
                            }
                            token_tracker.add_usage(token_counts)

                        llm_response = response.choices[0].message.content
                        suggested_types = _extract_json_callback(llm_response)

                        if suggested_types and isinstance(suggested_types, list):
                            return suggested_types
                        else:
                            continue

                    except Exception as e:
                        logger.warning(f"Attempt {attempt + 1}/{max_retries}: Error processing chunk: {e}")
                        if attempt < max_retries - 1:
                            continue
                        else:
                            return []

                return []

        # Process chunks in batches of REFINE_INTERVAL
        total_chunks = len(chunk_pairs)
        processed_count = 0

        for batch_start in range(0, total_chunks, REFINE_INTERVAL):
            batch_end = min(batch_start + REFINE_INTERVAL, total_chunks)
            batch = chunk_pairs[batch_start:batch_end]
            batch_size = len(batch)

            logger.info(f"Processing batch {batch_start + 1}-{batch_end}/{total_chunks} ({batch_size} chunks)")

            # Create tasks for current batch using current_entity_types
            tasks = [
                process_chunk_with_limit(chunk_id, chunk_content, current_entity_types)
                for chunk_id, chunk_content in batch
            ]

            # Execute batch in parallel
            results = await asyncio.gather(*tasks)

            # Collect results from this batch
            for new_entity_types in results:
                all_new_entity_types.extend(new_entity_types)

            processed_count += batch_size
            logger.info(f"Completed {processed_count}/{total_chunks} chunks")

            # Perform refinement after each batch (except for the last batch if it will be refined at the end)
            if batch_end < total_chunks or batch_end == total_chunks:
                logger.info(f"🔄 Performing refinement after {processed_count} chunks...")

                # Combine current entity types with new suggestions
                combined_entity_types = current_entity_types + all_new_entity_types
                logger.debug(f"Combined {len(current_entity_types)} existing + {len(all_new_entity_types)} new = {len(combined_entity_types)} total")

                # Refine to remove duplicates
                refined_entity_types = await refine_entity_types(
                    combined_entity_types,
                    rag_instance.tool_llm_model_name,
                    _extract_json_callback,
                    rag_instance.openai_client,  # 傳入共用的 OpenAI client
                    token_tracker,  # 傳入 token tracker
                    tool_llm_model_kwargs  # 傳入 tool LLM kwargs
                )

                # Update current_entity_types for next batch
                current_entity_types = refined_entity_types
                all_new_entity_types = []  # Clear accumulated suggestions

                # Save intermediate results
                save_entity_types(rag_instance.working_dir, refined_entity_types)

                logger.info(f"✅ Refinement completed. Current entity types: {len(refined_entity_types)}")

        # Use the final refined entity types
        refined_entity_types = current_entity_types
        logger.info(f"All chunks processed ({total_chunks} chunks from {len(file_text_pairs)} documents). Final entity types: {len(refined_entity_types)}")

        # Step 6: Save final entity types (already saved during periodic refinement, but ensure final state)
        save_entity_types(rag_instance.working_dir, refined_entity_types)

        # Step 7: Update processing status
        update_processing_status(rag_instance.working_dir, file_text_pairs)

        # Step 8: Reload entity types into addon_params for immediate use
        update_entity_types_from_working_dir(rag_instance)

        logger.info(f"Entity type discovery completed: {len(refined_entity_types)} total entity types")
        logger.info("Entity types have been automatically reloaded for immediate use")

        # Calculate initial entity count for return value
        initial_entity_types_count = len(load_existing_entity_types(rag_instance.working_dir))

        return {
            "status": "success",
            "message": "Entity type discovery completed successfully",
            "initial_entity_types": initial_entity_types_count,
            "final_entity_types": len(refined_entity_types),
            "new_entity_types": len(refined_entity_types) - initial_entity_types_count,
            "documents_processed": len(file_text_pairs),
            "chunks_processed": total_chunks
        }

    except Exception as e:
        logger.error(f"Entity type discovery failed: {e}")
        raise
