# index_trace_token.py
import os
import asyncio
from pathlib import Path
from typing import List, Tuple
from lightrag import LightRAG
from lightrag.llm.openai import gpt_4o_mini_complete, gpt_4o_complete, openai_embed
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import setup_logger, EmbeddingFunc, TokenTracker
import config
import nest_asyncio
nest_asyncio.apply()
import time

setup_logger("lightrag", level="INFO")

if not os.path.exists(os.environ["WORKING_DIR"]):
    os.mkdir(os.environ["WORKING_DIR"])

# ============================================================
# Global Token Tracker Instance
# ============================================================

# 全局 token tracker 實例
global_token_tracker = TokenTracker()


def wrap_llm_func_with_token_tracker(original_llm_func, token_tracker):
    """包裝 LLM 函數，自動添加 token_tracker 參數

    Args:
        original_llm_func: 原始的 LLM 函數
        token_tracker: TokenTracker 實例

    Returns:
        包裝後的 LLM 函數
    """
    async def wrapper(*args, **kwargs):
        # 自動添加 token_tracker 到 kwargs
        kwargs['token_tracker'] = token_tracker
        result = await original_llm_func(*args, **kwargs)
        return result

    return wrapper


def extract_txt_text(path: Path) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read().strip()


def process_files_to_list(folder_path: str) -> Tuple[List[str], List[str], List[str]]:
    texts = []
    file_names = []
    fiscal_years = []

    for file_name in os.listdir(folder_path):
        file_path = Path(folder_path) / file_name
        if file_path.suffix.lower() != ".txt":
            print(f"Skip (not TXT): {file_name}")
            continue
        try:
            text_content = extract_txt_text(file_path)
            if text_content:
                texts.append(text_content)
                file_names.append(file_name)

                # 從檔名抓出 "2025 q3"
                # 假設檔名格式像 "apple 2025 q3.txt"
                parts = file_name.replace(".txt", "").split()
                fiscal = None
                for i in range(len(parts) - 1):
                    if parts[i].isdigit() and parts[i+1].lower().startswith("q"):
                        fiscal = f"Fiscal Year {parts[i]} {parts[i+1].lower()}"
                        break
                fiscal_years.append(fiscal if fiscal else "Unknown")

                print(f"Success: {file_name}")
            else:
                print(f"Empty content: {file_name}")
        except Exception as e:
            print(f"Fail: {file_name}, cause: {e}")

    return texts, file_names, fiscal_years


async def initialize_rag():
    """初始化 RAG，使用包裝後的 LLM 函數"""

    # 包裝 LLM 函數以自動傳遞 token_tracker
    tracked_llm_func = wrap_llm_func_with_token_tracker(
        gpt_4o_mini_complete,
        global_token_tracker
    )

    rag = LightRAG(
        working_dir=os.environ["WORKING_DIR"],
        embedding_func=openai_embed,
        llm_model_func=tracked_llm_func,  # 使用包裝後的函數
        chunk_token_size=600,
        chunk_overlap_token_size=100,
        llm_model_max_async=32,
        enable_node_embedding=True,
        enable_llm_cache=False,
        max_parallel_insert=8
    )

    # IMPORTANT: Both initialization calls are required!
    await rag.initialize_storages()  # Initialize storage backends
    await initialize_pipeline_status()  # Initialize processing pipeline

    # 將 token_tracker 添加到 LightRAG 實例
    rag.token_tracker = global_token_tracker
    print(f"[Debug] Added token_tracker to RAG instance: {rag.token_tracker}")

    return rag


def print_phase_statistics(phase_name: str, start_usage: dict, end_usage: dict):
    """打印單一階段的 token 統計

    Args:
        phase_name: 階段名稱
        start_usage: 階段開始時的 token 使用量
        end_usage: 階段結束時的 token 使用量
    """
    phase_calls = end_usage['call_count'] - start_usage['call_count']
    phase_prompt = end_usage['prompt_tokens'] - start_usage['prompt_tokens']
    phase_completion = end_usage['completion_tokens'] - start_usage['completion_tokens']
    phase_total = end_usage['total_tokens'] - start_usage['total_tokens']

    # 估算成本 (基於 GPT-4o-mini 價格: $0.15/1M input, $0.60/1M output)
    input_cost = phase_prompt * 0.15 / 1_000_000
    output_cost = phase_completion * 0.60 / 1_000_000
    total_cost = input_cost + output_cost

    print(f"\n{'='*80}")
    print(f"{phase_name} - TOKEN USAGE STATISTICS")
    print(f"{'='*80}")
    print(f"📊 LLM Calls:            {phase_calls}")
    print(f"📝 Prompt Tokens:        {phase_prompt:,}")
    print(f"✍️  Completion Tokens:    {phase_completion:,}")
    print(f"🔢 Total Tokens:         {phase_total:,}")
    print(f"{'-'*80}")
    print(f"💰 Estimated Cost (GPT-4o-mini):")
    print(f"   Input:  ${input_cost:.6f}")
    print(f"   Output: ${output_cost:.6f}")
    print(f"   Total:  ${total_cost:.6f}")
    print(f"{'='*80}")


async def main():
    rag = None
    try:
        print("=" * 80)
        print("Starting Indexing with Token Tracking")
        print("=" * 80)
        print("✓ Tracking: Phase 1 - Entity Type Augmentation")
        print("✓ Tracking: Phase 2 - Entity Extraction")
        print("✓ Tracking: Phase 3 - Agentic Merging")
        print("=" * 80)

        # 重置 token tracker
        global_token_tracker.reset()

        # 初始化 RAG
        rag = await initialize_rag()
        print("✓ Initialization success!\n")

        # 讀取檔案
        path = "./inputs"
        texts, file_paths, fiscal_years = process_files_to_list(path)

        if not texts:
            print("No valid TXT content found, indexing aborted.")
            return

        # ============================================================
        # Phase 1: Entity Type Augmentation
        # ============================================================
        print("\n" + "=" * 80)
        print("PHASE 1: Entity Type Augmentation")
        print("=" * 80)

        phase1_start = global_token_tracker.get_usage()
        start_time = time.time()

        rag.entity_type_aug(path)

        phase1_end = global_token_tracker.get_usage()
        phase1_time = time.time() - start_time

        print(f"\n✓ Phase 1 completed in {phase1_time:.2f} seconds")
        print_phase_statistics("Phase 1: Entity Type Augmentation", phase1_start, phase1_end)

        # ============================================================
        # Phase 2 & 3: Entity Extraction + Agentic Merging
        # ============================================================
        print("\n" + "=" * 80)
        print("PHASE 2 & 3: Entity Extraction + Agentic Merging")
        print("=" * 80)

        phase2_start = global_token_tracker.get_usage()
        start_time = time.time()

        rag.insert(
            texts,
            file_paths=file_paths,
            timestamps=fiscal_years,
            agentic_merging=True,
            agentic_merging_threshold=0.7
        )

        phase23_end = global_token_tracker.get_usage()
        phase23_time = time.time() - start_time

        print(f"\n✓ Phase 2 & 3 completed in {phase23_time:.2f} seconds")
        print_phase_statistics("Phase 2 & 3: Entity Extraction + Agentic Merging", phase2_start, phase23_end)

        # ============================================================
        # Final Statistics
        # ============================================================
        print("\n" + "=" * 80)
        print("FINAL CONSOLIDATED STATISTICS")
        print("=" * 80)

        final_usage = global_token_tracker.get_usage()
        print(f"📊 Total LLM Calls:       {final_usage['call_count']}")
        print(f"📝 Prompt Tokens:         {final_usage['prompt_tokens']:,}")
        print(f"✍️  Completion Tokens:     {final_usage['completion_tokens']:,}")
        print(f"🔢 Total Tokens:          {final_usage['total_tokens']:,}")
        print("-" * 80)

        print("\n✓ Graph building success!")

    except Exception as e:
        print(f"\n❌ Graph building failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if rag:
            await rag.finalize_storages()
            print("\n✓ Storage finalized")


if __name__ == "__main__":
    asyncio.run(main())
