from lightrag import QueryParam
import config
import os
from lightrag import LightRAG
from lightrag.llm.openai import gpt_5_complete, openai_embed, gpt_4o_mini_complete, gpt_5_mini_complete, gpt_4o_complete
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import setup_logger, EmbeddingFunc, TokenTracker

import nest_asyncio
nest_asyncio.apply()

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
        tool_llm_model_name="gpt-4o-mini",
        chunk_token_size=600,
        chunk_overlap_token_size=100,
        enable_node_embedding=True,
        enable_llm_cache=False,
        llm_model_max_async=8,
    )

    # IMPORTANT: Both initialization calls are required!
    await rag.initialize_storages()  # Initialize storage backends
    await initialize_pipeline_status()  # Initialize processing pipeline

    # 將 token_tracker 添加到 LightRAG 實例
    # global_config 是從 asdict(self) 創建的，所以需要添加到實例屬性
    rag.token_tracker = global_token_tracker
    print(f"[Debug] Added token_tracker to RAG instance: {rag.token_tracker}")

    return rag


async def main():
    try:
        print("=" * 80)
        print("Starting Query with Token Tracking")
        print("=" * 80)
        print("✓ Tracking: Main LLM calls (query generation, entity extraction)")
        print("✓ Tracking: Recognition LLM calls (entity/relation filtering)")
        print("")
        print("Note: Token count depends on:")
        print("  - Query complexity")
        print("  - Number of entities/relations retrieved")
        print("  - Whether recognition is triggered (needs entities to filter)")
        print("=" * 80)

        # 重置 token tracker
        global_token_tracker.reset()

        # 初始化 RAG object
        rag = None
        rag = await initialize_rag()
        print("✓ Initialization success!!\n")

        query = """
我也想詢問 Apple 主席對於關稅的議題是否有變化過?
用繁體中文回答我
        """

        print(f"Query: {query}\n")
        print("-" * 80)
        print("Executing query with token tracking...")
        print("-" * 80)

        response, context = rag.query(
            query,
            param=QueryParam(
                mode="hybrid",
                # conversation_history
                history_turns=0,
                top_k=10,
                chunk_top_k=5,
                max_total_tokens=120000,
                max_entity_tokens=30000,
                max_relation_tokens=30000,
                max_hop=2,
                top_neighbors=5,
                multi_hop_relevance_threshold=0.40,
                enable_rerank=False,
                top_ppr_nodes=20,
                top_fastrp_nodes=10,
                enable_recognition=True,
                recognition_batch_size=50,
                response_type="Single Paragraph",
                only_need_context=False,
                user_prompt="""
                            You have to answer the question following the format below:
                            ## <Title of the report>
                            ### Overview
                            <description: Provide a high-level summary, including scope, purpose, and context.>

                            ### Key Themes
                            <description: Extract the main recurring themes, announcements, priorities, challenges, and opportunities.>

                            ### Comparative Insights
                            <description: Highlight similarities, differences, shifts in tone, and evolving trends.>

                            ### Actionable Insights
                            <description: Summarize practical implications or recommendations (e.g., for strategy, investment, or risk management).>

                            ### Reference
                            <description: List ALL sources used>"""
            ),
        )

        print("\n" + "=" * 80)
        print("RESPONSE")
        print("=" * 80)
        print(response)
        print("=" * 80)

        # 顯示 token 使用統計
        print("\n" + "=" * 80)
        print("TOKEN USAGE STATISTICS")
        print("=" * 80)
        usage = global_token_tracker.get_usage()
        print(f"📊 Total LLM Calls:       {usage['call_count']}")
        print(f"📝 Prompt Tokens:         {usage['prompt_tokens']:,}")
        print(f"✍️  Completion Tokens:     {usage['completion_tokens']:,}")
        print(f"🔢 Total Tokens:          {usage['total_tokens']:,}")
        print("-" * 80)

    except Exception as e:
        print(f"\n❌ Failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if rag:
            await rag.finalize_storages()
            print("\n✓ Storage finalized")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
