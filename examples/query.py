
from lightrag import QueryParam
import config
import os
from lightrag import  LightRAG
from lightrag.llm.openai import gpt_5_complete, openai_embed, gpt_4o_mini_complete, gpt_5_mini_complete, gpt_4o_complete, gpt_5_mini_complete
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import setup_logger, EmbeddingFunc

import nest_asyncio
nest_asyncio.apply()

setup_logger("lightrag", level="INFO")

if not os.path.exists(os.environ["WORKING_DIR"]):
    os.mkdir(os.environ["WORKING_DIR"])

async def initialize_rag():
    rag =  LightRAG(
        working_dir=os.environ["WORKING_DIR"],
        embedding_func=openai_embed,
        llm_model_func=gpt_4o_mini_complete,
        #llm_model_kwargs={"reasoning_effort": "low"},
        tool_llm_model_name= "gpt-4o-mini",
        #tool_llm_model_kwargs={"reasoning_effort": "low"},
        chunk_token_size=600,
        chunk_overlap_token_size=100,
        enable_node_embedding=True,
        enable_llm_cache= False,
        llm_model_max_async=8,
    )
    
    # IMPORTANT: Both initialization calls are required!
    await rag.initialize_storages()  # Initialize storage backends
    await initialize_pipeline_status()  # Initialize processing pipeline
    return rag

async def main():
    try:
        # 初始化 RAG object
        rag = None
        rag = await initialize_rag()
        print("Initialization success!!")

        query = """
我想詢問 Apple 主席對於關稅的議題是否有變化過?
Apple 在這幾季的趨勢變化，有什麼值得可以注意的警訊?
用繁體中文回答我

        """

        # 將時間相關模糊問題進行 rewrite
        #rewritten_query = rewriter(query)
        print("query: ", query)

        response,context = rag.query(
            query,
            param=QueryParam(
                mode="hybrid",
                # conversation_history
                history_turns=0,
                top_k=10,
                chunk_top_k= 5,
                max_total_tokens=120000,
                max_entity_tokens=30000,
                max_relation_tokens=30000,
                max_hop=2,
                top_neighbors= 5,
                multi_hop_relevance_threshold=0.4,
                enable_rerank=False,
                top_ppr_nodes=10,       
                top_fastrp_nodes=5,     
                enable_recognition=True,
                recognition_batch_size=30,
                response_type="Single Paragraph",
            ),
        )

        print("===============================")
        print(response)
        print("===============================")
        # print("Context: ", context)
        

    except Exception as e:
        print(f"Fail: {e}")
    finally:
        if rag:
            await rag.finalize_storages()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
"""
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
<description: List ALL sources used>
"""


"""
台積電在民國 113Q1、114Q1的其他流動資產和資產總計金額是多少? 可以針對報表進行分析其它資訊嗎?
台積電在這段時間的表現相較於聯發科呢? 
是否有資料顯示近期蘋果對於台積電的投資或合作? 
另外台積電和輝達近期是否有任何潛在合作嗎?
我也想詢問 Apple 主席對於關稅的議題是否有變化過?
Apple 在這幾季的趨勢變化，有什麼值得可以注意的警訊?
用繁體中文回答我
        """