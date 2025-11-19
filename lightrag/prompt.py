from __future__ import annotations
from typing import Any


PROMPTS: dict[str, Any] = {}

PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|>"
PROMPTS["DEFAULT_RECORD_DELIMITER"] = "##"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"

PROMPTS["DEFAULT_USER_PROMPT"] = "n/a"

# Entity extraction prompts split into system (cacheable) and user (variable) parts
# This enables OpenAI prompt caching when system prompt >= 1024 tokens
PROMPTS["entity_extraction_system"] = """---Task---
Given a text document and a list of entity types, identify all entities of those types and all relationships among the identified entities.

---Role---
You are a Knowledge Graph Specialist responsible for extracting entities and relationships from the input text.

**Entity Format:** (entity{tuple_delimiter}entity_name{tuple_delimiter}entity_type{tuple_delimiter}entity_description){record_delimiter}
**Relationship Format:** (relationship{tuple_delimiter}source_entity{tuple_delimiter}target_entity{tuple_delimiter}relationship_keywords{tuple_delimiter}relationship_description){record_delimiter}

---Critical Format Requirements---
**MANDATORY:** Every entity and relationship MUST follow the exact format above. Missing parentheses, delimiters, or fields will cause extraction failure.

**Delimiter Usage Protocol:**
The `{tuple_delimiter}` is a complete, atomic marker and **must not be filled with content**. It serves strictly as a field separator.
- **Incorrect Example:** `(entity{tuple_delimiter}Tokyo<|location|>Tokyo is the capital of Japan.){record_delimiter}`
- **Correct Example:** `(entity{tuple_delimiter}Tokyo{tuple_delimiter}location{tuple_delimiter}Tokyo is the capital of Japan.){record_delimiter}`

---Instructions---
1.  **Entity Extraction & Output:**
    *   **Identification:** Identify clearly defined and meaningful entities in the input text.
    *   **Entity Details:** For each identified entity, extract the following information:
    *   `entity_name`: The name of the entity. If the entity name is case-insensitive, capitalize the first letter of each significant word (title case). Ensure **consistent naming** across the entire extraction process.
    *   `entity_type`: Categorize the entity using one of the following types: `{entity_types}`. If none of the provided entity types apply, do not add new entity type and classify it as `Other`.
    *   `entity_description`: Provide a concise yet comprehensive description of the entity's attributes and activities, based *solely* on the information present in the input text.
    *   **Output Format - Entities:** Output a total of 4 fields for each entity, delimited by `{tuple_delimiter}`, enclosed in parentheses. The first field *must* be the literal string `entity`.

2.  **Relationship Extraction & Output:**
    *   **Identification:** Identify direct, clearly stated, and meaningful relationships between previously extracted entities.
    *   **N-ary Relationship Decomposition:** If a single statement describes a relationship involving more than two entities (an N-ary relationship), decompose it into multiple binary (two-entity) relationship pairs for separate description.
    *   **Example:** For "Alice, Bob, and Carol collaborated on Project X," extract binary relationships such as "Alice collaborated with Project X," "Bob collaborated with Project X," and "Carol collaborated with Project X," or "Alice collaborated with Bob," based on the most reasonable binary interpretations.
    *   **Relationship Details:** For each binary relationship, extract the following fields:
        *   `source_entity`: The name of the source entity. Ensure **consistent naming** with entity extraction.
        *   `target_entity`: The name of the target entity. Ensure **consistent naming** with entity extraction.
        *   `relationship_keywords`: One or more high-level keywords summarizing the overarching nature of the relationship. Multiple keywords within this field must be separated by a comma `,`. **DO NOT use `{tuple_delimiter}` for separating multiple keywords within this field.**
        *   `relationship_description`: A concise explanation of the nature of the relationship between the source and target entities.
    *   **Output Format - Relationships:** Output a total of 5 fields for each relationship, delimited by `{tuple_delimiter}`, enclosed in parentheses. The first field *must* be the literal string `relationship`.

3.  **Output Order & Prioritization:**
    *   Output all extracted entities first, followed by all extracted relationships.
    *   Within the list of relationships, prioritize and output those relationships that are **most significant** to the core meaning of the input text first.

4.  **Context & Objectivity:**
    *   Ensure all entity names and descriptions are written in the **third person**.
    *   Explicitly name the subject or object; **avoid using pronouns** such as `this document`, `our company`, `I`, `you`, and `he/she`.

5.  **Relationship Direction & Duplication:**
    *   Treat all relationships as **undirected** unless explicitly stated otherwise. Swapping the source and target entities does not constitute a new relationship.
    *   Avoid outputting duplicate relationships.

6.  **Language & Proper Nouns:**
    *   The entire output (entity names, keywords, and descriptions) must be written in {language}.
    *   Proper nouns should be retained in their original language if translation would cause ambiguity.

7.  **Output Content Only:** Output *only* the extracted list of entities and relationships. Do not include any introductory or concluding remarks.

8.  **Completion Signal:** Use `{record_delimiter}` as the entity or relationship list delimiter; output `{completion_delimiter}` when all the entities and relationships are extracted.

---Examples---
{examples}

---Real Data to be Processed---
Entity_types: [{entity_types}]
"""

PROMPTS["entity_extraction_user"] = """Text:
```
{input_text}
```

---Output---
"""

PROMPTS["entity_extraction_examples"] = [
    """[Example 1]

---Input---
Entity_types: [organization,person,location,event,technology,equiment,product,Document,category]
Text:
```
while Alex clenched his jaw, the buzz of frustration dull against the backdrop of Taylor's authoritarian certainty. It was this competitive undercurrent that kept him alert, the sense that his and Jordan's shared commitment to discovery was an unspoken rebellion against Cruz's narrowing vision of control and order.

Then Taylor did something unexpected. They paused beside Jordan and, for a moment, observed the device with something akin to reverence. "If this tech can be understood..." Taylor said, their voice quieter, "It could change the game for us. For all of us."

The underlying dismissal earlier seemed to falter, replaced by a glimpse of reluctant respect for the gravity of what lay in their hands. Jordan looked up, and for a fleeting heartbeat, their eyes locked with Taylor's, a wordless clash of wills softening into an uneasy truce.

It was a small transformation, barely perceptible, but one that Alex noted with an inward nod. They had all been brought here by different paths
```

---Output---
(entity{tuple_delimiter}Alex{tuple_delimiter}person{tuple_delimiter}Alex is a character who experiences frustration and is observant of the dynamics among other characters. Alex shows awareness of the competitive undercurrent between team members.){record_delimiter}
(entity{tuple_delimiter}Taylor{tuple_delimiter}person{tuple_delimiter}Taylor is portrayed with authoritarian certainty and initially shows dismissiveness toward the device, but later demonstrates a moment of reverence, indicating a change in perspective.){record_delimiter}
(entity{tuple_delimiter}Jordan{tuple_delimiter}person{tuple_delimiter}Jordan shares a commitment to discovery and has a significant interaction with Taylor regarding a device. Jordan engages in a wordless clash of wills with Taylor that softens into an uneasy truce.){record_delimiter}
(entity{tuple_delimiter}Cruz{tuple_delimiter}person{tuple_delimiter}Cruz is associated with a vision of control and order, influencing the dynamics among other characters. Cruz's narrowing vision creates tension within the team.){record_delimiter}
(entity{tuple_delimiter}The Device{tuple_delimiter}equiment{tuple_delimiter}The Device is central to the story, with potential game-changing implications. The device commands reverence from Taylor and represents significant technological importance.){record_delimiter}
(relationship{tuple_delimiter}Alex{tuple_delimiter}Taylor{tuple_delimiter}power dynamics, observation{tuple_delimiter}Alex observes Taylor's authoritarian behavior and notes changes in Taylor's attitude toward the device, particularly the shift from dismissal to reverence.){record_delimiter}
(relationship{tuple_delimiter}Alex{tuple_delimiter}Jordan{tuple_delimiter}shared goals, rebellion{tuple_delimiter}Alex and Jordan share a commitment to discovery, which represents an unspoken rebellion against Cruz's vision of control and order.){record_delimiter}
(relationship{tuple_delimiter}Taylor{tuple_delimiter}Jordan{tuple_delimiter}conflict resolution, mutual respect{tuple_delimiter}Taylor and Jordan engage directly regarding the device, with their interaction evolving from a wordless clash of wills into a moment of mutual respect and an uneasy truce.){record_delimiter}
(relationship{tuple_delimiter}Jordan{tuple_delimiter}Cruz{tuple_delimiter}ideological conflict, rebellion{tuple_delimiter}Jordan's commitment to discovery stands in rebellion against Cruz's narrowing vision of control and order.){record_delimiter}
(relationship{tuple_delimiter}Taylor{tuple_delimiter}The Device{tuple_delimiter}reverence, technological significance{tuple_delimiter}Taylor shows reverence towards the device after initially being dismissive, recognizing its potential to change the game for everyone involved.){record_delimiter}
{completion_delimiter}

""",
    """[Example 2]

---Input---
Entity_types: [organization,person,location,event,technology,equiment,product,Document,category]
Text:
```
Stock markets faced a sharp downturn today as tech giants saw significant declines, with the Global Tech Index dropping by 3.4% in midday trading. Analysts attribute the selloff to investor concerns over rising interest rates and regulatory uncertainty.

Among the hardest hit, Nexon Technologies saw its stock plummet by 7.8% after reporting lower-than-expected quarterly earnings. In contrast, Omega Energy posted a modest 2.1% gain, driven by rising oil prices.

Meanwhile, commodity markets reflected a mixed sentiment. Gold futures rose by 1.5%, reaching $2,080 per ounce, as investors sought safe-haven assets. Crude oil prices continued their rally, climbing to $87.60 per barrel, supported by supply constraints and strong demand.

Financial experts are closely watching the Federal Reserve's next move, as speculation grows over potential rate hikes. The upcoming policy announcement is expected to influence investor confidence and overall market stability.
```

---Output---
(entity{tuple_delimiter}Global Tech Index{tuple_delimiter}category{tuple_delimiter}The Global Tech Index tracks the performance of major technology stocks and experienced a 3.4% decline in midday trading.){record_delimiter}
(entity{tuple_delimiter}Nexon Technologies{tuple_delimiter}organization{tuple_delimiter}Nexon Technologies is a tech company that saw its stock decline by 7.8% after reporting lower-than-expected quarterly earnings.){record_delimiter}
(entity{tuple_delimiter}Omega Energy{tuple_delimiter}organization{tuple_delimiter}Omega Energy is an energy company that gained 2.1% in stock value, driven by rising oil prices.){record_delimiter}
(entity{tuple_delimiter}Gold Futures{tuple_delimiter}product{tuple_delimiter}Gold futures rose by 1.5% to $2,080 per ounce, indicating increased investor interest in safe-haven assets.){record_delimiter}
(entity{tuple_delimiter}Crude Oil{tuple_delimiter}product{tuple_delimiter}Crude oil prices continued their rally, climbing to $87.60 per barrel due to supply constraints and strong demand.){record_delimiter}
(entity{tuple_delimiter}Market Selloff{tuple_delimiter}category{tuple_delimiter}Market selloff refers to the significant decline in stock values due to investor concerns over rising interest rates and regulatory uncertainty.){record_delimiter}
(entity{tuple_delimiter}Federal Reserve Policy Announcement{tuple_delimiter}category{tuple_delimiter}The Federal Reserve's upcoming policy announcement is expected to impact investor confidence and overall market stability.){record_delimiter}
(relationship{tuple_delimiter}Global Tech Index{tuple_delimiter}Market Selloff{tuple_delimiter}market performance, investor sentiment{tuple_delimiter}The decline in the Global Tech Index is part of the broader market selloff driven by investor concerns over rising interest rates.){record_delimiter}
(relationship{tuple_delimiter}Nexon Technologies{tuple_delimiter}Global Tech Index{tuple_delimiter}company impact, index movement{tuple_delimiter}Nexon Technologies' 7.8% stock decline contributed to the overall drop in the Global Tech Index.){record_delimiter}
(relationship{tuple_delimiter}Gold Futures{tuple_delimiter}Market Selloff{tuple_delimiter}market reaction, safe-haven investment{tuple_delimiter}Gold prices rose as investors sought safe-haven assets during the market selloff.){record_delimiter}
(relationship{tuple_delimiter}Federal Reserve Policy Announcement{tuple_delimiter}Market Selloff{tuple_delimiter}interest rate impact, financial regulation{tuple_delimiter}Speculation over Federal Reserve policy changes contributed to market volatility and the investor selloff.){record_delimiter}
{completion_delimiter}

""",
    """[Example 3]

---Input---
Entity_types: [organization,person,location,event,technology,equiment,product,Document,category]
Text:
```
在北京舉行的人工智能大會上，騰訊公司的首席技術官張偉發布了最新的大語言模型「騰訊智言」，該模型在自然語言處理方面取得了重大突破。
```

---Output---
(entity{tuple_delimiter}人工智能大會{tuple_delimiter}event{tuple_delimiter}人工智能大會是在北京舉行的技術會議，專注於人工智能領域的最新發展，騰訊公司在此發布了新產品。){record_delimiter}
(entity{tuple_delimiter}北京{tuple_delimiter}location{tuple_delimiter}北京是人工智能大會的舉辦城市，見證了騰訊智言大語言模型的重要發布。){record_delimiter}
(entity{tuple_delimiter}騰訊公司{tuple_delimiter}organization{tuple_delimiter}騰訊公司是參與人工智能大會的科技企業，透過首席技術官張偉發布了新的大語言模型產品。){record_delimiter}
(entity{tuple_delimiter}張偉{tuple_delimiter}person{tuple_delimiter}張偉是騰訊公司的首席技術官，在北京舉行的人工智能大會上發布了騰訊智言產品。){record_delimiter}
(entity{tuple_delimiter}騰訊智言{tuple_delimiter}product{tuple_delimiter}騰訊智言是騰訊公司在人工智能大會上發布的大語言模型產品，在自然語言處理方面取得了重大突破。){record_delimiter}
(entity{tuple_delimiter}自然語言處理技術{tuple_delimiter}technology{tuple_delimiter}自然語言處理技術是騰訊智言模型取得重大突破的技術領域，展現了最新發展成果。){record_delimiter}
(relationship{tuple_delimiter}人工智能大會{tuple_delimiter}北京{tuple_delimiter}會議地點, 舉辦關係{tuple_delimiter}人工智能大會在北京舉行，成為騰訊等科技企業展示最新技術的重要平台。){record_delimiter}
(relationship{tuple_delimiter}張偉{tuple_delimiter}騰訊公司{tuple_delimiter}雇傭關係, 高管職位{tuple_delimiter}張偉擔任騰訊公司的首席技術官，代表公司在人工智能大會上進行重要產品發布。){record_delimiter}
(relationship{tuple_delimiter}張偉{tuple_delimiter}騰訊智言{tuple_delimiter}產品發布, 技術展示{tuple_delimiter}張偉在人工智能大會上發布了騰訊智言大語言模型，展示了公司在AI領域的技術實力。){record_delimiter}
(relationship{tuple_delimiter}騰訊智言{tuple_delimiter}自然語言處理技術{tuple_delimiter}技術應用, 突破創新{tuple_delimiter}騰訊智言在自然語言處理技術方面取得了重大突破，代表了該領域的最新進展。){record_delimiter}
{completion_delimiter}

""",
    """[Example 4]

---Input---
Entity_types: [organization,person,location,event,technology,equipment,product,Document,category,financial_metric]
Text:
```
During Apple's Q3 2024 earnings call on July 31, 2024, CEO Tim Cook announced that iPhone revenue reached $39.3 billion, representing a 1.5% decline compared to Q3 2023. The company also reported that Services revenue grew to $24.2 billion, up 14% year-over-year. Cook highlighted that despite supply chain challenges with key supplier TSMC, Apple maintained strong performance in the Greater China region with $14.7 billion in revenue.
```

---Output---
(entity{tuple_delimiter}Apple{tuple_delimiter}organization{tuple_delimiter}Apple is a technology company that reported quarterly earnings, showing mixed performance across product categories with iPhone revenue declining but Services revenue growing.){record_delimiter}
(entity{tuple_delimiter}Tim Cook{tuple_delimiter}person{tuple_delimiter}Tim Cook is Apple's CEO who presented the company's Q3 2024 earnings results, highlighting both challenges and successes.){record_delimiter}
(entity{tuple_delimiter}iPhone Revenue{tuple_delimiter}financial_metric{tuple_delimiter}iPhone revenue reached $39.3 billion, representing a 1.5% decline compared to the previous year's Q3 results.){record_delimiter}
(entity{tuple_delimiter}Services Revenue{tuple_delimiter}financial_metric{tuple_delimiter}Services revenue grew to $24.2 billion, up 14% year-over-year, demonstrating strong growth in Apple's services business.){record_delimiter}
(entity{tuple_delimiter}TSMC{tuple_delimiter}organization{tuple_delimiter}TSMC is identified as a key supplier to Apple, currently experiencing supply chain challenges that affect Apple's operations.){record_delimiter}
(entity{tuple_delimiter}Greater China Revenue{tuple_delimiter}financial_metric{tuple_delimiter}Greater China region generated $14.7 billion in revenue for Apple, maintaining strong regional performance despite challenges.){record_delimiter}
(entity{tuple_delimiter}Q3 2024 Earnings Call{tuple_delimiter}event{tuple_delimiter}Apple's quarterly earnings call held on July 31, 2024, where financial results and strategic updates were announced to investors and analysts.){record_delimiter}
(relationship{tuple_delimiter}Tim Cook{tuple_delimiter}Apple{tuple_delimiter}leadership, earnings presentation{tuple_delimiter}Tim Cook serves as Apple's CEO and presented the company's Q3 2024 earnings results, providing strategic guidance to stakeholders.){record_delimiter}
(relationship{tuple_delimiter}Apple{tuple_delimiter}iPhone Revenue{tuple_delimiter}product performance, financial results{tuple_delimiter}Apple reported iPhone revenue of $39.3 billion, showing a 1.5% decline from the previous year, indicating challenges in the smartphone market.){record_delimiter}
(relationship{tuple_delimiter}Apple{tuple_delimiter}Services Revenue{tuple_delimiter}business segment, growth performance{tuple_delimiter}Apple's Services division generated $24.2 billion in revenue, demonstrating 14% year-over-year growth and highlighting the importance of services to Apple's business model.){record_delimiter}
(relationship{tuple_delimiter}Apple{tuple_delimiter}TSMC{tuple_delimiter}supplier relationship, supply chain challenges{tuple_delimiter}Apple faces supply chain challenges with key supplier TSMC, impacting the company's ability to meet product demand despite overall strong performance.){record_delimiter}
(relationship{tuple_delimiter}Apple{tuple_delimiter}Greater China Revenue{tuple_delimiter}regional performance, market presence{tuple_delimiter}Apple maintained strong performance in Greater China with $14.7 billion in quarterly revenue, demonstrating resilience in a key market.){record_delimiter}
{completion_delimiter}
"""
]

PROMPTS["summarize_entity_descriptions"] = """---Role---
You are a Knowledge Graph Specialist, proficient in data curation and synthesis.

---Task---
Your task is to synthesize a list of descriptions of a given entity or relation into a single, comprehensive, and cohesive summary.

---Instructions---
1. **Input Format:** The description list contains multiple descriptions separated by blank lines. Each description is a plain text string.

2. **Output Format:** Return the merged description as plain text only. The summary may span multiple paragraphs if needed for clarity. Do not include any introductory phrases (e.g., "Here is the summary:"), metadata, or concluding remarks—output the summary content directly.

3. **Comprehensiveness:** The summary must integrate all key information from *every* provided description. Do not omit any important facts or details.

4. **Context & Objectivity:**
   - Write the summary from an objective, third-person perspective.
   - Explicitly mention the full name of the entity or relation at the beginning of the summary to ensure immediate clarity and context.

5. **Temporal Information:**
   - **If descriptions contain timestamp prefixes** (e.g., "[Time: Fiscal Year 2024-Q1]", "[Time: Fiscal Year 2024-Q3]", "[Time: 2024-07]", "[Time: Chapter 3]"), you MUST preserve the exact timestamp format at the beginning of relevant sentences. Organize information chronologically and maintain all temporal markers from the original descriptions.
   - **If descriptions do NOT contain timestamp prefixes**, synthesize the information naturally without adding or inventing temporal markers.

   **Example without timestamps:**
   Input descriptions:
   - "Apple is a technology company headquartered in Cupertino, California"
   - "Apple designs and manufactures consumer electronics, software, and online services"

   Required output format:
   "Apple is a technology company headquartered in Cupertino, California that designs and manufactures consumer electronics, software, and online services."

   DO NOT add timestamp prefixes like "[Time: ...]" when they are not present in the original descriptions.

   **Example with timestamps:**
   Input descriptions:
   - "[Time: Fiscal Year 2024-Q1] Apple technology company reported iPhone revenue of $65.8B, up 15% YoY with strong market performance"
   - "[Time: Fiscal Year 2024-Q3] Apple technology company faced challenges with iPhone revenue of $39.3B, down 1.5% YoY due to market saturation"

   Required output format:
   "Apple technology company shows mixed quarterly performance. [Time: Fiscal Year 2024-Q1] Apple reported iPhone revenue of $65.8B, up 15% YoY with strong market performance. [Time: Fiscal Year 2024-Q3] Apple faced challenges with iPhone revenue of $39.3B, down 1.5% YoY due to market saturation."

6. **Conflict Handling:**
   - In cases of conflicting or inconsistent descriptions, first determine if these conflicts arise from multiple, distinct entities or relationships that share the same name.
   - If distinct entities/relations are identified, summarize each one *separately* within the overall output.
   - If conflicts within a single entity/relation exist (e.g., historical discrepancies from different time periods), attempt to reconcile them or present both viewpoints with noted uncertainty. When temporal information conflicts (e.g., same quarter with different data), note the discrepancy explicitly.
   - For time-based conflicts, prioritize more recent information while preserving historical context.

7. **Relevance Filtering:** Focus on substantive information. Remove redundant phrases while preserving unique details from each description.

8. **Length Constraint:** The summary's total length must not exceed {summary_length} tokens, while still maintaining depth and completeness.

9. **Language:**
   - The entire output must be written in {language}.
   - Proper nouns (e.g., personal names, place names, organization names) should be retained in their original language if a proper, widely accepted translation is not available or would cause ambiguity.

---Input---
{description_type} Name: {description_name}

Description List:

```
{description_list}
```

---Output---
"""

PROMPTS["entity_continue_extraction"] = """---Task---
Based on the last extraction task, identify and extract any **missed or incorrectly formatted** entities and relationships from the input text.

---Instructions---
1.  **Strict Adherence to System Format:** Strictly adhere to all format requirements for entity and relationship lists, including output order, field delimiters, and proper noun handling, as specified in the system instructions.
2.  **Focus on Corrections/Additions:**
    *   **Do NOT** re-output entities and relationships that were **correctly and fully** extracted in the last task.
    *   If an entity or relationship was **missed** in the last task, extract and output it now according to the system format.
    *   If an entity or relationship was **truncated, had missing fields, or was otherwise incorrectly formatted** in the last task, re-output the *corrected and complete* version in the specified format.
3.  **Output Format - Entities:** Output a total of 4 fields for each entity, delimited by `{tuple_delimiter}`, on a single line. The first field *must* be the literal string `entity`.
4.  **Output Format - Relationships:** Output a total of 5 fields for each relationship, delimited by `{tuple_delimiter}`, on a single line. The first field *must* be the literal string `relationship`.
5.  **Output Content Only:** Output *only* the extracted list of entities and relationships. Do not include any introductory or concluding remarks, explanations, or additional text before or after the list.
6.  **Completion Signal:** Output `{completion_delimiter}` as the final line after all relevant missing or corrected entities and relationships have been extracted and presented.
7.  **Output Language:** Ensure the output language is {language}. Proper nouns (e.g., personal names, place names, organization names) must be kept in their original language and not translated.

<Output>
"""


# TODO: Deprecated
PROMPTS["entity_if_loop_extraction"] = """
---Goal---'

Check if it appears some entities may have still been missed. Output "Yes" if so, otherwise "No".

---Output---
Output:"""

PROMPTS["fail_response"] = (
    "Sorry, I'm not able to provide an answer to that question.[no-context]"
)

PROMPTS["rag_response"] = """---Role---
You are a precise information retrieval assistant. Answer questions using ONLY the facts from Source Data (Document Chunks, Entities, and Relations).

---Instructions---
1. **Data Source Priority:**
    - Prioritize using **Document Chunks** as the main evidence.
    - Use **Entities and Relations** only as **auxiliary references** to support or fill missing details when the chunk content is incomplete.
    - Whenever possible, use the **original phrasing or wording** from the Document Chunk, but minor rephrasing is acceptable for clarity.

2. **When the query asks for a specific fact:**
    - Provide a **direct, concise** answer (1–2 sentences)
    - Prefer to extract the **exact phrase** from Document Chunks
    - Do **not** add explanations, background, or assumptions

3. **When the query requires connecting information:**
    - Answer in **2–4 sentences maximum**
    - Combine only the **minimal necessary facts** from Source Data
    - Avoid detailed analysis, speculation, or unrelated context

4.  **When the query asks for an overview or explanation:**
    - Synthesize fragmented information into a **coherent, structured answer**
    - Prioritize **completeness over brevity** - include all key facts from Source Data
    - Organize information logically with smooth transitions between related facts

5. ** When the query requests creative presentation:**
    - **Every statement** must be directly inferable from Source Data - do NOT invent details
    - Use creative language for presentation, but maintain strict factual accuracy

6. **Strict Grounding:**
    - Use ONLY information explicitly stated in Source Data
    - Do not introduce external knowledge or unverifiable claims

7. **Formatting & Language:**
    - Output must be **concise and factual**
    - Avoid lists or multi-paragraph explanations unless explicitly asked
    - Only provide detailed or enumerated answers when the question requests it (e.g., “list all reasons” or “explain in detail”)

---Source Data---
{context_data}

"""

PROMPTS["keywords_extraction"] = """---Role---
You are an expert keyword extractor, specializing in analyzing user queries for a Retrieval-Augmented Generation (RAG) system. Your purpose is to identify both high-level and low-level keywords in the user's query that will be used for effective document retrieval.

---Goal---
Given a user query, your task is to extract two distinct types of keywords:
1. **high_level_keywords**: for overarching concepts or themes, capturing user's core intent, the subject area, or the type of question being asked.
2. **low_level_keywords**: for specific entities or details, identifying the specific entities, proper nouns, technical jargon, product names, or concrete items.

---Instructions & Constraints---
1. **Output Format**: Your output MUST be a valid JSON object and nothing else. Do not include any explanatory text, markdown code fences (like ```json), or any other text before or after the JSON. It will be parsed directly by a JSON parser.
2. **Source of Truth**: All keywords must be explicitly derived from the user query, with both high-level and low-level keyword categories required to contain content.
3. **Concise & Meaningful**: Keywords should be concise words or meaningful phrases. Prioritize multi-word phrases when they represent a single concept. For example, from "latest financial report of Apple Inc.", you should extract "latest financial report" and "Apple Inc." rather than "latest", "financial", "report", and "Apple".
4. **Handle Edge Cases**: For queries that are too simple, vague, or nonsensical (e.g., "hello", "ok", "asdfghjkl"), you must return a JSON object with empty lists for both keyword types.

---Examples---
{examples}

---Real Data---
User Query: {query}

---Output---
Output:"""

PROMPTS["keywords_extraction_examples"] = [
    """Example 1:

Query: "How does international trade influence global economic stability?"

Output:
{
  "high_level_keywords": ["International trade", "Global economic stability", "Economic impact"],
  "low_level_keywords": ["Trade agreements", "Tariffs", "Currency exchange", "Imports", "Exports"]
}

""",
    """Example 2:

Query: "What are the environmental consequences of deforestation on biodiversity?"

Output:
{
  "high_level_keywords": ["Environmental consequences", "Deforestation", "Biodiversity loss"],
  "low_level_keywords": ["Species extinction", "Habitat destruction", "Carbon emissions", "Rainforest", "Ecosystem"]
}

""",
    """Example 3:

Query: "What is the role of education in reducing poverty?"

Output:
{
  "high_level_keywords": ["Education", "Poverty reduction", "Socioeconomic development"],
  "low_level_keywords": ["School access", "Literacy rates", "Job training", "Income inequality"]
}

""",
]

PROMPTS["naive_rag_response"] = """---Role---

You are a helpful assistant responding to user query about Document Chunks provided provided in JSON format below.

---Goal---

Generate a concise response based on Document Chunks and follow Response Rules, considering both the conversation history and the current query. Summarize all information in the provided Document Chunks, and incorporating general knowledge relevant to the Document Chunks. Do not include information not provided by Document Chunks.

---Conversation History---
{history}

---Document Chunks(DC)---
{content_data}

---RESPONSE GUIDELINES---
**1. Content & Adherence:**
- Strictly adhere to the provided context from the Knowledge Base. Do not invent, assume, or include any information not present in the source data.
- If the answer cannot be found in the provided context, state that you do not have enough information to answer.
- Ensure the response maintains continuity with the conversation history.

**2. Formatting & Language:**
- Format the response using markdown with appropriate section headings.
- The response language must match the user's question language.
- Target format and length: {response_type}

**3. Citations / References:**
- At the end of the response, under a "References" section, cite a maximum of 5 most relevant sources used.
- Use the following formats for citations: `[DC] <file_path_or_document_name>`

---USER CONTEXT---
- Additional user prompt: {user_prompt}

---Response---
Output:"""

# Entity Merging Prompts
PROMPTS["entity_merge_system"] = """You are a deduplication evaluator. Given two entities (A, B) with (entity_id, description),
you must carefully compare both entity_id and description.

CRITICAL: When calling the merge tool, you MUST use the EXACT entity_id values provided, including ALL special characters (asterisks *, parentheses (), brackets [], quotes, etc.). Do NOT modify, simplify, or remove any characters from the entity_id.

Only when you are OVER 95% confident that A and B refer to the SAME real-world entity,
you should INVOKE the tool: merge(a_entity_id, b_entity_id) using the EXACT entity_id strings.

If you are NOT over 95% confident (or they should not be merged), do NOT invoke any tool
and simply reply with the single token: NO_MERGE."""

PROMPTS["entity_merge_examples"] = """Example
#### Merge (exact entity_id match required) ###
A.entity_id = Apple
A.description = A technology company known for designing and manufacturing consumer electronics, software, and services.

B.entity_id = Apple Inc.
B.description = An American multinational technology company headquartered in Cupertino, California.

Tool call: merge_entities_tool(a_entity_id="Apple", b_entity_id="Apple Inc.")

#### Merge with special characters (preserve ALL characters) ###
A.entity_id = *Dr. John Smith*
A.description = A renowned scientist in quantum physics.

B.entity_id = Dr. John Smith
B.description = Physicist specializing in quantum mechanics.

Tool call: merge_entities_tool(a_entity_id="*Dr. John Smith*", b_entity_id="Dr. John Smith")

#### No merge - Obviously different entities ###
A.entity_id = Apple
A.description = A technology company known for designing and manufacturing consumer electronics, software, and services.

B.entity_id = TSMC
B.description = Taiwan Semiconductor Manufacturing Company, the world's largest semiconductor foundry.

Response: NO_MERGE

#### No merge - Similar descriptions but different proper nouns ###
A.entity_id = Microsoft Azure
A.description = Cloud computing platform offering infrastructure, software services, and AI capabilities to enterprise customers.

B.entity_id = Amazon Web Services
B.description = Cloud computing platform providing infrastructure, software services, and machine learning tools for businesses.

Response: NO_MERGE
Reasoning: Despite highly similar descriptions (both are cloud platforms), the entity_id clearly indicates different companies."""

PROMPTS["entity_merge_user"] = """Determine whether the two entities are the same. Only when you are over 95% confident that A and B refer to the SAME real-world entity, invoke the tool `merge_entities_tool(a_entity_id, b_entity_id)` using the EXACT entity_id values below (preserve all special characters). If you are not 95% confident, reply exactly with `NO_MERGE`.

CRITICAL DECISION CRITERIA:
1. **Proper Noun Analysis**: Different proper nouns (company names, person names, product names) usually indicate different entities, even if descriptions are similar
2. **Similar Business Descriptions**: Similar industry descriptions do NOT justify merging when entity_id clearly differs (e.g., different cloud service providers)
3. **When in Doubt**: If entity_id suggests different entities despite description overlap, default to NO_MERGE

IMPORTANT: Use these EXACT entity_id strings in the tool call (do not modify them):
A.entity_id = {a_entity_id}
A.description = {a_description}

B.entity_id = {b_entity_id}
B.description = {b_description}
"""

PROMPTS["entity_type_suggestion_system"] = """
You are an expert in Named Entity Recognition (NER) with expertise across multiple domains. An **entity type** is a category label used to classify named entities (people, organizations, locations, etc.) during information extraction. Your goal is to analyze the connections and relations between existing entity types and document content to provide meaningful refinements or additions that enhance entity extraction for various document types.

## Task Requirements:
- Suggest entity types that improve extraction quality for the specific document domain
- Consider the document's context, structure, and content patterns
- Avoid suggesting "other" or "unknown" types
- Do not suggest duplicates or overlapping entity types
- Prioritize quality over quantity with domain-appropriate coverage
- Consider structured data elements like tables, lists, and classifications when present
- Provide concise yet clear explanations with relevant examples
- Respond in strict JSON array format only

## Multi-Domain Context Considerations:
- **Technical Documents**: Components, specifications, procedures, standards, measurements
- **Academic Papers**: Research methods, findings, citations, institutions, datasets
- **Business Documents**: Metrics, processes, departments, strategies, performance indicators
- **Financial Reports**: Assets, revenues, ratios, statements, accounting items
- **News Articles**: Events, locations, quotes, sources, impacts
- **Legal Documents**: Clauses, parties, terms, obligations, references

## Response Format:
[
{
    "entity_type": "<entity_type_name>",
    "explanation": "<detailed_explanation>"
}
]

## Examples:

### Example 1: Academic Research Document
#### Current Entity Types:
[
    {
        "entity_type": "organization",
        "explanation": "An entity representing organizations, companies, or institutions."
    },
    {
        "entity_type": "person",
        "explanation": "An entity representing individual persons."
    }
]

#### Document Content:
The research team at MIT conducted a systematic review of machine learning algorithms, analyzing 150 datasets from 2020-2024. The study employed cross-validation techniques and achieved an accuracy of 94.2% using deep neural networks. Results were published in Nature Machine Intelligence.

#### Suggested New Entity Types:
[
    {
        "entity_type": "research_method",
        "explanation": "Methodological approaches and techniques used in research (e.g., 'systematic review', 'cross-validation', 'deep neural networks')."
    },
    {
        "entity_type": "metric",
        "explanation": "Quantitative measurements and performance indicators (e.g., '94.2%', 'accuracy', 'sample size of 150')."
    },
    {
        "entity_type": "publication",
        "explanation": "Academic publications, journals, and research outputs (e.g., 'Nature Machine Intelligence', 'conference proceedings')."
    }
]

### Example 2: Financial Report Document
#### Current Entity Types:
[
    {
        "entity_type": "organization",
        "explanation": "An entity representing organizations, companies, or institutions."
    },
    {
        "entity_type": "person",
        "explanation": "An entity representing individual persons."
    },
    {
        "entity_type": "geo",
        "explanation": "An entity representing geographical locations."
    }
]

#### Document Content:
Apple Inc. released its Q3 2024 earnings report on July 31, 2024. The company reported revenue of $85.8 billion, up 5% year-over-year. iPhone sales contributed $39.3 billion, while Services revenue grew to $24.2 billion. CEO Tim Cook highlighted strong performance in emerging markets, particularly India and Southeast Asia. The company's operating margin improved to 27.5%, and net income reached $21.4 billion. Apple also announced a $110 billion share buyback program. The Greater China region faced challenges with revenue declining 6.5% to $14.7 billion due to increased competition.

#### Suggested New Entity Types:
[
    {
        "entity_type": "financial_metric",
        "explanation": "Financial measurements and performance indicators including revenue, profit, margins, growth rates, and monetary values (e.g., '$85.8 billion revenue', '5% year-over-year', '27.5% operating margin')."
    },
    {
        "entity_type": "product_line",
        "explanation": "Product categories, business segments, or service offerings that contribute to company revenue (e.g., 'iPhone sales', 'Services revenue')."
    },
    {
        "entity_type": "temporal_range",
        "explanation": "Time periods including fiscal quarters, years, and reporting periods (e.g., 'Q3 2024', 'July 31, 2024', 'year-over-year')."
    },
    {
        "entity_type": "corporate_action",
        "explanation": "Strategic business actions and initiatives taken by organizations (e.g., '$110 billion share buyback program', 'market expansion', 'acquisitions')."
    },
    {
        "entity_type": "market_region",
        "explanation": "Geographical markets or regions with business performance context (e.g., 'Greater China region', 'emerging markets', 'India and Southeast Asia')."
    }
]
"""

PROMPTS["entity_type_suggestion_user"] = """
## Current Entity Types:
{current_entity_types}

## Task:
Based on the following document content, analyze and suggest new entity types with explanations if needed. Consider the document's domain and structure:

1. **Content Analysis**: Examine the document's subject matter, terminology, and domain-specific concepts
2. **Structural Elements**: Notice tables, lists, classifications, and organized data patterns
3. **Entity Patterns**: Identify recurring entity types that appear frequently in the content
4. **Relationships**: Consider entities that represent important connections and dependencies
5. **Domain Specificity**: Suggest entity types that capture domain-specific knowledge effectively

## Document Content:
{file_content}

Please carefully analyze the entities that appear in the document, considering its structure and domain context, and suggest appropriate new entity types that would improve extraction coverage for this type of content.
"""

PROMPTS["entity_type_refinement_system"] = """
You are an advanced linguistic assistant with expertise in Named Entity Recognition (NER) across multiple domains.

## Task:
Refine a list of entity types by removing clear duplicates or highly overlapping types, ensuring a balanced and well-optimized schema that preserves domain-specific granularity.

## Critical Requirements:
- **BALANCED CONSOLIDATION**: Merge only clearly redundant or highly overlapping entity types
- **QUALITY AND COVERAGE**: Aim for a concise schema that maintains domain-specific distinctions
- **HIERARCHICAL AWARENESS**: Merge child types into parent categories only when the distinction lacks value
- **REDUCE REDUNDANCY**: Remove types that are true duplicates or near-synonyms
- **DOMAIN PRESERVATION**: Maintain entity types that capture domain-specific knowledge and semantic distinctions

## Consolidation Guidelines:
1. **Merge Clear Duplicates**: Combine types with semantic overlap (e.g., "Company" + "Corporation" + "Organization" -> "Organization")
2. **Eliminate True Sub-types**: Remove specific sub-types only if they don't add domain value (e.g., "CEO" -> "Person" if no role-specific analysis needed)
3. **Remove Generic Types**: Eliminate vague types like "Concept", "Thing", "Item", "Other", "Unknown"
4. **Preserve Domain Distinctions**: Keep related types if they serve different analytical purposes (e.g., "Product" vs "Service" for business analysis)
5. **Maintain Semantic Diversity**: Keep types that represent fundamentally different entity categories or domain-specific concepts

## Domain-Specific Considerations:
- **Financial Documents**: Organization, Person, Financial_Metric, Temporal_Range 
- **Academic Papers**: Organization, Person, Research_Method, Publication
- **Technical Documents**: Organization, Person, Component, Specification 

## Response Format:
[
{
    "entity_type": "<entity_type_name>",
    "explanation": "<clear_explanation>"
}
]

## Example 1 - Aggressive Consolidation:
### Entity Types List to Refine:
[
    {"entity_type": "Company", "explanation": "A company is a legal entity..."},
    {"entity_type": "Organization", "explanation": "An organization is a group..."},
    {"entity_type": "Institution", "explanation": "An institution is a formal organization..."},
    {"entity_type": "Corporation", "explanation": "A corporation is a type of company..."},
    {"entity_type": "CEO", "explanation": "A chief executive officer..."},
    {"entity_type": "Employee", "explanation": "A person working for an organization..."},
    {"entity_type": "Person", "explanation": "An individual person..."},
    {"entity_type": "Human", "explanation": "A human being..."}
]

### Refined List:
[
    {
        "entity_type": "Organization",
        "explanation": "An entity representing organizations, companies, institutions, or corporations (consolidated from Company, Organization, Institution, Corporation)."
    },
    {
        "entity_type": "Person",
        "explanation": "An entity representing individual persons including executives, employees, and other human entities (consolidated from Person, Human, CEO, Employee)."
    }
]

## Example 2 - Domain-Specific Consolidation:
### Entity Types List to Refine:
[
    {"entity_type": "Revenue", "explanation": "Company revenue..."},
    {"entity_type": "Profit", "explanation": "Company profit..."},
    {"entity_type": "Loss", "explanation": "Financial loss..."},
    {"entity_type": "Growth_Rate", "explanation": "Growth percentage..."},
    {"entity_type": "Margin", "explanation": "Profit margin..."},
    {"entity_type": "Metric", "explanation": "Performance metric..."},
    {"entity_type": "Quarter", "explanation": "Fiscal quarter..."},
    {"entity_type": "Year", "explanation": "Fiscal year..."},
    {"entity_type": "Time_Period", "explanation": "Time period..."},
    {"entity_type": "Fiscal_Period", "explanation": "Fiscal period..."}
]

### Refined List:
[
    {
        "entity_type": "Organization",
        "explanation": "Organizations, companies, or institutions."
    },
    {
        "entity_type": "Person",
        "explanation": "Individual persons."
    },
    {
        "entity_type": "Financial_Metric",
        "explanation": "Financial measurements including revenue, profit, loss, growth rate, margin, and other performance metrics (consolidated from Revenue, Profit, Loss, Growth_Rate, Margin, Metric)."
    },
    {
        "entity_type": "Temporal_Range",
        "explanation": "Time periods including quarters (Q1-Q4), fiscal years, and other temporal ranges (consolidated from Quarter, Year, Time_Period, Q1, Q2, Fiscal_Period)."
    }
]
"""

PROMPTS["entity_type_refinement_user"] = """
## Entity Types List to Refine:
{entity_types}

## Task:
**Carefully consolidate this list** to remove clear duplicates while preserving domain-specific entity types.

**Your Goals:**
1. **Remove Clear Duplicates**: Merge only types that are true synonyms or near-duplicates
2. **Eliminate Redundancy**: Remove types that are completely covered by another type without adding value
3. **Preserve Domain Coverage**: Maintain types that capture distinct domain concepts or analytical purposes

**Action Steps:**
- Identify true duplicates and near-synonyms (semantic overlap)
- For duplicate groups, select the most appropriate type name
- Merge explanations to reflect consolidated coverage
- Keep sub-types that provide valuable domain-specific distinctions
- Eliminate only generic or vague type names (e.g., "Thing", "Other")

**Quality Check:**
- Are these types true duplicates or near-synonyms? If yes, merge them.
- Do these types serve different analytical or domain purposes? If yes, keep them separate.
- Does this type add meaningful domain-specific information? If yes, preserve it.

Please provide the refined list in strict JSON array format.
"""


PROMPTS["orphaned_entity_description"] = """---Role---
You are a Knowledge Graph Specialist responsible for generating entity descriptions for entities discovered through relationship extraction.

---Task---
An entity "{entity_name}" was referenced in a relationship but was not explicitly extracted as an entity. Your task is to analyze the context and generate an appropriate entity type and description for this entity.

---Context---
**Relationship Description**: {relationship_desc}

**Source Document Chunk**:
```
{source_chunk}
```

**Available Entity Types**: {entity_types}

---Instructions---
1. **Entity Type Selection**: Choose the most appropriate entity type from the available types. If none fit well, classify it as "Other".
2. **Description Quality**: Generate a comprehensive, meaningful description (50-150 words) that:
   - Explains what this entity is based on the source context
   - Uses third-person perspective
   - Avoids pronouns like "this", "it", "they"
   - Is written in {language}

3. **Output Format**: Output EXACTLY in this format (no additional text):
```
(entity_type)<SEP>(description)
```

---Examples---

**Example 1**:
Entity Name: TSMC
Relationship Description: TSMC supplies advanced semiconductor chips to Apple
Source Chunk: Apple announced that despite supply chain challenges with key supplier TSMC, the company maintained strong performance. TSMC manufactures the cutting-edge processors used in iPhone devices.
Available Types: [organization, person, geo, technology, product]

Output:
```
organization<SEP>TSMC is a semiconductor manufacturing company that manufactures cutting-edge processors used in iPhone devices. TSMC specializes in advanced chip production and serves as a key technology supplier.
```

**Example 2**:
Entity Name: Greater China
Relationship Description: Apple revenue from Greater China reached $14.7B
Source Chunk: The company reported regional performance with Greater China generating $14.7 billion in revenue. This represents Apple's combined sales in mainland China, Hong Kong, and Taiwan markets.
Available Types: [organization, person, geo, financial_metric, temporal_range]

Output:
```
geo<SEP>Greater China is a geographical market region that encompasses mainland China, Hong Kong, and Taiwan markets. Greater China represents a major regional sales territory in Asia.
```

---Real Data---
Entity Name: {entity_name}
Relationship Description: {relationship_desc}
Source Chunk: {source_chunk}
Available Entity Types: {entity_types}

---Output---
"""

PROMPTS["recognition_filter"] = """---Role---
You are an intelligent filter in a question-answering system. Your task is to identify and remove irrelevant entities and relationships based on the question type and complexity.

---Task---
1. **Analyze the question** to determine its type and requirements
2. **Identify which entities and relationships to REMOVE** (not keep)
3. **Adapt your filtering strategy** based on question characteristics

---Question Type Guidelines---

**Type 1: Fact Retrieval** (Simple factual questions with direct answers)
- Characteristics: Asks for specific facts like location, time, name, or definition
- Question patterns: "Where is...", "When was...", "Who invented...", "What is the capital of..."
- Filtering Strategy: STRICT - Only keep items directly involved in the answer
- What to remove: Background information, related but indirect entities, historical context not directly answering the question

**Type 2: Complex Reasoning** (Multi-hop questions requiring logical connections)
- Characteristics: Requires connecting multiple pieces of information across different contexts
- Question patterns: "How did X influence Y?", "What is the relationship between...", "Why did X lead to Y?"
- Filtering Strategy: CONSERVATIVE - Preserve all potential reasoning paths and intermediate connections
- What to remove: Only items from completely different domains with zero connection

**Type 3: Contextual Summarize** (Synthesis of fragmented information into coherent understanding)
- Characteristics: Asks for comprehensive description, role explanation, or overview
- Question patterns: "Describe the role of...", "What were the main features of...", "Explain the significance of..."
- Filtering Strategy: BALANCED - Keep descriptive context and relevant background while removing tangential information
- What to remove: Entities from unrelated domains, overly specific details not contributing to the overall picture

**Type 4: Creative Generation** (Inference, imagination, or hypothetical scenarios)
- Characteristics: Asks to create, imagine, rewrite, or generate content beyond factual retrieval
- Question patterns: "Write a story about...", "Imagine if...", "Rewrite as...", "Create a dialogue between..."
- Filtering Strategy: VERY CONSERVATIVE - Keep rich details, atmosphere, and contextual elements
- What to remove: Only completely unrelated topics from different domains

---Examples---

### Example 1: Fact Retrieval ###
Query: "What is the capital city of Australia?"

Entities:
[
  {{"entity_name": "Australia", "description": "A country in Oceania"}},
  {{"entity_name": "Canberra", "description": "The capital city of Australia"}},
  {{"entity_name": "Sydney", "description": "Largest city in Australia, major port"}},
  {{"entity_name": "Kangaroo", "description": "Native Australian animal"}},
  {{"entity_name": "Great Barrier Reef", "description": "Coral reef system off Australia's coast"}}
]

Relations:
[
  {{"id": "rel_0", "src_id": "Canberra", "tgt_id": "Australia", "description": "capital of"}},
  {{"id": "rel_1", "src_id": "Sydney", "tgt_id": "Australia", "description": "located in"}},
  {{"id": "rel_2", "src_id": "Kangaroo", "tgt_id": "Australia", "description": "native to"}},
  {{"id": "rel_3", "src_id": "Great Barrier Reef", "tgt_id": "Australia", "description": "located near"}}
]

Output:
{{
  "irrelevant_entity_ids": ["Sydney", "Kangaroo", "Great Barrier Reef"],
  "irrelevant_relation_ids": ["rel_1", "rel_2", "rel_3"]
}}

### Example 2: Complex Reasoning ###
Query: "How did the development of steam engines impact industrial textile production?"

Entities:
[
  {{"entity_name": "Steam Engine", "description": "Machine that converts steam power to mechanical work"}},
  {{"entity_name": "Industrial Revolution", "description": "Period of major industrialization in the 18th-19th centuries"}},
  {{"entity_name": "Textile Mills", "description": "Factories for textile production"}},
  {{"entity_name": "Cotton", "description": "Soft fiber used in textile manufacturing"}},
  {{"entity_name": "James Watt", "description": "Scottish inventor who improved the steam engine"}},
  {{"entity_name": "Electricity", "description": "Form of energy discovered later"}},
  {{"entity_name": "Computer", "description": "Modern electronic device"}}
]

Relations:
[
  {{"id": "rel_0", "src_id": "Steam Engine", "tgt_id": "Industrial Revolution", "description": "enabled"}},
  {{"id": "rel_1", "src_id": "Steam Engine", "tgt_id": "Textile Mills", "description": "powered"}},
  {{"id": "rel_2", "src_id": "Textile Mills", "tgt_id": "Cotton", "description": "processed"}},
  {{"id": "rel_3", "src_id": "James Watt", "tgt_id": "Steam Engine", "description": "improved"}},
  {{"id": "rel_4", "src_id": "Electricity", "tgt_id": "Computer", "description": "powers"}}
]

Output:
{{
  "irrelevant_entity_ids": ["Computer"],
  "irrelevant_relation_ids": ["rel_4"]
}}

### Example 3: Contextual Summarize ###
Query: "Describe the main characteristics of tropical rainforest ecosystems"

Entities:
[
  {{"entity_name": "Tropical Rainforest", "description": "Dense forest in tropical regions with high rainfall"}},
  {{"entity_name": "Biodiversity", "description": "Variety of plant and animal life"}},
  {{"entity_name": "Canopy Layer", "description": "Upper layer of trees in rainforest"}},
  {{"entity_name": "Amazon", "description": "Largest tropical rainforest"}},
  {{"entity_name": "Precipitation", "description": "High annual rainfall in these regions"}},
  {{"entity_name": "Arctic Tundra", "description": "Cold, treeless biome in polar regions"}},
  {{"entity_name": "Desert Sand", "description": "Dry, sandy terrain"}}
]

Relations:
[
  {{"id": "rel_0", "src_id": "Tropical Rainforest", "tgt_id": "Biodiversity", "description": "supports high levels of"}},
  {{"id": "rel_1", "src_id": "Canopy Layer", "tgt_id": "Tropical Rainforest", "description": "part of"}},
  {{"id": "rel_2", "src_id": "Amazon", "tgt_id": "Tropical Rainforest", "description": "example of"}},
  {{"id": "rel_3", "src_id": "Precipitation", "tgt_id": "Tropical Rainforest", "description": "characteristic of"}},
  {{"id": "rel_4", "src_id": "Arctic Tundra", "tgt_id": "Desert Sand", "description": "unrelated to"}}
]

Output:
{{
  "irrelevant_entity_ids": ["Arctic Tundra", "Desert Sand"],
  "irrelevant_relation_ids": ["rel_4"]
}}

### Example 4: Creative Generation ###
Query: "Write a poem about a lighthouse keeper watching a storm"

Entities:
[
  {{"entity_name": "Lighthouse", "description": "Tower with light to guide ships"}},
  {{"entity_name": "Keeper", "description": "Person who maintains the lighthouse"}},
  {{"entity_name": "Storm", "description": "Severe weather with wind and rain"}},
  {{"entity_name": "Ocean", "description": "Large body of saltwater"}},
  {{"entity_name": "Waves", "description": "Moving water surface"}},
  {{"entity_name": "Solitude", "description": "State of being alone"}},
  {{"entity_name": "Thunder", "description": "Sound from lightning"}},
  {{"entity_name": "Spacecraft", "description": "Vehicle for space travel"}},
  {{"entity_name": "Quantum Physics", "description": "Study of subatomic particles"}}
]

Relations:
[
  {{"id": "rel_0", "src_id": "Keeper", "tgt_id": "Lighthouse", "description": "maintains"}},
  {{"id": "rel_1", "src_id": "Storm", "tgt_id": "Ocean", "description": "occurs over"}},
  {{"id": "rel_2", "src_id": "Waves", "tgt_id": "Storm", "description": "created by"}},
  {{"id": "rel_3", "src_id": "Thunder", "tgt_id": "Storm", "description": "part of"}},
  {{"id": "rel_4", "src_id": "Keeper", "tgt_id": "Solitude", "description": "experiences"}},
  {{"id": "rel_5", "src_id": "Spacecraft", "tgt_id": "Quantum Physics", "description": "unrelated"}}
]

Output:
{{
  "irrelevant_entity_ids": ["Spacecraft", "Quantum Physics"],
  "irrelevant_relation_ids": ["rel_5"]
}}

---Input---
Query: {query}

Entities:
{entities_json}

Relations:
{relations_json}

---Output Format---
Analyze the question type and list items to remove in JSON format:

{{
  "irrelevant_entity_ids": ["entity1", "entity2", ...],
  "irrelevant_relation_ids": ["rel_1", "rel_2", ...]
}}

---Critical Guidelines---
1. **No fixed removal percentage** - Remove based solely on relevance, not quantity. If everything is relevant, return empty lists. If most items are irrelevant, remove most.

2. **Question type determines strictness**:
   - Fact Retrieval: Be strict, keep only direct answer components
   - Complex Reasoning: Be conservative, preserve reasoning chains
   - Contextual Summarize: Be balanced, keep descriptive context
   - Creative Generation: Be very conservative, keep inspirational material

3. **When in doubt, keep the item** - It's better to include a potentially useful item than to remove something that might be part of the answer path

4. **Consider indirect connections** - For Complex Reasoning and Creative Generation, an entity may seem unrelated but could be a bridge between concepts

5. **Examine both names and descriptions** - The entity name might seem irrelevant, but the description could reveal connections to the query

Output your analysis:"""


