# ... (Sections 1 through 9 remain identical to the previous block) ...

# =====================================================
# 10. USER COMMAND RUNNER WITH "NO 2-STROKE" HARD GATE
# =====================================================
# ... (Inside the Execution Loop / Scenario D) ...

                    # Adjust prompt rules to explicitly reinforce persistent topic memory
                    topic_context_injection = f"""
                    CRITICAL WORKSPACE LIMITATION:
                    You are explicitly assigned to find information ONLY for the following engine profile baseline: ROTAX {st.session_state.active_engine}.
                    
                    STRICT 2-STROKE BAN:
                    You are STRICTLY FORBIDDEN from outputting any information related to 2-stroke engines. 
                    If any manual extract or text chunk mentions: "503", "582", "618", "pre-mix", "oil injection pump cable", "two-stroke", "2-stroke", or "points ignition", you must IMMEDIATELY DISCARD that entire chunk of text. 
                    Do not suggest any parts, tool sizes, or procedures related to those engines. 
                    If the only text returned is 2-stroke data, you must output: "No 4-stroke maintenance data found for this query."
                    """

                    final_prompt = f"""You are supporting a licensed aircraft maintenance technician.
You must answer the user's question relying EXCLUSIVELY on the provided manual extracts.

{topic_context_injection}

CRITICAL DISCIPLINE DIRECTIVE FOR HYDRAULIC PRESSURE TESTING:
If the user query is asking about testing "OIL PRESSURE" or "FUEL PRESSURE", you are STRICTLY FORBIDDEN from outputting any procedure that mentions "spark plugs", "pistons", "TDC", "cylinder heads", or "differential pressure drop tests". 

CRITICAL DISCIPLINE DIRECTIVE FOR TOOLING:
For 9-Series engines (912/914/915/916), ALWAYS verify: Spark plug socket MUST be 16mm (5/8"). If extract says 18mm, it is a 2-stroke legacy error—DISCARD IT and use 16mm.

Structure your response exactly like this:

### 1. QUICK SPEC / PROCEDURE
* Provide the concrete, sequential maintenance steps, checks, settings, or technical values.
* If the task text is missing from the extracts or contaminated by 2-stroke data, explicitly state that manual data for the 4-stroke engine is not present.

### 2. PARTS & MANUAL DATA
* List specific part numbers and tool codes.
* If the data is missing or derived from an excluded 2-stroke chapter, state: \"Manual data gaps present\".

---
MANUAL EXTRACTS:
{context_str}
---
USER QUESTION: {user_query}"""

                    assistant_response = call_llm(final_prompt)
                    
                    # Footers...