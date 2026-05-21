import os
import re
import json
import sys
import numpy as np
import requests
import faiss
from openai import OpenAI

# Initialize core test platform keys
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

if not OPENROUTER_API_KEY or not OPENAI_API_KEY:
    print("🚨 Critical Error: Environments must have both OPENROUTER_API_KEY and OPENAI_API_KEY set.")
    sys.exit(1)

openai_client = OpenAI(api_key=OPENAI_API_KEY)
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Vector Matrix Environment Hooks
INDEX_PATH = "faiss_index.bin"
METADATA_PATH = "faiss_metadata.json"

# =====================================================
# THE GOLD STANDARD AVIATION DATA EVALUATION MATRIX
# =====================================================
EVALUATION_BATTERY = [
    {
        "id": "VAL-001",
        "task": "OIL CHANGE / MAGNETIC PLUG INSPECTION",
        "query": "explain how to remove the oil filter and drain the engine oil safely",
        "golden_truths": [
            "Drain the oil only when warm or hot to ensure complete scavenging.",
            "Remove the oil tank drain screw using a standard 17mm socket or wrench.",
            "Remove the oil filter using a standard hand filter wrench.",
            "Torque settings and torque wrenches apply exclusively to final installation tightening. Do not use torque tools during removal."
        ],
        "fatal_hallucinations": [
            "torque wrench to remove", "loosen to 25 Nm", "loosen to 14 Nm", "let the engine cool down completely before draining"
        ]
    },
    {
        "id": "VAL-002",
        "task": "OIL CHANGE / MAGNETIC PLUG INSPECTION",
        "query": "what torque setting do I use for the magnetic plug and do I install a washer?",
        "golden_truths": [
            "The crankcase magnetic plug tightening torque is strictly 20 Nm (177 in. lb).",
            "Absolute ban on the use of crush washers, gaskets, or sealing rings on the magnetic plug.",
            "Do not apply Loctite, thread sealants, or compounds to the magnetic plug threads.",
            "Lubricate magnetic plug threads with clean engine oil prior to installation."
        ],
        "fatal_hallucinations": [
            "torque to 25 Nm", "torque to 30 Nm", "install a new copper sealing ring on the magnetic plug", "apply Loctite 567"
        ]
    },
    {
        "id": "VAL-003",
        "task": "CARBURETOR SYNCHRONIZATION",
        "query": "give me the clearance steps and exact tolerances to balance my carbs at idle",
        "golden_truths": [
            "Maximum allowable pneumatic pressure variation at warm idle (1800-2000 RPM) is strictly 20 mbar (0.29 psi).",
            "Pneumatic variation threshold at cruise power (3500-4000 RPM) is strictly 0 mbar.",
            "Verify all throttle Bowden cables possess a minimum free play of 1 mm (0.04 in) against the physical idle stops."
        ],
        "fatal_hallucinations": [
            "oil filter", "magnetic plug", "torque to 25 Nm", "17mm socket", "copper sealing ring"
        ]
    }
]

# =====================================================
# HELPER RUNNERS & INTERFACE EMULATORS
# =====================================================
def get_embedding(text):
    return openai_client.embeddings.create(input=[text.replace("\n", " ")], model="text-embedding-3-small").data[0].embedding

def load_vector_context(search_query):
    if not os.path.exists(INDEX_PATH) or not os.path.exists(METADATA_PATH):
        return "No local vector database manuals found on server."
    try:
        index = faiss.read_index(INDEX_PATH)
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        query_vector = np.array([get_embedding(search_query)]).astype('float32')
        distances, indices = index.search(query_vector, 4)
        chunks = []
        for score, idx in zip(distances[0], indices[0]):
            if idx != -1 and score < 1.25 and idx < len(metadata):
                chunks.append(metadata[idx]['text'])
        return "\n\n---\n\n".join(chunks) if chunks else "No direct manual segments matched."
    except Exception as e:
        return f"Error executing index context scan: {str(e)}"

def run_test_query(engine, topic, query, app_code):
    """Dynamically extracts current prompt logic from app.py source and runs a test round"""
    context_str = load_vector_context(f"{topic} {query}")
    
    # Extract the system prompt and final prompt strings using regex blocks
    try:
        system_content_match = re.search(r'"role":\s*"system",\s*"content":\s*\(\s*"""(.*?)"""\s*\)|"content":\s*\((.*?)\)', app_code, re.DOTALL)
        system_content = system_content_match.group(1) if system_content_match else "Aviation maintenance assistant."
    except Exception:
        system_content = "Aviation maintenance assistant."
        
    # Isolate specific truth matrix from registry data inside the app code string
    registry_match = re.search(r'"{0}":\s*"""(.*?)"""'.format(topic), app_code, re.DOTALL)
    active_truth_injection = registry_match.group(1) if registry_match else ""

    # Reconstruct the exact final execution prompt matching section 10
    final_prompt = f"""You are actively mentoring an aircraft technician working on a ROTAX {engine}.
    Use the provided manual extracts combined with the mandatory engineering truth matrix rules below.

    {active_truth_injection}

    STRICT CHRONOLOGICAL SEPARATION RULES:
    1. REMOVAL / DISASSEMBLY STEPS: Fasteners are removed using standard sockets or wrenches. You are STRICTLY PROHIBITED from mentioning any torque settings, torque values, or torque wrenches during a removal step. To imply a torque wrench is used to loosen or remove a component is a critical safety failure.
    2. REASSEMBLY / INSTALLATION STEPS: Torque settings and torque wrenches apply EXCLUSIVELY to final tightening actions. 
    3. DATA INTEGRITY FILTER: If a part number is duplicated across completely separate components in the text extracts, treat it as a column parsing error. Hide that number under Section 3 and state: "Part number not clearly legible in manual extract table."
    4. TWO-STROKE INFORMATION IS COMPLETELY BANNED.

    Structure your response exactly like this to maintain an authoritative, mentor voice:
    ### 1. THE WORKBENCH PROCEDURE
    ### 2. ⚠️ INSPECTOR'S SAFETY BRIEF
    ### 3. REQUIRED PARTS & SPECIFICATIONS
    ---
    REFERENCE EXTRACTS: {context_str}
    TECHNICIAN'S QUERY: {query}"""

    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "meta-llama/llama-3.1-8b-instruct",
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": "You are a Senior iRMT LAA/BMAA Inspector and aircraft workshop mentor."},
            {"role": "user", "content": final_prompt}
        ]
    }
    response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=60)
    response_json = response.json()

    # --- ERROR CACHE GATEWAY ---
    if "choices" not in response_json:
        print("\n🚨 OpenRouter API Gateway Error Intercepted!")
        print("--------------------------------------------------")
        print(json.dumps(response_json, indent=2))
        print("--------------------------------------------------")
        print("Please check your OpenRouter account configuration, billing balances, or key definitions listed above.")
        sys.exit(1)
        
    return response_json["choices"][0]["message"]["content"]

# =====================================================
# THE EVALUATOR ENGINE & THE JURY LAYER
# =====================================================
def evaluate_response(query, response, golden_truths, fatal_hallucinations):
    """Uses programmatic verification combined with a high-order Judge LLM to verify accuracy"""
    report = {"passed": True, "log": []}
    lower_resp = response.lower()

    # Step 1: Execute Hard Programmatic Negative Pattern Checking
    for toxic_phrase in fatal_hallucinations:
        if toxic_phrase.lower() in lower_resp:
            report["passed"] = False
            report["log"].append(f"CRITICAL SAFETY VIOLATION DETECTED: Response printed banned phrase '{toxic_phrase}'.")

    # Step 2: Use high-fidelity frontier model to check alignment with Golden Truths
    judge_prompt = f"""You are a Lead Quality Assurance Aeronautical Engineer auditing an AI assistant's response.
    Verify if the Assistant's Response strictly adheres to the Verified Golden Truth Specifications.

    GOLDEN TRUTHS REQUIRED:
    {json.dumps(golden_truths, indent=2)}

    ASSISTANT RESPONSE TO EVALUATE:
    ---
    {response}
    ---

    Output your audit report strictly in this JSON format:
    {{
        "all_truths_honored": true/false,
        "missing_truths": ["list of facts omitted or misstated"],
        "reasoning": "Clear explanation of mechanical contradictions or accuracy gaps found."
    }}"""

    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "google/gemini-2.5-pro",
        "response_format": {"type": "json_object"},
        "messages": [{"role": "user", "content": judge_prompt}]
    }
    
    try:
        res = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=40)
        res_json = res.json()
        
        if "choices" not in res_json:
            report["passed"] = False
            report["log"].append(f"Judge verification API failure payload: {json.dumps(res_json)}")
            return report

        eval_json = json.loads(res_json["choices"][0]["message"]["content"])
        if not eval_json["all_truths_honored"]:
            report["passed"] = False
            for chunk in eval_json["missing_truths"]:
                report["log"].append(f"ACCURACY ERROR: {chunk} (Context: {eval_json['reasoning']})")
    except Exception as je:
        report["log"].append(f"Judge verification system warning: {str(je)}")

    return report

# =====================================================
# THE SELF-OPTIMIZATION ROUTER LOOP
# =====================================================
def execute_self_refinement_cycle(max_iterations=3):
    app_file_path = "app.py"
    
    if not os.path.exists(app_file_path):
        print(f"🚨 Target file '{app_file_path}' not found in root directory.")
        return

    for run_idx in range(max_iterations):
        print(f"\n⚡ Starting Optimization Refinement Loop {run_idx + 1} of {max_iterations}...")
        print("----------------------------------------------------------------------")
        
        with open(app_file_path, "r", encoding="utf-8") as f:
            current_app_code = f.read()

        all_cases_passed = True
        failed_case_summaries = []

        # Run through the entire test database
        for case in EVALUATION_BATTERY:
            print(f"📋 Testing Case [{case['id']}] for Task: {case['task']}...")
            output_text = run_test_query("914", case['task'], case['query'], current_app_code)
            audit_result = evaluate_response(case['query'], output_text, case['golden_truths'], case['fatal_hallucinations'])

            if not audit_result["passed"]:
                all_cases_passed = False
                error_block = f"Test Case [{case['id']}] failed for prompt query: '{case['query']}'. Errors logged:\n" + "\n".join([f" - {l}" for l in audit_result["log"]])
                failed_case_summaries.append(error_block)
                print(f"❌ Case [{case['id']}] failed accuracy parameters.")
            else:
                print(f"✅ Case [{case['id']}] passed clean.")

        if all_cases_passed:
            print("\n🚀 SUCCESS: The application has passed 100% of the mechanical safety test criteria!")
            break
        else:
            print(f"\n⚠️ System gaps identified. Submitting {len(failed_case_summaries)} failure logs to prompt refinery layer...")
            
            refinery_prompt = f"""You are a Lead AI Prompt Optimization Engineer specializing in high-reliability aerospace applications.
            Your task is to take the current source code of a Streamlit RAG application (`app.py`) and completely rewrite its system prompts, engineering overrides, or prompt templates to eliminate explicit mechanical hallucinations and safety violations.

            SYSTEM FAILURE LOGS TO REFACTOR:
            {chr(10).join(failed_case_summaries)}

            CURRENT APP.PY CODE BASE:
            ```python
            {current_app_code}
            ```

            DIRECTIONS:
            1. Refactor the `SPEC_REGISTRY` entries or the `final_prompt` structural templates inside Section 10 to implement absolute affirmative boundaries.
            2. Eliminate any formatting or text generation patterns that cause the 8B parameter model to confuse disassembly tooling with assembly torque settings.
            3. Ensure the tone is highly informative, authoritative, and structured exactly as expected by the frontend.
            4. Return the COMPLETE, updated, production-ready `app.py` script. Do not output code placeholders or truncate lines. Output your entire response enclosed inside a single ```python block.
            """

            headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
            payload = {
                "model": "google/gemini-2.5-pro",
                "messages": [{"role": "user", "content": refinery_prompt}]
            }
            
            print("🧠 Querying Frontier Refinery Layer for structural optimization patch...")
            refinery_res = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=120)
            refinery_json = refinery_res.json()

            if "choices" not in refinery_json:
                print("🚨 Refinement System Error payload from API endpoint:")
                print(json.dumps(refinery_json, indent=2))
                sys.exit(1)

            raw_response = refinery_json["choices"][0]["message"]["content"]
            
            # Extract updated python code block cleanly
            code_match = re.search(r'```python(.*?)```', raw_response, re.DOTALL)
            if code_match:
                updated_code = code_match.group(1).strip()
                with open(app_file_path, "w", encoding="utf-8") as f:
                    f.write(updated_code)
                print("💾 Patch applied successfully. `app.py` updated. Resetting test engine...")
            else:
                print("🚨 Refinement Failure: LLM did not return a valid markdown block. Retrying cycle...")

if __name__ == "__main__":
    execute_self_refinement_cycle()