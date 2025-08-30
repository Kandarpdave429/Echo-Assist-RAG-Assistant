import json
from datasets import Dataset
from ragas_eval import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision
)

# Load the ragas_eval.json file
with open("ragas_eval.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Convert list of dicts into HuggingFace Dataset
dataset = Dataset.from_list(data)

# Run evaluation with Ragas metrics
print("🔍 Running evaluation...")
result = evaluate(
    dataset=dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_recall,
        context_precision
    ]
)

# Print the results
print("\n✅ Evaluation Completed!")
print(result)

# Save the results to a file
result.save_json("ragas_results.json")
print("📄 Results saved to ragas_results.json")
