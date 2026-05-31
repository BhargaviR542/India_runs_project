import os
from src.data_loader import DataLoader
from src.ranker import SemanticTalentMatcher

def run_pipeline():
    # Configure relative paths
    root_dir = os.path.dirname(os.path.abspath(__file__))
    jd_file = os.path.join(root_dir, "data", "job_description.json")
    candidates_file = os.path.join(root_dir, "data", "candidates.json")
    output_destination = os.path.join(root_dir, "output", "ranked_candidates.csv")
    
    print("\n" + "="*60)
    print(" REDROB AI #INDIARUNS MATCHING ENGINE RUNTIME ENVIRONMENT")
    print("="*60 + "\n")
    
    # Extract operational configurations
    try:
        jd_payload = DataLoader.load_json(jd_file)
        candidates_payload = DataLoader.load_json(candidates_file)
    except Exception as error:
        print(f"[Fatal Runtime Block] Ingestion Pipeline Halted: {error}")
        return

    # Initialize context ranking framework
    engine = SemanticTalentMatcher()
    
    # Run the pipeline
    results = engine.process(jd_payload, candidates_payload, top_n=50)
    
    # Export results
    os.makedirs(os.path.dirname(output_destination), exist_ok=True)
    results.to_csv(output_destination, index=False)
    
    print("\n" + "-"*60)
    print(f"[Success] Match execution completed. System output saved to: {output_destination}")
    print("-"*60)
    print(results.head(5).to_string(index=False))
    print("="*60 + "\n")

if __name__ == "__main__":
    run_pipeline()