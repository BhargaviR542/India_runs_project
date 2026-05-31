import os
from src.data_loader import DataLoader
from src.ranker import SemanticTalentMatcher

def run_pipeline():
    # Configure relative paths
    root_dir = os.path.dirname(os.path.abspath(__file__))
    jd_file = os.path.join(root_dir, "data", "job_description.json")
    candidates_file = os.path.join(root_dir, "data", "candidates.json")
    output_destination = os.path.join(root_dir, "output", "ranked_candidates.csv")
    
    print("\n" + "="*75)
    print(" REDROB AI #INDIARUNS MATCHING ENGINE RUNTIME ENVIRONMENT")
    print("="*75 + "\n")
    
    # Extract operational configurations
    try:
        jd_payload = DataLoader.load_json(jd_file)
        candidates_payload = DataLoader.load_json(candidates_file)
    except Exception as error:
        print(f"[Fatal Runtime Block] Ingestion Pipeline Halted: {error}")
        return

    # Initialize context ranking framework
    engine = SemanticTalentMatcher()
    
    # Run the pipeline to get the dataframe
    results = engine.process(jd_payload, candidates_payload, top_n=50)
    
    # Export results to CSV file safely
    os.makedirs(os.path.dirname(output_destination), exist_ok=True)
    results.to_csv(output_destination, index=False)
    
    print("\n" + "-"*75)
    print(f"[Success] Match execution completed. Output saved to: {output_destination}")
    print("-"*75 + "\n")
    
    # ====================================================================
    # THE PRINT FORMATTING FIX: Clean, Beautiful Tabular Console Output
    # ====================================================================
    print(" " + "_" * 73)
    # Header format string adjusting spaces perfectly
    print(f"| {'Rank':<6} | {'Candidate ID':<14} | {'Candidate Name':<25} | {'Match Score':<15} |")
    print("|" + "="*8 + "|" + "="*16 + "|" + "="*27 + "|" + "="*17 + "|")
    
    for _, row in results.iterrows():
        rank = row['rank']
        c_id = row['candidate_id']
        name = row['name']
        score = f"{row['match_score_pct']}% Match"
        
        # Row data formatting
        print(f"| {rank:<6} | {c_id:<14} | {name:<25} | {score:<15} |")
        print("|" + "-"*8 + "+" + "-"*16 + "+" + "-"*27 + "+" + "-"*17 + "|")
        
    print("\n" + "="*75 + "\n")

if __name__ == "__main__":
    run_pipeline()