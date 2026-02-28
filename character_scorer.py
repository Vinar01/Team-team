import pandas as pd
import numpy as np

def compute_squared_mad_penalties(row, epsilon=1e-5):
    """
    Computes the squared MAD penalty for 5 transcription options.
    Handles empty/null strings gracefully.
    """
    try:
        # Extract options, ensuring any NaN/null values become empty strings 
        # to prevent 'nan' being counted as a 3-character string.
        options = [
            str(row['option_1']) if pd.notna(row['option_1']) else "",
            str(row['option_2']) if pd.notna(row['option_2']) else "",
            str(row['option_3']) if pd.notna(row['option_3']) else "",
            str(row['option_4']) if pd.notna(row['option_4']) else "",
            str(row['option_5']) if pd.notna(row['option_5']) else ""
        ]
        
        # 1. Compute character counts (C_i)
        C = np.array([len(opt) for opt in options], dtype=float)
        
        # 2. Compute median (m)
        m = np.median(C)
        
        # 3. Compute absolute deviations and MAD
        abs_deviations = np.abs(C - m)
        mad = np.median(abs_deviations)
        
        # 4. Compute L_i
        L = -(abs_deviations / (mad + epsilon))
        
        # 5. Apply the square (S_i)
        S = L ** 2
        
        return pd.Series(
            S, 
            index=['char_score_1', 'char_score_2', 'char_score_3', 'char_score_4', 'char_score_5']
        )
        
    except Exception as e:
        # Failsafe: If a row is completely corrupted, return 0.0 scores so the pipeline continues
        print(f"Error processing row {row.get('audio_id', 'Unknown')}: {e}")
        return pd.Series(
            [0.0, 0.0, 0.0, 0.0, 0.0], 
            index=['char_score_1', 'char_score_2', 'char_score_3', 'char_score_4', 'char_score_5']
        )

# ==========================================
# Phase 3 Execution Pipeline
# ==========================================

def run_character_scoring_pipeline(input_csv_path, output_csv_path):
    print(f"Loading data from {input_csv_path}...")
    
    # Load the dataset
    df = pd.read_csv(input_csv_path)
    
    # Verify the necessary columns exist
    required_cols = ['option_1', 'option_2', 'option_3', 'option_4', 'option_5']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Input CSV is missing one of the required columns: {required_cols}")
    
    print("Calculating squared MAD penalties...")
    # Apply the vectorized scoring function to the dataframe
    score_columns = ['char_score_1', 'char_score_2', 'char_score_3', 'char_score_4', 'char_score_5']
    df[score_columns] = df.apply(compute_squared_mad_penalties, axis=1)
    
    # Save the enriched dataset for the next pipeline stage
    df.to_csv(output_csv_path, index=False)
    print(f"Success! Scored data saved to {output_csv_path}")
    
    # Preview the results
    print("\nPreview of the calculated scores:")
    print(df[['option_1', 'char_score_1', 'option_2', 'char_score_2']].head())

if __name__ == "__main__":
    # Point this to the file downloaded from your SharePoint link
    INPUT_FILE = "transcription_assessment.csv" 
    OUTPUT_FILE = "transcription_assessment_scored.csv"
    
    # Run the pipeline
    run_character_scoring_pipeline(INPUT_FILE, OUTPUT_FILE)