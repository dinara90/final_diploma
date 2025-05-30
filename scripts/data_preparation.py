import os
import json
import re
import time
import logging
from pathlib import Path
import fitz
import openai
import tiktoken
from tqdm import tqdm
import pandas as pd
import argparse
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("DDR-DataPrep")

MAX_TOKENS = 4000
OVERLAP_TOKENS = 200

SECTION_LABELS = [
    #"current_operations",
    #"planned_operations",
    #"operations_summary",
    "comments",
    #"management_summary",
    #"safety_information",
    "well_information",
]

def setup_api_key():
    """Retrieves the OpenAI API key from environment variables, loading .env file first."""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key not found. Set the OPENAI_API_KEY environment variable or place it in a .env file.")
    return api_key

def count_tokens(text):
    """Count the number of tokens in a text string."""
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        logger.error(f"Error extracting text from {pdf_path}: {e}")
        return ""
    
def split_text_into_chunks(text, max_tokens=MAX_TOKENS, overlap=OVERLAP_TOKENS):
    """Split text into chunks respecting token limits."""
    if not text:
        return []
    
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    
    chunks = []
    i = 0
    while i < len(tokens):
        end = min(i + max_tokens, len(tokens))
        
        chunk_tokens = tokens[i:end]
        chunk = encoding.decode(chunk_tokens)
        
        if chunk.strip():
            chunks.append(chunk)
        
        i = end - overlap if end < len(tokens) else end
        
    return chunks

# You are an expert in analyzing drilling reports (DDRs) from the oil and gas industry. 
# Below is a chunk of text from a drilling report that has been extracted from a PDF, which may have lost some formatting.

# Your task is to:
# 1. Identify DISTINCT and SEPARATE sections within the text.
# 2. Assign the appropriate label to each section from these options:
#    - current_operations: Current status and activities (often begins with "Current Operations:")
#    - planned_operations: Future plans and upcoming work (often begins with "Planned Operations:")
#    - operations_summary: Detailed timeline of operations (often includes From/To times or a chronological list)
#    - comments: General notes, observations, remarks (often under "Comments" header)
#    - management_summary: Summary of key activities and decisions (often a bullet list of completed work)
#    - safety_information: Safety-related notes and incidents (often has "Safety" in the header)
#    - well_information: Basic well identifying data (name, ID, location, depth, etc.)

# Look for section headers and natural divisions in the text. Be careful to properly separate distinct sections.

# For the drilling report format, pay special attention to these common section markers:
# - "Current Operations:" typically introduces current_operations
# - "Planned Operations:" typically introduces planned_operations
# - "Operations Summary" or tables with times typically indicate operations_summary
# - "Comments" sections are usually clearly labeled
# - "Management Summary" or bulleted summaries indicate management_summary
# - "Safety Summary" or safety statistics indicate safety_information
# - Well identification information at the top of the report is well_information

# Format your response as a JSON array of objects (even if only one section is identified):
# [
#   {{"text": "exact section text 1", "label": "appropriate_label"}},
#   {{"text": "exact section text 2", "label": "appropriate_label"}},
#   ...
# ]

# Maintain the exact text as it appears. Keep section headers with their sections.

# Text to analyze:
# {text_chunk}

def identify_ddr_sections(text_chunk, client):
    """GPT-4 to identify and label sections in a text chunk."""
    prompt = f"""You are an expert in analyzing drilling reports (DDRs) from the oil and gas industry.
    Below is a chunk of text from a drilling report that has been extracted from a PDF, which may have lost some formatting.
    
    Your task is to:
    1. Identify DISTINCT and SEPARATE sections within the text.
    2. Assign the appropriate label to each section from ONLY these two options:
       - comments: General notes, observations, remarks, summaries, operational details (current/planned), or safety information.
       - well_information: Basic well identifying data (name, ID, location, depth, etc.) usually found at the start of the report.
    
    Look for section headers and natural divisions in the text. If a section doesn't clearly fit 'well_information', label it as 'comments'.
    
    Format your response as a JSON array of objects (even if only one section is identified):
    [
      {{"text": "exact section text 1", "label": "appropriate_label"}},
      {{"text": "exact section text 2", "label": "appropriate_label"}},
      ...
    ]
    
    Maintain the exact text as it appears. Keep section headers with their sections.
    
    Text to analyze:
    {text_chunk}
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini-2025-04-14",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            response_format={"type": "json_object"},
            max_tokens=32768,
        )
        
        response_text = response.choices[0].message.content
        logger.debug(f"Response from GPT (first 500 chars): {response_text[:500]}...")
        
        try:
            results = json.loads(response_text)
            
            # Handle various response formats
            if isinstance(results, list):
                # The ideal case - a list of section objects
                return results
            elif isinstance(results, dict):
                if "sections" in results and isinstance(results["sections"], list):
                    return results["sections"]
                elif all(key in results for key in ["text", "label"]):
                    return [results]
                else:
                    extracted_sections = []
                    for key, value in results.items():
                        if isinstance(value, dict) and "text" in value and "label" in value:
                            extracted_sections.append(value)
                        elif isinstance(value, list):
                            for item in value:
                                if isinstance(item, dict) and "text" in item and "label" in item:
                                    extracted_sections.append(item)
                    if extracted_sections:
                        return extracted_sections
            
            logger.warning(f"Unexpected response structure: {response_text[:300]}...")
            return []
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON response: {e}\nResponse: {response_text[:300]}...")
            return []
    except Exception as e:
        logger.error(f"API call error: {e}")
        return []

def validate_section(section):
    """Validate that a section has the correct format and reasonable content."""
    if not isinstance(section, dict):
        logger.warning(f"Skipping invalid section (not a dict): {type(section)}")
        return False
    
    if "text" not in section or "label" not in section:
        logger.warning(f"Skipping invalid section (missing 'text' or 'label'): {str(section)[:100]}...")
        return False
    
    if not isinstance(section["text"], str) or not section["text"].strip():
        logger.warning(f"Skipping invalid section (empty or non-string 'text'): {str(section)[:100]}...")
        return False
    
    if section["label"] not in SECTION_LABELS:
        logger.warning(f"Skipping invalid section (unknown label '{section['label']}'): {str(section)[:100]}...")
        return False
    
    # Avoid very short sections that might be noise
    if len(section["text"]) < 10:
        logger.warning(f"Skipping invalid section (text too short): {str(section)[:100]}...")
        return False
    
    return True

def process_pdf_file(pdf_path, client):
    """Process a PDF file and generate labeled training data."""
    logger.info(f"Processing {pdf_path}")
    
    full_text = extract_text_from_pdf(pdf_path)
    if not full_text:
        logger.error(f"No text extracted from {pdf_path}")
        return []
    
    logger.debug(f"""Sample of extracted text (first 500 chars):
{full_text[:500]}...""")

    chunks = split_text_into_chunks(full_text)
    logger.info(f"Split into {len(chunks)} chunks")
    
    all_sections = []
    
    for i, chunk in enumerate(tqdm(chunks, desc=f"Processing chunks for {Path(pdf_path).name}")):
        sections = identify_ddr_sections(chunk, client)
        logger.info(f"Chunk {i+1}/{len(chunks)}: Found {len(sections)} potential sections from GPT")

        valid_sections = []
        invalid_count = 0
        for section in sections:
            if validate_section(section):
                valid_sections.append(section)
            else:
                invalid_count += 1
        
        if invalid_count > 0:
            logger.warning(f"Skipped {invalid_count} invalid sections in chunk {i+1}/{len(chunks)}")

        all_sections.extend(valid_sections)
        logger.info(f"Added {len(valid_sections)} valid sections from chunk {i+1}. Total accumulated: {len(all_sections)}")
    
    return all_sections

def analyze_results(all_sections):
    """Analyze and report on the labeled data distribution."""
    if not all_sections:
        logger.warning("No sections to analyze")
        return
    
    label_counts = {}
    for section in all_sections:
        label = section["label"]
        label_counts[label] = label_counts.get(label, 0) + 1
    
    if not label_counts:
        return None
        
    summary_df = pd.DataFrame({
        "Label": list(label_counts.keys()),
        "Count": list(label_counts.values())
    })
    summary_df = summary_df.sort_values("Count", ascending=False)
    
    logger.info("\nLabel Distribution:")
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        percentage = (count / len(all_sections)) * 100
        logger.info(f"{label}: {count} ({percentage:.2f}%)")
    
    return summary_df

def main():
    """Main function to process DDR PDFs."""
    parser = argparse.ArgumentParser(description="Process drilling reports for ML training data")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing PDF files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for output files")
    parser.add_argument("--pattern", type=str, default="*.pdf", help="File pattern to match (default: *.pdf)")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        api_key = setup_api_key()
        client = openai.OpenAI(api_key=api_key)
    except ValueError as e:
        logger.error(str(e))
        return
    
    pdf_files = list(Path(args.input_dir).glob(args.pattern))
    if not pdf_files:
        logger.error(f"No PDF files found in {args.input_dir} matching pattern {args.pattern}")
        return
    
    logger.info(f"Found {len(pdf_files)} PDF files to process")
    
    all_sections = []
    for pdf_file in pdf_files:
        sections = process_pdf_file(str(pdf_file), client)
        all_sections.extend(sections)
        
    output_json_path = os.path.join(args.output_dir, "labeled_sections.json")
    logger.info(f"Saving all {len(all_sections)} sections to {output_json_path}")
    try:
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(all_sections, f, indent=4, ensure_ascii=False)
    except IOError as e:
        logger.error(f"Failed to save sections to JSON: {e}")
    
    summary_df = analyze_results(all_sections)
    if summary_df is not None:
        output_csv_path = os.path.join(args.output_dir, "label_distribution_summary.csv")
        logger.info(f"Saving label distribution summary to {output_csv_path}")
        try:
            summary_df.to_csv(output_csv_path, index=False)
        except IOError as e:
            logger.error(f"Failed to save summary to CSV: {e}")
    else:
        logger.warning("Analysis did not produce a summary DataFrame.")
    
    logger.info("Processing complete!")
    
if __name__ == "__main__":
    main()