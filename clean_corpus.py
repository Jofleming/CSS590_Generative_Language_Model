import sys
import string
import re

def clean_corpus(input_path, output_path):
    """
    Reads a text file, removes special characters not in a defined whitelist,
    outputs the before/after vocabulary sizes, and saves the cleaned text.
    """
    print(f"Reading from: {input_path}")
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            original_text = f.read()
    except FileNotFoundError:
        print(f"Error: Could not find {input_path}")
        return

    # Remove text between square brackets (handling potential nesting)
    prev_text = None
    while original_text != prev_text:
        prev_text = original_text
        original_text = re.sub(r'\[[^\[\]]*\]', '', original_text)
        
    # Calculate original character-level vocabulary
    original_vocab = set(original_text)
    print(f"Original vocabulary size: {len(original_vocab)}")
    
    # Replace typeset quotes with ordinary ASCII quotes
    original_text = original_text.replace('“', '"').replace('”', '"')
    original_text = original_text.replace('‘', "'").replace('’', "'")

    # Define allowed characters: letters, digits, standard whitespace, and basic punctuation
    allowed_chars = set(string.ascii_letters + string.digits + " \n\t.,!?'\"-;:(){}")
    
    # Filter out any character not in the whitelist
    cleaned_text = ''.join(c for c in original_text if c in allowed_chars)

    # Calculate cleaned character-level vocabulary
    cleaned_vocab = set(cleaned_text)
    print(f"Cleaned vocabulary size: {len(cleaned_vocab)}")
    
    # Identify exactly which characters were stripped out
    removed_chars = original_vocab - cleaned_vocab
    print(f"Removed characters: {repr(''.join(removed_chars))}")

    # Save the cleaned output
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(cleaned_text)
        
    print(f"Cleaned corpus successfully saved to: {output_path}")

if __name__ == "__main__":
    # Set default paths based on the Generative AI project structure
    input_file = "../../Transformers/Pride-and-Prejudice.txt"
    output_file = "../../Transformers/Pride-and-Prejudice-Cleaned.txt"
    
    if len(sys.argv) == 3:
        input_file, output_file = sys.argv[1], sys.argv[2]
        
    clean_corpus(input_file, output_file)