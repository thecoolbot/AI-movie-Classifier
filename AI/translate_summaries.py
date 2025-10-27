import pandas as pd
from googletrans import Translator
from tqdm import tqdm
import time

def split_text(text, max_len=5000):
    """
    Splits text into chunks of max_len characters without cutting words.
    """
    chunks = []
    while len(text) > max_len:
        split_index = text.rfind(' ', 0, max_len)
        if split_index == -1:
            split_index = max_len
        chunks.append(text[:split_index])
        text = text[split_index:].strip()
    if text:
        chunks.append(text)
    return chunks

def translate_text(text, lang_code, translator):
    """
    Safely translates text, handling chunking and API errors.
    """
    try:
        chunks = split_text(text)
        translated_chunks = []
        for chunk in chunks:
            translated_chunk = translator.translate(chunk, dest=lang_code).text
            translated_chunks.append(translated_chunk)
            time.sleep(1)  # Prevent rate-limiting
        return ' '.join(translated_chunks)
    except Exception as e:
        print(f"Error translating to {lang_code}: {e}")
        return ''

def translate_summaries(input_file, output_file, num_summaries=50):
    """
    Translate movie summaries into Arabic, Urdu, and Korean using tqdm and chunk handling.
    """
    df = pd.read_csv(input_file)
    df = df.head(num_summaries)

    translator = Translator()
    languages = {'Arabic': 'ar', 'Urdu': 'ur', 'Korean': 'ko'}

    for lang in languages:
        df[f'Summary_{lang}'] = ''

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Translating Summaries"):
        summary = str(row.get('Cleaned_Summary', '')).strip()
        if not summary:
            continue
        for lang_name, lang_code in languages.items():
            translated = translate_text(summary, lang_code, translator)
            df.at[idx, f'Summary_{lang_name}'] = translated

    df.to_csv(output_file, index=False)
    return df

if __name__ == "__main__":
    input_file = "data/cleaned_dataset.csv"
    output_file = "data/translated_summaries.csv"

    translated_df = translate_summaries(input_file, output_file)
    print(f"\n✅ Translated summaries saved to '{output_file}' with {len(translated_df)} records.")
