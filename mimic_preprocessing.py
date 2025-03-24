import pandas as pd
from tqdm import tqdm
from yaml import safe_load
from preprocessing import chunkizer, text_preprocessing


def load_config(file_path: str) -> dict:
    """Load YAML configuration file."""
    with open(file_path, "r") as f:
        return safe_load(f)


def process_notes(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Chunkize the texts and convert the result to a DataFrame of chunks."""
    mimic_chunks = [
        chunkizer(text, config)
        for text in tqdm(df["text"], total=df.shape[0], desc="Chunkizing notes")
    ]
    note_ids = df["note_id"].tolist()
    mimic_chunk_length = [len(c) for c in mimic_chunks]
    repeated_note_ids = [
        note_id
        for count, note_id in zip(mimic_chunk_length, note_ids)
        for _ in range(count)
    ]

    # Flatten the list of chunk lists
    mimic_chunks_flatten = [chunk for chunks in mimic_chunks for chunk in chunks]

    if not mimic_chunks_flatten:
        raise ValueError("No chunks were generated from the texts.")

    columns = (
        mimic_chunks_flatten[0]._fields
        if hasattr(mimic_chunks_flatten[0], "_fields")
        else None
    )
    mimic_chunks_df = pd.DataFrame(mimic_chunks_flatten, columns=columns)
    mimic_chunks_df["note_id"] = repeated_note_ids

    return mimic_chunks_df


def sample_chunks(df: pd.DataFrame, n_per_group: int = 5) -> pd.DataFrame:
    """
    Sample n rows per unique keyword.
    """
    return df.groupby("keyword", group_keys=False).apply(
        lambda x: x.sample(n_per_group, random_state=42)
    )


def apply_text_processing(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Apply text preprocessing for both the raw text copy and the processed text.
    This function updates the DataFrame with two new columns:
    - 'text_raw': a copy of the original text.
    - 'text_preproc': the result of preprocessing.
    It also updates 'text' with an alternative preprocessing if required.
    """
    df["text_raw"] = df["text"]

    def process_row(row):
        row["text_preproc"] = text_preprocessing(
            row["text"], row["keyword"], text_col="text_preproc", config=config
        )
        row["text"] = text_preprocessing(
            row["text"], row["keyword"], text_col="text", config=config
        )
        return row

    return df.apply(process_row, axis=1)


def process_mimic_data():
    mimic_notes = pd.read_csv("../discharge.csv")
    config_file = load_config("config/keywords.yaml")

    mimic_chunks_df = process_notes(mimic_notes, config_file)

    mimic_chunks_df_sampled = sample_chunks(mimic_chunks_df, n_per_group=5)

    mimic_chunks_df_sampled = apply_text_processing(
        mimic_chunks_df_sampled, config_file
    )

    mimic_chunks_df_sampled.reset_index().to_csv("mimic_discharged_chunks_sampled.csv")


if __name__ == "__main__":
    process_mimic_data()
