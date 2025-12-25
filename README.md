# Project Gnazim

## Description
The Ganazim Institute is a literary research institute that contains the largest archive of Hebrew literature in the world. The archive contains about 800 archival collections of writers, poets, playwrights, thinkers, editors, and writers from the end of the 19th century to the present day. It includes manuscripts, letters, personal documents, photos, and a unique recording collection.

<p float="left">
  <img src="https://github.com/YarinBeni/DHC--Gnazim-Project/blob/main/data_images_examples/POC_sample2_withnoise%20(1).png?raw=true" width="350" height="250" alt="Noisy OCR Sample Image" />
  <img src="https://github.com/YarinBeni/DHC--Gnazim-Project/blob/main/data_images_examples/POC_sample3_handwriten.png?raw=true" width="350" height="250" alt="Handwritten OCR Sample Image" />
</p>

*Figures: Contrasting examples from the archive. The first is a clean document suitable for standard OCR; the second is a complex handwritten document requiring specialized HTR.*

## Project Goals
The goal of the project is to extract metadata and Hebrew text from the 180,000 scanned images in the Ganazim Institute archive and make it accessible for research.

## Methodology: A Two-Phase Approach

The project evolved from a classical machine learning pipeline to a benchmarking study against modern Large Language Models (LLMs).

### Phase 1: Specialized HTR & Layout Development
In this phase, we built a robust pipeline for ground truth creation and model training.
- **Data Acquisition**: `main.py` orchestrates the traversal of Google Drive, downloading images and extracting initial metadata from folder structures.
- **eScriptorium Integration**: We utilized the eScriptorium platform for data labeling. The notebook `Additions_to_Daniels_basic_functions_4_eScriptorium_API.ipynb` was developed to interact with the eScriptorium API, automating the management of documents, regions, and transcriptions. This allowed for precise refinement of segmentation and transcription workflows.
- **Kraken Training**: Using the labeled data, we trained specialized **Kraken** models for page layout analysis and Handwritten Text Recognition (HTR). These models were fine-tuned specifically for the unique scripts and layouts found in the Gnazim archive.

### Phase 2: LLM Benchmarking (OpenAI)
In the second phase, we began benchmarking OpenAI's vision models (GPT-4o) against our specialized Kraken models.
- **Semantic Extraction**: We tested the ability of zero-shot LLMs to perform semantic layout analysis—identifying fields like `author`, `title`, `date`, and `page_number` directly from images.
- **Initial Findings**: While the zero-shot results were highly impressive and demonstrated a strong semantic understanding of the documents, the classical machine learning models (Kraken) fine-tuned on our specific dataset performed slightly better overall in terms of raw HTR accuracy and precise layout segmentation.
- **Status**: This phase is in its early stages. The results suggest that while specialized models currently lead, LLMs are a viable path, especially if moved toward few-shot configurations or more complex prompting strategies.

## Pseudo Code Logic of main.py
1. **GCP Connection**: Initialize GCP Connection via `pydrive2`.
2. **Queue Initialization**: Initialize folder queue and set processing thresholds.
3. **Main Processing Loop**:
   - Dequeue folder and retrieve processed/problematic file logs.
   - **File Processing**:
     - Extract metadata from path (`gcp_extract_years_author_type`).
     - Download image and detect paragraphs (`ParagraphDetector.py`).
     - Extract text using initial OCR and log results.
   - **Post-Processing**: Update local CSVs and perform function profiling.
4. **Finalization**: Generate GCP links and save the final metadata database.

