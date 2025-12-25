# Project Gnazim - AI Coding Instructions

## Big Picture Architecture
Project Gnazim is a literary research pipeline for extracting text and metadata from the Gnazim Institute's Hebrew archive.
- **Data Flow**: Google Drive (GCP) -> Local Processing -> eScriptorium (Annotation/HTR) -> Kraken (Model Training) -> LLM Benchmarking.
- **Core Pipeline**: `main.py` orchestrates folder traversal on Google Drive, image downloading, and initial OCR.
- **Image Processing**: `ParagraphDetector.py` uses OpenCV and PyTesseract to detect text regions (ROIs) and extract Hebrew text.
- **HTR & Annotation**: eScriptorium is used for ground truth creation. `Additions_to_Daniels_basic_functions_4_eScriptorium_API.ipynb` provides an API toolkit for automated segmentation refinement.
- **Data Formats**: Uses **ALTO XML** for HTR ground truth and layout information.
- **LLM Benchmarking**: Phase 2 involves comparing specialized Kraken models against OpenAI's zero-shot vision models for semantic layout analysis (see `OpenAI_Data/`).

## Critical Developer Workflows
- **GCP Pipeline**: Run `main.py` to process images from Google Drive. It uses `pydrive2` and requires `client_secrets.json` in the root directory.
- **HTR Training**: Use `Compile Kraken Dataset.ipynb` to prepare ALTO XML and images for Kraken training.
- **eScriptorium Integration**: Use `escriptorium-connector` or the `Additions_to_Daniels...` notebook for advanced API-based document management.
- **LLM Benchmarking**: Compare Kraken outputs with OpenAI JSON results in `OpenAI_Data/` to evaluate semantic extraction accuracy.
- **Debugging**: Many functions use a custom `@profile` decorator (defined in `main.py`) to track execution time and call counts.

## Project-Specific Conventions
- **Language**: Primary focus is **Hebrew** (`lang='heb'`). Ensure RTL (Right-to-Left) support in all text processing.
- **Metadata**: Extracted from folder structures using `gcp_extract_years_author_type` in `main.py`.
- **Data Storage**: Results are tracked in CSV files (e.g., `gnazim_db_meta_data_2k_fixed_coords.csv`).
- **Error Handling**: Problematic files are logged to a separate CSV (e.g., `gnazim_db_problem_folders.csv`) to allow resuming.
- **Benchmarking Insights**: Specialized Kraken models currently outperform zero-shot LLMs in raw HTR, but LLMs show strong semantic potential.
- **Profiling**: Use the `@profile` decorator for new processing functions to maintain performance visibility.

## Key Files & Directories
- `main.py`: Entry point for the GCP-to-OCR pipeline.
- `ParagraphDetector.py`: Core image processing and ROI detection logic.
- `*.mlmodel`: CoreML/Kraken models for segmentation and HTR.
- `HTR_pages/`: Contains ALTO XML files for ground truth.
- `training_data/`: Prepared datasets for Kraken.

## External Dependencies
- `pydrive2`: Google Drive API.
- `pytesseract`: Tesseract OCR engine.
- `opencv-python`: Image processing.
- `escriptorium-connector`: eScriptorium API.
