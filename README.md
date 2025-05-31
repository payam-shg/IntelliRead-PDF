# IntelliRead-PDF
### AI-Powered Summarization, Explanation, and Translation of PDF Files<br/>
A versatile Python script for processing PDF documents using various text extraction methods and AI services for summarization, explanation, and translation.

## 🌟 Features

* **Multiple Text Extraction Options:**
    * PyMuPDF (for text-based PDFs)
    * Tesseract OCR (local OCR for image-based PDFs)
    * OCR.space API (cloud-based OCR)
* **AI-Powered Content Processing:**
    * **Summarization:** Generate concise summaries of PDF content.
    * **Explanation:** Get detailed explanations of the text.
    * **Translation:** Translate content (e.g., to Persian and other configured languages).
* **Support for Various AI Services:**
    * Local: LM Studio (via SDK), Ollama
    * Cloud: Groq, OpenAI (Official), OpenRouter, SiliconFlow, Google Gemini
* **Flexible Configuration:**
    * User-friendly command-line interface to select services for each task.
    * Configuration via a `.env` file for API keys, model names, and other settings.
* **Output Management:**
    * Saves summaries and explanations in both `.txt` and `.pdf` formats.
    * Organized output into `pdf_summaries` and `pdf_explanations` folders.
* **Customizable PDF Output:** Supports custom fonts for generated PDF files.

## ⚙️ Prerequisites

* Python 3.8 or higher.
* [Tesseract OCR Engine](https://github.com/tesseract-ocr/tesseract#installing-tesseract) installed on your system (if you plan to use the Tesseract OCR option). Make sure to install language data (e.g., for English, Persian).
* Active internet connection for cloud-based AI and OCR services.
* LM Studio application running if using the "LM Studio (Local SDK)" option, with the desired model downloaded and loaded.
* Ollama service running if using the "Local Ollama" option, with the desired model pulled.

## 🚀 Setup & Installation

1.  **Clone the repository (if applicable) or download the script files.**
    ```bash
    # git clone https://github.com/payam-shg/IntelliRead-PDF.git
    # cd IntelliRead-PDF
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv IntelliRead
    # Windows
    IntelliRead\Scripts\activate
    # macOS/Linux
    source IntelliRead/bin/activate
    ```

3.  **Install dependencies:**<br/>
    Install following libraries:
    ```txt
    requests
    python-dotenv
    PyMuPDF
    Pillow
    pytesseract
    tqdm
    reportlab
    google-generativeai
    groq
    openai
    lmstudio
    ```
    Or run:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up the Environment File (`.env`):**
    * Modofy `.env` in the root directory of the project.

    **Example:**
    ```env
    # AI Service API Keys
    GROQ_API_KEY = your_groq_api_key_here
    OPENAI_API_KEY = your_openai_api_key_here
    OPENROUTER_API_KEY = your_openrouter_api_key_here
    SILICONFLOW_API_KEY = your_siliconflow_api_key_here
    GOOGLE_API_KEY = your_google_api_key_here
    OCR_SPACE_API_KEY = your_ocr_space_api_key_here

    # Model Configurations (Repository IDs or specific names)
    OLLAMA_MODEL = llama3.1:8b
    GROQ_CHAT_MODEL = meta-llama/llama-4-scout-17b-16e-instruct
    OPENAI_CHAT_MODEL = gpt-4o-mini
    OPENROUTER_CHAT_MODEL = deepseek/deepseek-r1:free
    SILICONFLOW_CHAT_MODEL = Qwen/Qwen3-8B
    GOOGLE_GEMINI_MODEL = gemma-3-27b-it
    LM_STUDIO_MODEL = gemma-3-12b-it-qat

    # Font Configuration for PDF Output (Optional)
    # PDF_OUTPUT_FONT_NAME = tahoma
    # PDF_OUTPUT_FONT_PATH = ./fonts/tahoma.ttf # Example path, ensure font file exists

    # Tesseract Command Path (Windows - if Tesseract not in PATH)
    # TESSERACT_CMD_PATH = C:\Program Files\Tesseract-OCR\tesseract.exe
    ```

## 🛠️ Usage

1.  **Place your PDF files** into the `input_pdfs` folder (create it if it doesn't exist).
2.  **Run the main script** from your terminal:
    ```bash
    python app.py 
    ```

3.  **Follow the interactive prompts** to select:
    * PDF text extraction method (PyMuPDF, Tesseract, OCR.space).
    * AI service for Summarization.
    * AI service for Explanation.
    * Target language for Translation (or "No Translation").
    * AI service for Translation (if a target language is selected).

4.  **Outputs** will be saved in:
    * `./pdf_summaries/` (for summaries in `.txt` and `.pdf`)
    * `./pdf_explanations/` (for explanations in `.txt` and `.pdf`)

## 🔧 Configuration Details

* **API Keys:** Must be set in the `.env` file for the respective cloud services to work.
* **Model Names/IDs:** Default models are set in the script, but can be overridden by setting the corresponding environment variables in `.env` (e.g., `OLLAMA_MODEL`, `LM_STUDIO_MODEL`). For LM Studio SDK, `LM_STUDIO_MODEL` should be the **Repository ID** of the model as shown in LM Studio.
* **Font for PDF Output:** You can specify `PDF_OUTPUT_FONT_NAME` and `PDF_OUTPUT_FONT_PATH` in `.env` to use a custom font for generated PDFs. Ensure the `.ttf` font file is accessible.
* **Tesseract Language Data:** For Tesseract OCR, ensure you have the necessary language data files (e.g., `eng.traineddata`, `fas.traineddata`) in your Tesseract `tessdata` directory.

## 💡 Troubleshooting Tips

* **LM Studio Connection Error (SDK):**
    * Ensure the LM Studio application is running.
    * Ensure the model specified by `LM_STUDIO_MODEL` (repository ID) is downloaded in LM Studio.
    * Check LM Studio's server logs for any errors on its side.
* **Tesseract Errors:**
    * Make sure Tesseract OCR engine is installed correctly and its command is accessible (either in PATH or `TESSERACT_CMD_PATH` in `.env` for Windows).
    * Verify that the `tessdata` directory is correctly configured (often via `TESSDATA_PREFIX` environment variable if not in default location) and contains the `.traineddata` files for the languages you specify.
* **API Key Errors:** Double-check that your API keys in the `.env` file are correct and have not expired or exceeded quotas.
* **Dependency Issues:** Ensure all libraries in `requirements.txt` are installed in your active Python environment.
