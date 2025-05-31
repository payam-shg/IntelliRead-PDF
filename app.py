import os
import time
import requests
import json
from dotenv import load_dotenv
import fitz
from tqdm import tqdm
import io
import traceback
import re

# PDF Generation Libraries
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.lib.enums import TA_LEFT
except ImportError:
    print("ReportLab library not found. PDF generation will be disabled. Run: pip install reportlab")
    SimpleDocTemplate = Paragraph = Spacer = getSampleStyleSheet = pdfmetrics = TTFont = None
    TA_LEFT = 0

# AI Service Libraries
try:
    import google.generativeai as genai
except ImportError:
    print("Google Generative AI library not found. Run: pip install google-generativeai")
    genai = None
try:
    from groq import Groq
except ImportError:
    print("Groq library not found. Run: pip install groq")
    Groq = None
try:
    from openai import OpenAI
except ImportError:
    print("OpenAI library not found (used for OpenAI, OpenRouter, SiliconFlow). Run: pip install openai")
    OpenAI = None
try:
    import lmstudio as lms
except ImportError:
    print("lmstudio library not found. LM Studio (Local SDK) will be unavailable. Run: pip install lmstudio")
    lms = None

# OCR Libraries
try:
    from PIL import Image
except ImportError:
    print("Pillow library not found. It's required for OCR. Run: pip install Pillow")
    Image = None
try:
    import pytesseract
except ImportError:
    print("pytesseract library not found. It's required for Tesseract OCR. Run: pip install pytesseract")
    pytesseract = None

load_dotenv()

# ========== API Keys & Configuration ========== #
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OCR_SPACE_API_KEY = os.getenv("OCR_SPACE_API_KEY")

# Model Names
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
GROQ_CHAT_MODEL = os.getenv("GROQ_CHAT_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
OPENROUTER_CHAT_MODEL = os.getenv("OPENROUTER_CHAT_MODEL", "deepseek/deepseek-r1:free")
SILICONFLOW_CHAT_MODEL = os.getenv("SILICONFLOW_CHAT_MODEL", "Qwen/Qwen3-8B")
GOOGLE_GEMINI_MODEL = os.getenv("GOOGLE_GEMINI_MODEL", "gemma-3-27b-it")
LM_STUDIO_MODEL = os.getenv("LM_STUDIO_MODEL", "gemma-3-12b-it-qat")

OLLAMA_API_URL = "http://localhost:11434/api/generate"
OCR_SPACE_API_URL = "https://api.ocr.space/parse/image"
SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"

INPUT_FOLDER_PDF = "./input_pdfs"
SUMMARIES_FOLDER = "./pdf_summaries"
EXPLANATIONS_FOLDER = "./pdf_explanations"

DEFAULT_PDF_FONT_NAME = "Tahoma"
DEFAULT_PDF_FONT_PATH = "tahoma.ttf"
PDF_OUTPUT_FONT_NAME = os.getenv("PDF_OUTPUT_FONT_NAME", DEFAULT_PDF_FONT_NAME)
PDF_OUTPUT_FONT_PATH = os.getenv("PDF_OUTPUT_FONT_PATH", DEFAULT_PDF_FONT_PATH)

os.makedirs(INPUT_FOLDER_PDF, exist_ok=True)
os.makedirs(SUMMARIES_FOLDER, exist_ok=True)
os.makedirs(EXPLANATIONS_FOLDER, exist_ok=True)

if TTFont and PDF_OUTPUT_FONT_PATH:
    try:
        pdfmetrics.registerFont(TTFont(PDF_OUTPUT_FONT_NAME, PDF_OUTPUT_FONT_PATH))
        print(f"Font '{PDF_OUTPUT_FONT_NAME}' loaded from '{PDF_OUTPUT_FONT_PATH}' for PDF output.")
    except Exception as e:
        print(f"Error loading font '{PDF_OUTPUT_FONT_NAME}' from '{PDF_OUTPUT_FONT_PATH}'. Error: {e}")
        if PDF_OUTPUT_FONT_NAME != "Helvetica":
            print(f"Attempting to load default font '{DEFAULT_PDF_FONT_NAME}' from '{DEFAULT_PDF_FONT_PATH}'.")
            try:
                if os.path.exists(DEFAULT_PDF_FONT_PATH):
                    pdfmetrics.registerFont(TTFont(DEFAULT_PDF_FONT_NAME, DEFAULT_PDF_FONT_PATH))
                    PDF_OUTPUT_FONT_NAME = DEFAULT_PDF_FONT_NAME
                    print(f"Default font '{DEFAULT_PDF_FONT_NAME}' loaded for PDF output.")
                else:
                    print(f"Default font file '{DEFAULT_PDF_FONT_PATH}' not found. Using Helvetica.")
                    PDF_OUTPUT_FONT_NAME = "Helvetica"
            except Exception as e2:
                print(f"Error loading default font '{DEFAULT_PDF_FONT_NAME}'. Using Helvetica. Error: {e2}")
                PDF_OUTPUT_FONT_NAME = "Helvetica"
        else:
            print("Using Helvetica as fallback font.")
else:
    if not TTFont: print("ReportLab not available, PDF font registration skipped.")
    elif not PDF_OUTPUT_FONT_PATH: print(f"PDF_OUTPUT_FONT_PATH for '{PDF_OUTPUT_FONT_NAME}' is not set. Using Helvetica.")
    PDF_OUTPUT_FONT_NAME = "Helvetica"

class TextProcessingResult:
    def __init__(self, text, status="completed", error_message=None):
        self.text = text
        self.status = status if not error_message else "error"
        self.error_message = error_message

# AI Clients
groq_client = None
openai_client = None
openrouter_client = None
siliconflow_client = None
google_gemini_model_client = None
ollama_is_available = True
lm_studio_llm_model_instance = None

def get_user_choice(prompt_message, options_dict):
    print(f"\n{prompt_message}")
    for key, (value, description) in options_dict.items():
        print(f"  {key}. {description} (Code/ID: {value})")
    while True:
        choice_num = input("Enter the number of your choice: ")
        if choice_num in options_dict:
            return options_dict[choice_num][0]
        else:
            print("Invalid choice. Please enter a number from the list above.")

def get_description_from_code(options_dict, code_to_find):
    for _key, (code, description) in options_dict.items():
        if code == code_to_find:
            return description
    return "Unknown"

def initialize_ai_clients(service_keys_to_init):
    global groq_client, openai_client, openrouter_client, siliconflow_client, google_gemini_model_client, lm_studio_llm_model_instance, ollama_is_available

    if "groq" in service_keys_to_init and Groq and GROQ_API_KEY:
        try: groq_client = Groq(api_key=GROQ_API_KEY); print("Groq client initialized.")
        except Exception as e: print(f"Failed to initialize Groq: {e}")
    if "openai" in service_keys_to_init and OpenAI and OPENAI_API_KEY:
        try: openai_client = OpenAI(api_key=OPENAI_API_KEY); print("OpenAI (official) client initialized.")
        except Exception as e: print(f"Failed to initialize OpenAI (official): {e}")
    if "openrouter" in service_keys_to_init and OpenAI and OPENROUTER_API_KEY:
        try: openrouter_client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY); print("OpenRouter client initialized.")
        except Exception as e: print(f"Failed to initialize OpenRouter: {e}")
    if "siliconflow" in service_keys_to_init and OpenAI and SILICONFLOW_API_KEY:
        try: siliconflow_client = OpenAI(base_url=SILICONFLOW_BASE_URL, api_key=SILICONFLOW_API_KEY); print("SiliconFlow client initialized.")
        except Exception as e: print(f"Failed to initialize SiliconFlow: {e}")
    if "google_gemini" in service_keys_to_init and genai and GOOGLE_API_KEY:
        try: genai.configure(api_key=GOOGLE_API_KEY); google_gemini_model_client = genai.GenerativeModel(GOOGLE_GEMINI_MODEL); print(f"Google Gemini client initialized ({GOOGLE_GEMINI_MODEL}).")
        except Exception as e: print(f"Failed to initialize Google Gemini: {e}")
    
    if "lm_studio_local" in service_keys_to_init:
        if lms and LM_STUDIO_MODEL:
            try:
                print(f"LM Studio SDK: Attempting to load model '{LM_STUDIO_MODEL}'...")
                lm_studio_llm_model_instance = lms.llm(LM_STUDIO_MODEL)
                print(f"LM Studio SDK: Model '{LM_STUDIO_MODEL}' instance created/loaded successfully.")
            except Exception as e:
                tb_str = traceback.format_exc()
                print(f"LM Studio SDK: Failed to load model '{LM_STUDIO_MODEL}'. Error: {type(e).__name__} - {e}")
                print(f"LM Studio SDK init traceback:\n{tb_str}")
                lm_studio_llm_model_instance = None
        elif not lms:
            print("LM Studio SDK: 'lmstudio' library not found. Please install it (pip install lmstudio).")
        elif not LM_STUDIO_MODEL:
            print("LM Studio SDK: LM_STUDIO_MODEL (repository ID) not set in .env file.")
            print("Please set LM_STUDIO_MODEL to the repository ID of your model in LM Studio (e.g., NousResearch/Hermes-2-Pro-Llama-3-8B-GGUF).")
    
    if "ollama_local" in service_keys_to_init:
        try:
            requests.get(OLLAMA_API_URL.replace("/api/generate", "/"), timeout=5)
            print("Ollama server seems to be running.")
            ollama_is_available = True
        except requests.exceptions.ConnectionError:
            print("Ollama server not found or not running at the specified URL. Ollama services will be unavailable.")
            ollama_is_available = False
        except Exception as e:
            print(f"Error checking Ollama server status: {e}")
            ollama_is_available = False

# ========== PDF Text Extraction Methods ========== #
def _extract_text_pymupdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = "".join(page.get_text() for page in doc)
        doc.close()
        if not text.strip(): tqdm.write(f"Warning: No text extracted by PyMuPDF from {os.path.basename(pdf_path)} or text is empty.")
        return text
    except Exception as e:
        tqdm.write(f"Error extracting text with PyMuPDF from '{os.path.basename(pdf_path)}': {e}")
        return None

def _convert_page_to_image_bytes(page, dpi=150, output_format="jpeg"):
    try:
        pix = page.get_pixmap(dpi=dpi)
        actual_image_bytes = pix.tobytes(output_format.lower())
        return actual_image_bytes, output_format.lower()
    except Exception as e:
        tqdm.write(f"Error converting PDF page to image (format: {output_format}, dpi: {dpi}): {e}")
        return None, None

def _extract_text_tesseract(pdf_path, lang_code='eng'):
    if not pytesseract or not Image:
        tqdm.write("Tesseract OCR or Pillow library not available. Skipping Tesseract extraction.")
        return None
    full_text_parts = []
    try:
        doc = fitz.open(pdf_path)
        for page_num in tqdm(range(len(doc)), desc=f"Tesseract OCR ({os.path.basename(pdf_path)})", leave=False, unit="page"):
            page = doc.load_page(page_num)
            image_actual_bytes, _ = _convert_page_to_image_bytes(page, dpi=300, output_format="png")
            if image_actual_bytes:
                try:
                    pil_image = Image.open(io.BytesIO(image_actual_bytes))
                    page_text_content = pytesseract.image_to_string(pil_image, lang=lang_code)
                    if page_text_content.strip(): full_text_parts.append(page_text_content.strip())
                except Exception as ocr_err:
                    tqdm.write(f"Tesseract OCR error on page {page_num+1} of {os.path.basename(pdf_path)}: {ocr_err}")
            else: tqdm.write(f"Could not convert page {page_num+1} of {os.path.basename(pdf_path)} to image for Tesseract.")
        doc.close()
        if not full_text_parts:
            tqdm.write(f"No text could be extracted from '{os.path.basename(pdf_path)}' using Tesseract.")
            return None
        return "\n\n".join(full_text_parts)
    except Exception as e:
        tqdm.write(f"Error processing PDF with Tesseract for '{os.path.basename(pdf_path)}': {e}")
        tqdm.write(traceback.format_exc())
        return None

def _extract_text_ocr_space(pdf_path, api_key, lang_code='eng'):
    if not api_key:
        tqdm.write("OCR.space API key not provided. Skipping.")
        return None
    full_text_parts = []
    try:
        doc = fitz.open(pdf_path)
        for page_num in tqdm(range(len(doc)), desc=f"OCR.space ({os.path.basename(pdf_path)})", leave=False, unit="page"):
            page = doc.load_page(page_num)
            image_actual_bytes, image_format_str = _convert_page_to_image_bytes(page, dpi=150, output_format="jpeg")
            if image_actual_bytes and image_format_str:
                payload = {'apikey': api_key, 'language': lang_code, 'isOverlayRequired': "false", 'detectOrientation': "true"}
                filename_for_upload = f'page.{image_format_str}'
                mimetype_for_upload = f'image/{image_format_str}'
                files_to_send = {'file': (filename_for_upload, image_actual_bytes, mimetype_for_upload)}
                try:
                    r = requests.post(OCR_SPACE_API_URL, files=files_to_send, data=payload, timeout=60)
                    r.raise_for_status()
                    result = r.json()
                    if result.get('IsErroredOnProcessing'):
                        error_messages = result.get('ErrorMessage', ['Unknown processing error'])
                        tqdm.write(f"OCR.space API processing error for page {page_num+1}: {error_messages[0] if error_messages else 'Unknown'}")
                    elif result.get('ParsedResults') and len(result['ParsedResults']) > 0:
                        page_text_content = result['ParsedResults'][0].get('ParsedText', '').strip()
                        if page_text_content: full_text_parts.append(page_text_content)
                        else: tqdm.write(f"OCR.space: No text found on page {page_num+1}.")
                    else: tqdm.write(f"OCR.space: No parsed results or unexpected response for page {page_num+1}. API Response: {result}")
                except requests.exceptions.HTTPError as http_err: tqdm.write(f"OCR.space HTTP error for page {page_num+1}: {http_err}. Response: {r.text[:200]}")
                except requests.exceptions.RequestException as req_err: tqdm.write(f"OCR.space request error for page {page_num+1}: {req_err}")
                except json.JSONDecodeError: tqdm.write(f"OCR.space JSON decode error for page {page_num+1}. Response: {r.text[:200]}")
                except Exception as inner_ex: tqdm.write(f"Unexpected error during OCR.space API call for page {page_num+1}: {inner_ex}")
            else: tqdm.write(f"Could not convert page {page_num+1} of {os.path.basename(pdf_path)} to image for OCR.space.")
            time.sleep(1)
        doc.close()
        if not full_text_parts:
            tqdm.write(f"No text could be extracted from '{os.path.basename(pdf_path)}' using OCR.space.")
            return None
        return "\n\n".join(full_text_parts)
    except Exception as e:
        tqdm.write(f"Error processing PDF with OCR.space for '{os.path.basename(pdf_path)}': {type(e).__name__} - {e}")
        tqdm.write(traceback.format_exc())
        return None

def orchestrate_text_extraction(pdf_path, method_id, **kwargs):
    tqdm.write(f"Extracting text from '{os.path.basename(pdf_path)}' using method: {get_description_from_code(EXTRACTION_METHOD_OPTIONS, method_id)}")
    if method_id == "pymupdf": return _extract_text_pymupdf(pdf_path)
    elif method_id == "tesseract": return _extract_text_tesseract(pdf_path, lang_code=kwargs.get('ocr_lang_tesseract', 'eng'))
    elif method_id == "ocr_space": return _extract_text_ocr_space(pdf_path, api_key=kwargs.get('ocr_space_api_key'), lang_code=kwargs.get('ocr_lang_ocrspace', 'eng'))
    else:
        tqdm.write(f"Unknown extraction method ID: {method_id}. Defaulting to PyMuPDF.")
        return _extract_text_pymupdf(pdf_path)

def save_text_content_as_pdf(text_content, pdf_path, font_name=PDF_OUTPUT_FONT_NAME):
    if not SimpleDocTemplate: tqdm.write("ReportLab not available. Cannot save text as PDF."); return False
    try:
        doc = SimpleDocTemplate(pdf_path, pagesize=letter)
        styles = getSampleStyleSheet()
        try:
            style = styles["Normal"]; style.fontName = font_name; style.fontSize = 10; style.leading = 12; style.alignment = TA_LEFT
        except Exception as style_err:
            print(f"Warning: Could not apply font '{font_name}' to PDF style. Using default. Error: {style_err}"); style = styles["Normal"]
        paragraphs_text = text_content.split('\n\n')
        story = []
        for para_text in paragraphs_text:
            if para_text.strip(): story.append(Paragraph(para_text.replace('\n', '<br/>'), style)); story.append(Spacer(1, 0.1*72))
        if not story: story.append(Paragraph("No displayable content.", style))
        doc.build(story)
        tqdm.write(f"📄 Text content saved as PDF: {pdf_path} (Font: {style.fontName})")
        return True
    except Exception as e:
        tqdm.write(f"❌ Error saving text content as PDF to '{pdf_path}': {str(e)}"); tqdm.write(traceback.format_exc()); return False

# ========== AI Text Processing Backend Functions ========== #
def _process_text_ollama(prompt_text, task_desc):
    global ollama_is_available
    if not ollama_is_available:
        return TextProcessingResult(None, error_message="Ollama server not available or not responding.")
    data = {"model": OLLAMA_MODEL, "prompt": prompt_text, "stream": False}
    try:
        tqdm.write(f"Processing with Ollama (Model: {OLLAMA_MODEL}) for {task_desc}...")
        r = requests.post(OLLAMA_API_URL, json=data, timeout=300); r.raise_for_status(); return TextProcessingResult(r.json()['response'])
    except Exception as e:
        tb_str = traceback.format_exc()
        print(f"Full error in Ollama ({task_desc}): {type(e).__name__} - {e}\nTraceback:\n{tb_str}")
        return TextProcessingResult(None, error_message=f"Ollama Error ({task_desc}): {e}")

def _process_text_openai_compatible(client, model_name, prompt_text, task_desc, service_name):
    if not client: return TextProcessingResult(None, error_message=f"{service_name} client not initialized.")
    try:
        tqdm.write(f"Processing with {service_name} (Model: {model_name}) for {task_desc}...")
        completion = client.chat.completions.create(model=model_name, messages=[{"role": "user", "content": prompt_text}])
        return TextProcessingResult(completion.choices[0].message.content)
    except Exception as e:
        tb_str = traceback.format_exc()
        print(f"Full error in {service_name} ({task_desc}): {type(e).__name__} - {e}\nTraceback:\n{tb_str}")
        return TextProcessingResult(None, error_message=f"{service_name} Error ({task_desc}): {e}")

def _process_text_google_gemini(prompt_text, task_desc):
    global google_gemini_model_client
    if not google_gemini_model_client: return TextProcessingResult(None, error_message="Google Gemini client not initialized.")
    try:
        tqdm.write(f"Processing with Google Gemini (Model: {GOOGLE_GEMINI_MODEL}) for {task_desc}...")
        response = google_gemini_model_client.generate_content(prompt_text)
        return TextProcessingResult(response.text)
    except Exception as e:
        tb_str = traceback.format_exc()
        print(f"Full error in Google Gemini ({task_desc}): {type(e).__name__} - {e}\nTraceback:\n{tb_str}")
        return TextProcessingResult(None, error_message=f"Google Gemini Error ({task_desc}): {e}")

def _process_text_lmstudio_sdk(prompt_text, task_desc):
    global lm_studio_llm_model_instance
    if not lm_studio_llm_model_instance:
        return TextProcessingResult(None, error_message=f"LM Studio SDK: Model '{LM_STUDIO_MODEL}' not successfully initialized or app not running.")

    try:
        tqdm.write(f"LM Studio SDK: Processing '{task_desc}' using model '{LM_STUDIO_MODEL}'...")
        
        messages_list = [{"role": "user", "content": prompt_text}]
        payload_for_respond = {"messages": messages_list}
        
        raw_response_object = lm_studio_llm_model_instance.respond(payload_for_respond)
        
        raw_response_string = ""

        if hasattr(raw_response_object, 'choices') and \
           isinstance(raw_response_object.choices, list) and \
           len(raw_response_object.choices) > 0 and \
           hasattr(raw_response_object.choices[0], 'message') and \
           hasattr(raw_response_object.choices[0].message, 'content'):
            raw_response_string = raw_response_object.choices[0].message.content
        elif isinstance(raw_response_object, str):
             raw_response_string = raw_response_object
        elif hasattr(raw_response_object, 'text'):
             raw_response_string = raw_response_object.text
        elif hasattr(raw_response_object, 'content'):
             raw_response_string = raw_response_object.content
        else:
            tqdm.write(f"LM Studio SDK: Unexpected response object type: {type(raw_response_object)}. Converting to string.")
            raw_response_string = str(raw_response_object)

        if not isinstance(raw_response_string, str):
             tqdm.write(f"LM Studio SDK: Could not extract string from response object. Type: {type(raw_response_string)}")
             return TextProcessingResult(None, error_message="LM Studio SDK: Failed to extract string from response.")

        cleaned_response = re.sub(r"<think>.*?</think>", "", raw_response_string, flags=re.DOTALL | re.S)
        final_response = cleaned_response.strip()
        
        if not final_response:
            return TextProcessingResult(None, error_message=f"LM Studio SDK: Empty response after processing for '{task_desc}'.")
            
        return TextProcessingResult(final_response)

    except Exception as e:
        import traceback
        tb_str = traceback.format_exc()
        error_type_name = type(e).__name__
        print(f"\nLM Studio SDK: Error during respond for '{task_desc}'. Error: {error_type_name} - {e}")
        print(f"LM Studio SDK respond traceback:\n{tb_str}")
        return TextProcessingResult(None, error_message=f"LM Studio SDK Error ({task_desc}): {e}")
    
# ========== AI Dispatcher Functions ========== #
def dispatch_ai_processing(text_content, service_key, task_type, target_language_code=None, language_options=None):
    prompt_to_send = ""
    language_name_for_prompt = "the target language"
    if task_type == "translation" and language_options and target_language_code != "none":
        full_lang_desc = get_description_from_code(language_options, target_language_code)
        language_name_for_prompt = full_lang_desc.split('(')[0].strip() or language_name_for_prompt

    if task_type == "summary":
        prompt_to_send = f"Provide a concise summary of the following text. Output only the summary:\n\n{text_content}"
    elif task_type == "explanation":
        prompt_to_send = f"Explain the following text in a clear and detailed manner, highlighting key concepts, arguments, and their relationships. Structure the explanation logically. Output only the explanation:\n\n{text_content}"
    elif task_type == "translation":
        if not target_language_code or target_language_code == "none":
            return TextProcessingResult(None, error_message="Target language for translation not specified.")
        prompt_to_send = f"Translate the following English text to {language_name_for_prompt}. Provide only the {language_name_for_prompt} translation, without any English introductory phrases, explanations, or quotation marks around the translation:\n\n{text_content}"
    else: return TextProcessingResult(None, error_message=f"Unknown task type: {task_type}")

    task_description_for_log = task_type
    if task_type == "translation": task_description_for_log = f"{task_type} to {language_name_for_prompt}"
    
    service_desc_for_log = get_description_from_code(TEXT_GENERATION_SERVICE_OPTIONS, service_key)
    tqdm.write(f"Attempting {task_description_for_log} using AI service: {service_desc_for_log}")

    if service_key == "ollama_local": return _process_text_ollama(prompt_to_send, task_description_for_log)
    elif service_key == "groq": return _process_text_openai_compatible(groq_client, GROQ_CHAT_MODEL, prompt_to_send, task_description_for_log, "Groq")
    elif service_key == "openai": return _process_text_openai_compatible(openai_client, OPENAI_CHAT_MODEL, prompt_to_send, task_description_for_log, "OpenAI (Official)")
    elif service_key == "openrouter": return _process_text_openai_compatible(openrouter_client, OPENROUTER_CHAT_MODEL, prompt_to_send, task_description_for_log, "OpenRouter")
    elif service_key == "siliconflow": return _process_text_openai_compatible(siliconflow_client, SILICONFLOW_CHAT_MODEL, prompt_to_send, task_description_for_log, "SiliconFlow")
    elif service_key == "google_gemini": return _process_text_google_gemini(prompt_to_send, task_description_for_log)
    elif service_key == "lm_studio_local": return _process_text_lmstudio_sdk(prompt_to_send, task_description_for_log)
    return TextProcessingResult(None, error_message=f"Unknown AI service key: {service_key}")

# ========== Global Definitions for Main Execution ========== #
EXTRACTION_METHOD_OPTIONS = {
    "1": ("pymupdf", "PyMuPDF (Fast, for text-based PDFs)"),
    "2": ("tesseract", "Tesseract OCR (Local, for image-based PDFs)"),
    "3": ("ocr_space", "OCR.space API (Cloud, for image-based PDFs)"),
}
TEXT_GENERATION_SERVICE_OPTIONS = {
    "1": ("ollama_local", "Local Ollama"),
    "2": ("groq", "Groq API"),
    "3": ("openai", "OpenAI API (Official)"),
    "4": ("openrouter", "OpenRouter API"),
    "5": ("siliconflow", "SiliconFlow API"),
    "6": ("google_gemini", "Google Gemini API"),
    "7": ("lm_studio_local", "LM Studio (Local SDK)")
}
TRANSLATION_LANGUAGE_OPTIONS = {
    "0": ("none", "No Translation"), "1": ("fa", "Persian (فارسی)"),
    "2": ("ar", "Arabic (العربية)"), "3": ("es", "Spanish (Español)"),
    "4": ("fr", "French (Français)"), "5": ("de", "German (Deutsch)"),
    "6": ("ru", "Russian (Русский)"), "7": ("zh", "Chinese (中文)"),
    "8": ("it", "Italian (Italiano)"),
}

# ========== Main Execution ========== #
if __name__ == "__main__":
    print("Welcome to the PDF Processing Script!")
    if not lms and "7" in TEXT_GENERATION_SERVICE_OPTIONS and TEXT_GENERATION_SERVICE_OPTIONS["7"][0] == "lm_studio_local":
        print("Warning: 'lmstudio' library is not installed. LM Studio (Local SDK) option will not work.")

    chosen_extraction_method_id = get_user_choice("Select the PDF text extraction method:", EXTRACTION_METHOD_OPTIONS)
    
    ocr_params = {}
    if chosen_extraction_method_id == "tesseract":
        if not pytesseract or not Image:
            print("\nERROR: Tesseract OCR or Pillow library not available. Please install them or choose another method.")
            print("To install: pip install pytesseract Pillow")
            print("Also ensure Tesseract OCR engine is installed on your system.")
            exit()
        ocr_params['ocr_lang_tesseract'] = input("Enter Tesseract language code(s) (e.g., eng, fas, eng+fas, rus): ").strip() or "eng"
        print(f"Tesseract will use language(s): {ocr_params['ocr_lang_tesseract']}")
    elif chosen_extraction_method_id == "ocr_space":
        current_ocr_space_key_env = os.getenv("OCR_SPACE_API_KEY")
        if not current_ocr_space_key_env:
            current_ocr_space_key_input = input("Enter your OCR.space API key (or set OCR_SPACE_API_KEY in .env file): ").strip()
            if not current_ocr_space_key_input:
                print("\nERROR: OCR.space API key not provided. This method cannot be used. Exiting.")
                exit()
            ocr_params['ocr_space_api_key'] = current_ocr_space_key_input
        else:
            ocr_params['ocr_space_api_key'] = current_ocr_space_key_env
            print(f"Using OCR.space API key from environment variables.")
        ocr_params['ocr_lang_ocrspace'] = input("Enter OCR.space language code (e.g., eng, per, rus, chs, ita): ").strip() or "eng"
        print(f"OCR.space will use language: {ocr_params['ocr_lang_ocrspace']}")

    chosen_summary_service_key = get_user_choice("Select the service for Summarization:", TEXT_GENERATION_SERVICE_OPTIONS)
    chosen_explanation_service_key = get_user_choice("Select the service for Explanation:", TEXT_GENERATION_SERVICE_OPTIONS)
    chosen_translation_language_code = get_user_choice("Select the target language for Translation (0 for none):", TRANSLATION_LANGUAGE_OPTIONS)
    chosen_translation_service_key = None
    if chosen_translation_language_code != "none":
        lang_desc_for_prompt = get_description_from_code(TRANSLATION_LANGUAGE_OPTIONS, chosen_translation_language_code)
        chosen_translation_service_key = get_user_choice(f"Select the AI service for Translation to {lang_desc_for_prompt}:", TEXT_GENERATION_SERVICE_OPTIONS)

    services_to_init_ai = set([chosen_summary_service_key, chosen_explanation_service_key])
    if chosen_translation_service_key: services_to_init_ai.add(chosen_translation_service_key)
    initialize_ai_clients(services_to_init_ai)
    
    pdf_files = [f for f in os.listdir(INPUT_FOLDER_PDF) if f.lower().endswith(".pdf")]

    if not pdf_files: print(f"❌ No PDF files found in: {INPUT_FOLDER_PDF}")
    else:
        extraction_method_desc_print = get_description_from_code(EXTRACTION_METHOD_OPTIONS, chosen_extraction_method_id)
        print(f"\n🔍 Found {len(pdf_files)} PDF(s). Using Text Extraction: {extraction_method_desc_print}")
        if 'ocr_lang_tesseract' in ocr_params: print(f"   Tesseract Language: {ocr_params['ocr_lang_tesseract']}")
        if 'ocr_lang_ocrspace' in ocr_params: print(f"   OCR.space Language: {ocr_params['ocr_lang_ocrspace']}")
        if chosen_summary_service_key == "lm_studio_local" or chosen_explanation_service_key == "lm_studio_local" or chosen_translation_service_key == "lm_studio_local":
            if LM_STUDIO_MODEL: print(f"   LM Studio Model ID: {LM_STUDIO_MODEL}")
            else: print("   Warning: LM_STUDIO_MODEL (repository ID) not set for LM Studio SDK.")


        summary_service_desc_print = get_description_from_code(TEXT_GENERATION_SERVICE_OPTIONS, chosen_summary_service_key)
        explanation_service_desc_print = get_description_from_code(TEXT_GENERATION_SERVICE_OPTIONS, chosen_explanation_service_key)
        translation_lang_desc_print = get_description_from_code(TRANSLATION_LANGUAGE_OPTIONS, chosen_translation_language_code)
        
        print(f"📄 Summarization AI Service: {summary_service_desc_print}")
        print(f"💡 Explanation AI Service: {explanation_service_desc_print}")
        print(f"🈯 Translation Language: {translation_lang_desc_print}")
        if chosen_translation_service_key:
            translation_service_desc_print = get_description_from_code(TEXT_GENERATION_SERVICE_OPTIONS, chosen_translation_service_key)
            print(f"🌍 Translation AI Service: {translation_service_desc_print}\n")
        else:
            print(f"🌍 Translation AI Service: Not selected\n")

        for pdf_file_name in tqdm(pdf_files, desc="Total PDF Processing Progress"):
            base_name = os.path.splitext(pdf_file_name)[0]
            pdf_file_path = os.path.join(INPUT_FOLDER_PDF, pdf_file_name)
            tqdm.write(f"\n{'='*10} Processing: {pdf_file_name} {'='*10}")

            summary_en_txt_path = os.path.join(SUMMARIES_FOLDER, f"{base_name}_summary_en.txt")
            summary_en_pdf_path = os.path.join(SUMMARIES_FOLDER, f"{base_name}_summary_en.pdf")
            summary_translated_txt_path = None
            if chosen_translation_language_code != "none":
                summary_translated_txt_path = os.path.join(SUMMARIES_FOLDER, f"{base_name}_summary_{chosen_translation_language_code}.txt")

            explanation_en_txt_path = os.path.join(EXPLANATIONS_FOLDER, f"{base_name}_explanation_en.txt")
            explanation_en_pdf_path = os.path.join(EXPLANATIONS_FOLDER, f"{base_name}_explanation_en.pdf")
            explanation_translated_txt_path = None
            if chosen_translation_language_code != "none":
                explanation_translated_txt_path = os.path.join(EXPLANATIONS_FOLDER, f"{base_name}_explanation_{chosen_translation_language_code}.txt")

            extracted_text_result = orchestrate_text_extraction(pdf_file_path, chosen_extraction_method_id, **ocr_params)
            
            if not extracted_text_result or not extracted_text_result.strip():
                tqdm.write(f"❌ Failed to extract text from '{pdf_file_name}' or PDF is empty/unreadable with chosen method. Skipping.")
                continue
            tqdm.write(f"✅ Text extracted successfully ({len(extracted_text_result)} chars) using {get_description_from_code(EXTRACTION_METHOD_OPTIONS, chosen_extraction_method_id)}.")

            # Process Summaries
            tqdm.write("\n--- Processing Summary ---")
            english_summary = None
            if os.path.exists(summary_en_txt_path):
                tqdm.write(f"📖 English summary TXT exists. Loading: {summary_en_txt_path}")
                try:
                    with open(summary_en_txt_path, "r", encoding="utf-8") as f: english_summary = f.read()
                except Exception as e: tqdm.write(f"❌ Error loading English summary: {e}")
            else:
                ai_result = dispatch_ai_processing(extracted_text_result, chosen_summary_service_key, "summary")
                if ai_result.status == "completed" and ai_result.text:
                    english_summary = ai_result.text.strip()
                    with open(summary_en_txt_path, "w", encoding="utf-8") as f: f.write(english_summary)
                    tqdm.write(f"💾 English summary saved to TXT: {summary_en_txt_path}")
                else: tqdm.write(f"❌ English summary generation failed: {ai_result.error_message if ai_result else 'Unknown error'}")

            if english_summary:
                if not os.path.exists(summary_en_pdf_path) and SimpleDocTemplate:
                    save_text_content_as_pdf(english_summary, summary_en_pdf_path, PDF_OUTPUT_FONT_NAME)
                elif os.path.exists(summary_en_pdf_path):
                     tqdm.write(f"📄 English summary PDF already exists: {summary_en_pdf_path}")
                if chosen_translation_language_code != "none" and chosen_translation_service_key and summary_translated_txt_path:
                    if os.path.exists(summary_translated_txt_path):
                        tqdm.write(f"🈯 Translated ({chosen_translation_language_code}) summary TXT exists. Skipping translation.")
                    else:
                        ai_result_translated = dispatch_ai_processing(english_summary, chosen_translation_service_key, "translation", target_language_code=chosen_translation_language_code, language_options=TRANSLATION_LANGUAGE_OPTIONS)
                        if ai_result_translated.status == "completed" and ai_result_translated.text:
                            with open(summary_translated_txt_path, "w", encoding="utf-8") as f: f.write(ai_result_translated.text.strip())
                            tqdm.write(f"💾 Translated ({chosen_translation_language_code}) summary saved to TXT.")
                        else: tqdm.write(f"❌ Translated ({chosen_translation_language_code}) summary translation failed: {ai_result_translated.error_message if ai_result_translated else 'Unknown error'}")
            else: tqdm.write("ℹ️ No English summary generated to process further.")

            # Process Explanations
            tqdm.write("\n--- Processing Explanation ---")
            english_explanation = None
            if os.path.exists(explanation_en_txt_path):
                tqdm.write(f"💡 English explanation TXT exists. Loading: {explanation_en_txt_path}")
                try:
                    with open(explanation_en_txt_path, "r", encoding="utf-8") as f: english_explanation = f.read()
                except Exception as e: tqdm.write(f"❌ Error loading English explanation: {e}")
            else:
                ai_result = dispatch_ai_processing(extracted_text_result, chosen_explanation_service_key, "explanation")
                if ai_result.status == "completed" and ai_result.text:
                    english_explanation = ai_result.text.strip()
                    with open(explanation_en_txt_path, "w", encoding="utf-8") as f: f.write(english_explanation)
                    tqdm.write(f"💾 English explanation saved to TXT: {explanation_en_txt_path}")
                else: tqdm.write(f"❌ English explanation generation failed: {ai_result.error_message if ai_result else 'Unknown error'}")
            
            if english_explanation:
                if not os.path.exists(explanation_en_pdf_path) and SimpleDocTemplate:
                    save_text_content_as_pdf(english_explanation, explanation_en_pdf_path, PDF_OUTPUT_FONT_NAME)
                elif os.path.exists(explanation_en_pdf_path):
                    tqdm.write(f"📄 English explanation PDF already exists: {explanation_en_pdf_path}")
                if chosen_translation_language_code != "none" and chosen_translation_service_key and explanation_translated_txt_path:
                    if os.path.exists(explanation_translated_txt_path):
                        tqdm.write(f"🈯 Translated ({chosen_translation_language_code}) explanation TXT exists. Skipping.")
                    else:
                        ai_result_translated = dispatch_ai_processing(english_explanation, chosen_translation_service_key, "translation", target_language_code=chosen_translation_language_code, language_options=TRANSLATION_LANGUAGE_OPTIONS)
                        if ai_result_translated.status == "completed" and ai_result_translated.text:
                            with open(explanation_translated_txt_path, "w", encoding="utf-8") as f: f.write(ai_result_translated.text.strip())
                            tqdm.write(f"💾 Translated ({chosen_translation_language_code}) explanation saved to TXT.")
                        else: tqdm.write(f"❌ Translated ({chosen_translation_language_code}) explanation translation failed: {ai_result_translated.error_message if ai_result_translated else 'Unknown error'}")
            else: tqdm.write("ℹ️ No English explanation generated to process further.")
            
            time.sleep(0.2)

        print("\n✅ All PDF processing finished.")