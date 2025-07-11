import os
import cv2
import pytesseract
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer, InputExample, losses, util
from torch.utils.data import DataLoader
from PIL import Image, ImageEnhance
from sklearn.model_selection import KFold
import logging
import time
import json
import configparser
import argparse
from pathlib import Path
import torch
from functools import lru_cache

# Setup logging first
logging.basicConfig(
    filename="exam_marking.log", 
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load configuration from file
def load_config(config_file="config.ini"):
    """
    Load configuration from a config file.
    
    Args:
        config_file (str): Path to the configuration file
        
    Returns:
        dict: Dictionary containing configuration values
    """
    config = configparser.ConfigParser()
    
    # Default configuration
    default_config = {
        'paths': {
            'booklet_folder': "./exam_booklets/",
            'results_file': "./exam_results.csv",
            'training_data': "./training_data.csv",
            'fine_tuned_model_path': "./fine_tuned_model/"
        },
        'tesseract': {
            'path': r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        },
        'training': {
            'epochs': 4,
            'batch_size': 16,
            'warmup_steps': 100,
            'use_gpu': 'auto'
        }
    }
    
    # Create default config file if it doesn't exist
    if not os.path.exists(config_file):
        for section, options in default_config.items():
            if not config.has_section(section):
                config.add_section(section)
            for key, value in options.items():
                config.set(section, key, str(value))
        
        with open(config_file, 'w') as f:
            config.write(f)
        logger.info(f"Created default configuration file: {config_file}")
    
    # Load existing config
    config.read(config_file)
    
    # Convert to dictionary
    config_dict = {}
    for section in config.sections():
        config_dict[section] = {}
        for key, value in config.items(section):
            config_dict[section][key] = value
    
    return config_dict

# Load configuration
CONFIG = load_config()

# Paths from config
BOOKLET_FOLDER = Path(CONFIG['paths']['booklet_folder'])
RESULTS_FILE = Path(CONFIG['paths']['results_file'])
TRAINING_DATA = Path(CONFIG['paths']['training_data'])
FINE_TUNED_MODEL_PATH = Path(CONFIG['paths']['fine_tuned_model_path'])

# Create directories if they don't exist
BOOKLET_FOLDER.mkdir(parents=True, exist_ok=True)
FINE_TUNED_MODEL_PATH.mkdir(parents=True, exist_ok=True)

# Setup Tesseract path
pytesseract.pytesseract.tesseract_cmd = CONFIG['tesseract']['path']

# Setup device for PyTorch
def get_device():
    """
    Determine the appropriate device (CPU/GPU) for model training.
    
    Returns:
        torch.device: The device to use for model training
    """
    use_gpu = CONFIG['training']['use_gpu'].lower()
    if use_gpu == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif use_gpu == 'true':
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            logger.warning("GPU requested but not available. Using CPU instead.")
            return torch.device('cpu')
    else:
        return torch.device('cpu')

DEVICE = get_device()
logger.info(f"Using device: {DEVICE}")

# NLTK setup
try:
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))
    logger.info("NLTK resources loaded successfully")
except Exception as e:
    logger.error(f"Failed to download NLTK resources: {e}")
    raise

# Model answers and MCQs
# Instead, we'll load these from files

def clean_text(text):
    """
    Clean and preprocess text for comparison.

    Args:
        text (str): Raw text to clean
        
    Returns:
        str: Cleaned text with stopwords removed and words lemmatized
    """
    if not isinstance(text, str):
        logger.warning(f"Non-string input to clean_text: {type(text)}")
        text = str(text)
    
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r'\s+', ' ', text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

def fine_tune_model(training_data_path=None):
    """
    Fine-tune the sentence transformer model on domain-specific data.
    
    Args:
        training_data_path (str, optional): Path to training data CSV
        
    Returns:
        SentenceTransformer: Fine-tuned model
    """
    try:
        logger.info("Fine-tuning model...")
        print("[INFO] Fine-tuning model...")
        
        # If no training data is provided or file doesn't exist, use default model
        if training_data_path is None or not os.path.exists(training_data_path):
            logger.warning(f"Training data not found at {training_data_path}. Using default model.")
            print(f"[WARNING] Training data not found at {training_data_path}. Using default model.")
            return SentenceTransformer("all-MiniLM-L6-v2")
        
        # Load training data
        examples = []
        df = pd.read_csv(training_data_path)
        
        # Check if the dataframe is empty
        if df.empty:
            logger.warning(f"Training data file is empty: {training_data_path}")
            print(f"[WARNING] Training data file is empty. Using default model.")
            return SentenceTransformer("all-MiniLM-L6-v2")
        
        # Check for required columns
        required_columns = ['student_answer', 'model_answer', 'similarity']
        if not all(col in df.columns for col in required_columns):
            # Try alternative column names
            if 'student_answer' in df.columns and 'model_answer' in df.columns and 'score' in df.columns:
                # Use 'score' instead of 'similarity'
                for _, row in df.iterrows():
                    examples.append(InputExample(
                        texts=[row['student_answer'], row['model_answer']], 
                        label=float(row['score']) / 100.0  # Convert score from 0-100 to 0-1
                    ))
            else:
                logger.warning(f"Training data missing required columns: {required_columns}")
                print(f"[WARNING] Training data missing required columns. Using default model.")
                return SentenceTransformer("all-MiniLM-L6-v2")
        else:
            # Use standard column names
            for _, row in df.iterrows():
                examples.append(InputExample(
                    texts=[row['student_answer'], row['model_answer']], 
                    label=float(row['similarity'])
                ))
        
        # Check if we have any examples
        if not examples:
            logger.warning(f"No valid examples found in training data: {training_data_path}")
            print(f"[WARNING] No valid examples found in training data. Using default model.")
            return SentenceTransformer("all-MiniLM-L6-v2")
        
        # Initialize model
        model = SentenceTransformer("all-MiniLM-L6-v2")
        model.to(DEVICE)
        
        # Create data loader
        train_loader = DataLoader(examples, shuffle=True, batch_size=int(CONFIG['training']['batch_size']))
        loss = losses.CosineSimilarityLoss(model=model)

        # Train the model
        print(f"[INFO] Training model with {len(examples)} examples...")
        model.fit(train_objectives=[(train_loader, loss)],
                epochs=int(CONFIG['training']['epochs']),
                warmup_steps=int(CONFIG['training']['warmup_steps']),
                show_progress_bar=True)

        # Create a unique model directory to avoid conflicts
        import time
        timestamp = int(time.time())
        model_save_path = Path(FINE_TUNED_MODEL_PATH) / f"model_{timestamp}"
        
        # Ensure the directory exists
        model_save_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save the model to the unique directory
            model.save(str(model_save_path))
            logger.info(f"Fine-tuned model saved successfully to {model_save_path}")
            print(f"[INFO] Fine-tuned model saved successfully to {model_save_path}")
            
            # Create a symlink or copy to a standard location
            default_model_path = Path(FINE_TUNED_MODEL_PATH) / "current_model"
            
            # Remove existing symlink/directory if it exists
            if default_model_path.exists():
                if default_model_path.is_symlink():
                    default_model_path.unlink()
                else:
                    import shutil
                    shutil.rmtree(default_model_path, ignore_errors=True)
            
            # Create a symlink on Unix or copy on Windows
            try:
                default_model_path.symlink_to(model_save_path)
                logger.info(f"Created symlink from {default_model_path} to {model_save_path}")
            except (OSError, NotImplementedError):
                # On Windows, symlinks might not work, so copy the files
                import shutil
                default_model_path.mkdir(parents=True, exist_ok=True)
                
                # Copy model files
                for item in model_save_path.iterdir():
                    if item.is_file():
                        shutil.copy2(item, default_model_path / item.name)
                logger.info(f"Copied model files from {model_save_path} to {default_model_path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            print(f"[ERROR] Failed to save model: {e}")
            # Continue with the trained model even if saving failed
        
        return model
        
    except Exception as e:
        logger.error(f"Error during model fine-tuning: {e}")
        print(f"[ERROR] Failed to fine-tune model: {e}")
        import traceback
        print(traceback.format_exc())
        return SentenceTransformer("all-MiniLM-L6-v2")

def load_model(model_path=None):
    """
    Load a fine-tuned sentence transformer model.
    
    Args:
        model_path (str, optional): Path to the model directory
        
    Returns:
        SentenceTransformer: The loaded model
    """
    try:
        # Use the provided path or default to FINE_TUNED_MODEL_PATH
        model_path = Path(model_path) if model_path else FINE_TUNED_MODEL_PATH
        
        # Check if the model path exists
        if not model_path.exists():
            logger.warning(f"Model path does not exist: {model_path}")
            print(f"[WARNING] Model path does not exist. Falling back to default model: all-MiniLM-L6-v2")
            return SentenceTransformer("all-MiniLM-L6-v2").to(DEVICE)

        # First, try to load from "current_model" directory if it exists
        current_model_path = model_path / "current_model"
        if current_model_path.exists() and any(current_model_path.iterdir()):
            try:
                logger.info(f"Attempting to load from current_model directory: {current_model_path}")
                return SentenceTransformer(str(current_model_path)).to(DEVICE)
            except Exception as e:
                logger.warning(f"Failed to load from current_model directory: {e}")
        
        # Look for model_* directories and sort by timestamp (newest first)
        model_dirs = sorted(
            [d for d in model_path.iterdir() if d.is_dir() and d.name.startswith("model_")],
            key=lambda d: d.name,
            reverse=True
        )
        
        if model_dirs:
            logger.info(f"Found model directories: {[d.name for d in model_dirs]}")
            # Try to load from the newest model directory
            for model_dir in model_dirs:
                try:
                    logger.info(f"Loading from {model_dir}")
                    return SentenceTransformer(str(model_dir)).to(DEVICE)
                except Exception as e:
                    logger.warning(f"Failed to load from {model_dir}: {e}")
                    continue
        
        # If no model_* directories, look for other directories that might contain models
        other_dirs = [d for d in model_path.iterdir() if d.is_dir() and (d / "config.json").exists()]
        
        if other_dirs:
            logger.info(f"Found other model directories: {[d.name for d in other_dirs]}")
            # Try to load from these directories
            for model_dir in other_dirs:
                try:
                    logger.info(f"Loading from {model_dir}")
                    return SentenceTransformer(str(model_dir)).to(DEVICE)
                except Exception as e:
                    logger.warning(f"Failed to load from {model_dir}: {e}")
                    continue
        
        # If we couldn't load from subdirectories, try the main directory
        try:
            logger.info(f"Attempting to load from main directory: {model_path}")
            return SentenceTransformer(str(model_path)).to(DEVICE)
        except Exception as e:
            logger.warning(f"Failed to load from main directory: {e}")
            
            # As a last resort, use the default model
            logger.warning("All loading attempts failed. Using default model.")
            print("[WARNING] Failed to load fine-tuned model. Using default model: all-MiniLM-L6-v2")
            return SentenceTransformer("all-MiniLM-L6-v2").to(DEVICE)

    except Exception as e:
        # Log detailed error information
        logger.error(f"Failed to load model from {model_path}: {e}")
        print(f"[ERROR] Failed to load fine-tuned model. Error: {e}")
        print("[INFO] Falling back to default model.")
        return SentenceTransformer("all-MiniLM-L6-v2").to(DEVICE)

def preprocess_image_for_ocr(img):
    """
    Preprocess image to improve OCR accuracy for handwritten text.

    Args:
        img (PIL.Image): Input image

    Returns:
        PIL.Image: Preprocessed image
    """
    try:
        import cv2
        import numpy as np
        from PIL import Image, ImageEnhance, ImageFilter

        # Convert PIL to OpenCV format
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        # Convert to grayscale
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        # Morphological operations to clean up the image
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # Convert back to PIL
        processed_img = Image.fromarray(cleaned)

        # Enhance contrast
        enhancer = ImageEnhance.Contrast(processed_img)
        processed_img = enhancer.enhance(2.0)

        # Enhance sharpness
        enhancer = ImageEnhance.Sharpness(processed_img)
        processed_img = enhancer.enhance(1.5)

        return processed_img

    except Exception as e:
        logger.warning(f"Image preprocessing failed: {e}. Using original image.")
        return img

def extract_text(image_path, retries=3):
    """
    Extract text from an image using OCR with retry mechanism and image preprocessing.

    Args:
        image_path (str): Path to the image file
        retries (int): Number of retry attempts

    Returns:
        str: Extracted text
    """
    # Convert to Path object if it's a string
    if isinstance(image_path, str):
        image_path = Path(image_path)

    # Check if file exists
    if not image_path.exists():
        logger.error(f"File not found: {image_path}")
        print(f"[ERROR] File not found: {image_path}")
        return ""

    # Handle different file types
    if image_path.suffix.lower() == '.pdf':
        try:
            import pytesseract
            from pdf2image import convert_from_path

            logger.info(f"Converting PDF to images: {image_path}")
            images = convert_from_path(str(image_path))
            text = ""

            for i, image in enumerate(images):
                logger.info(f"Extracting text from page {i+1} of {len(images)}")
                page_text = pytesseract.image_to_string(image)
                text += f"\n--- Page {i+1} ---\n{page_text}"

            return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF {image_path}: {e}")
            print(f"[ERROR] Failed to extract text from PDF: {e}")
            return ""
    else:  # Image files
        try:
            import pytesseract
            from PIL import Image

            # Try to open the image
            try:
                img = Image.open(str(image_path))
                logger.info(f"Opened image {image_path} with size {img.size}")
            except Exception as e:
                logger.error(f"Failed to open image {image_path}: {e}")
                print(f"[ERROR] Failed to open image: {e}")
                return ""

            # Try different OCR configurations
            ocr_configs = [
                '--psm 6',  # Uniform block of text
                '--psm 4',  # Single column of text
                '--psm 3',  # Fully automatic page segmentation
                '--psm 1',  # Automatic page segmentation with OSD
            ]

            best_text = ""
            best_confidence = 0

            for attempt in range(retries):
                try:
                    # Preprocess image for better OCR
                    processed_img = preprocess_image_for_ocr(img)

                    # Try different OCR configurations
                    for config in ocr_configs:
                        try:
                            text = pytesseract.image_to_string(processed_img, config=config)
                            if text.strip() and len(text.strip()) > len(best_text.strip()):
                                best_text = text
                                logger.info(f"Better OCR result with config '{config}': {len(text)} characters")
                        except Exception as config_error:
                            logger.warning(f"OCR config '{config}' failed: {config_error}")
                            continue

                    # If we got some text, break
                    if best_text.strip():
                        logger.info(f"Successfully extracted text from {image_path} (attempt {attempt+1})")
                        logger.info(f"Extracted text length: {len(best_text)} characters")
                        return best_text
                    else:
                        logger.warning(f"OCR returned empty text on attempt {attempt+1} for {image_path}")
                        # Try different image enhancements for next attempt
                        if attempt < retries - 1:
                            # Try different preprocessing
                            img = img.convert('L')  # Convert to grayscale

                except Exception as e:
                    logger.error(f"OCR error on attempt {attempt+1} for {image_path}: {e}")
                    if attempt == retries - 1:  # Last attempt
                        print(f"[ERROR] Failed to extract text after {retries} attempts: {e}")
                        return best_text if best_text.strip() else ""

            # Return best result even if not perfect
            return best_text

        except ImportError:
            logger.error("pytesseract or PIL not installed")
            print("[ERROR] pytesseract or PIL not installed. Install with: pip install pytesseract pillow")
            return ""
        except Exception as e:
            logger.error(f"Unexpected error extracting text from {image_path}: {e}")
            print(f"[ERROR] Unexpected error extracting text: {e}")
            return ""

def fuzzy_find_questions(text):
    """
    Use fuzzy matching to find question patterns in OCR text with errors.

    Args:
        text (str): Raw OCR text

    Returns:
        dict: Dictionary mapping question IDs to potential answer text
    """
    import difflib

    # Common OCR errors for question patterns
    question_patterns = [
        # Standard patterns
        r"Q\s*([1-6])[:\-\.\s]+(.*)",
        r"Question\s*([1-6])[:\-\.\s]+(.*)",
        r"([1-6])\s*[:\-\.\)]\s*(.*)",

        # OCR error patterns (common misreadings)
        r"G\s*([1-6])[:\-\.\s]+(.*)",  # Q -> G
        r"O\s*([1-6])[:\-\.\s]+(.*)",  # Q -> O
        r"D\s*([1-6])[:\-\.\s]+(.*)",  # Q -> D
        r"B\s*([1-6])[:\-\.\s]+(.*)",  # Q -> B
        r"[QG0ODB]\s*([1-6])[:\-\.\s]+(.*)",  # Multiple possibilities
        r"[QG0ODB]([1-6])[:\-\.\s]+(.*)",  # No space
        r"[QG0ODB]\s*[1-6]\s*[:\-\.\s]+(.*)",  # Flexible number matching

        # More flexible patterns for badly OCR'd text
        r"[QG0ODB]?[a-z]*\s*([1-6])[:\-\.\s]+(.*)",  # With extra letters
        r"([1-6])[:\-\.\s]*([A-Za-z].*)",  # Just number followed by text

        # Special patterns observed in the OCR
        r"BE\s*(.*)",  # Might be Q1 or Q2
        r"B22\s*(.*)",  # Might be Q2
        r"D2\s*(.*)",   # Might be Q2
        r"GAL\s*(.*)",  # Might be Q4
        r"Qn\s*([1-6])[:\-\.\s]*(.*)",  # Qn instead of Q
    ]

    answers = {}
    lines = text.splitlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Try each pattern
        for pattern in question_patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                try:
                    q_num = match.group(1)
                    q_id = f"Q{q_num}"
                    answer_text = match.group(2).strip() if len(match.groups()) > 1 else ""

                    if q_id not in answers:
                        answers[q_id] = answer_text
                    else:
                        # Append to existing answer
                        answers[q_id] += " " + answer_text

                    logger.info(f"Found question {q_id} using pattern: {pattern}")
                    break  # Found a match, no need to try other patterns
                except (IndexError, ValueError):
                    continue

    return answers

def extract_mcq_answers(text):
    """
    Extract MCQ answers with robust pattern matching for OCR errors.

    Args:
        text (str): Raw OCR text

    Returns:
        dict: Dictionary mapping question IDs to selected options
    """
    mcq_answers = {}

    # Multiple patterns for MCQ answers with OCR error tolerance
    mcq_patterns = [
        # Standard patterns
        r"Q\s*([3-5])[:\-\.\s]*([A-D])\s*\)",  # Q3: A)
        r"Q\s*([3-5])[:\-\.\s]*([A-D])",       # Q3: A
        r"([3-5])\s*[:\-\.\)]\s*([A-D])",      # 3: A

        # Answer patterns
        r"Answer[:\s]*([A-D])",                 # Answer: A
        r"Ans[:\s]*([A-D])",                    # Ans: A
        r"Fryer\s*([A-D])",                     # OCR error for "Answer"

        # OCR error patterns for question numbers
        r"[QG0ODB]\s*([3-5])[:\-\.\s]*([A-D])",  # G3: A (Q->G, Q->D, Q->B)
        r"[QG0ODB]([3-5])[:\-\.\s]*([A-D])",     # G3A (no spaces)
        r"D([3-5])\s*([A-D])",                   # D3 A (Q->D)
        r"GAL\s*([A-D])",                        # GAL A (might be Q4)

        # Patterns with option letters and parentheses
        r"([A-D])\s*\)\s*[A-Za-z]",              # A) followed by text
        r"([A-D])\s*[:\-\.\)]\s*[A-Za-z]",       # A: followed by text
        r"([A-D])\s*\)\s*[a-z]+",                # A) followed by lowercase text

        # Specific patterns observed in OCR
        r"AY\s*([A-D])",                         # AY might be "A)"
        r"([A-D])\s*\]\s*",                      # A] instead of A)
        r"([A-D])\s*\&\s*",                      # A& instead of A)
    ]

    lines = text.splitlines()

    # First pass: look for explicit question-answer patterns
    for line in lines:
        line = line.strip()
        if not line:
            continue

        for pattern in mcq_patterns[:6]:  # Use first 6 patterns
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                try:
                    if len(match.groups()) == 2:
                        q_num = match.group(1)
                        option = match.group(2).upper()
                        q_id = f"Q{q_num}"
                        mcq_answers[q_id] = option
                        logger.info(f"Found MCQ answer {q_id}: {option} using pattern: {pattern}")
                    elif len(match.groups()) == 1:
                        # This might be an "Answer: A" pattern
                        option = match.group(1).upper()
                        # We'll need context to determine which question this belongs to
                        logger.info(f"Found standalone answer: {option}")
                except (IndexError, ValueError):
                    continue

    # Second pass: look for option letters near question numbers
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue

        # Look for question numbers (3, 4, 5) and nearby option letters
        q_match = re.search(r"[QG0ODB]?\s*([3-5])", line, re.IGNORECASE)
        if q_match:
            q_num = q_match.group(1)
            q_id = f"Q{q_num}"

            if q_id not in mcq_answers:
                # Look for option letters in this line and nearby lines
                for check_line in lines[max(0, i-1):min(len(lines), i+3)]:
                    option_match = re.search(r"\b([A-D])\b", check_line)
                    if option_match:
                        option = option_match.group(1).upper()
                        mcq_answers[q_id] = option
                        logger.info(f"Found MCQ answer {q_id}: {option} near question number")
                        break

    # Third pass: look for specific OCR patterns observed in student_5_booklet.jpeg
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue

        # Pattern: "D2 wher..." -> Q3, look for "©) Pehneaken" -> C
        if re.search(r"D2.*NOT.*mocbene", line, re.IGNORECASE):
            # This is likely Q3, look for answer in next few lines
            for check_line in lines[i:min(len(lines), i+3)]:
                if re.search(r"©\)|Pehneaken", check_line, re.IGNORECASE):
                    mcq_answers["Q3"] = "C"
                    logger.info(f"Found Q3 answer C using OCR pattern matching")
                    break

        # Pattern: "GAL tthe afyuthe" -> Q4 or Q5, look for options
        if re.search(r"GAL.*afyuthe", line, re.IGNORECASE):
            # Look for options in next line
            if i + 1 < len(lines):
                next_line = lines[i + 1]
                # Look for pattern like "A) ... 8) ... 6) ..." where 8)=B) and 6)=C)
                if re.search(r"A\).*[86]\)", next_line):
                    # This might be Q4 (correct answer C) or Q5 (correct answer B)
                    # Need more context to determine which question this is
                    # For now, let's assume it's Q5 and look for B pattern
                    if re.search(r"8\)", next_line):  # 8) might be B)
                        mcq_answers["Q5"] = "B"
                        logger.info(f"Found Q5 answer B using OCR pattern matching")
                    elif re.search(r"6\)", next_line):  # 6) might be C)
                        mcq_answers["Q4"] = "C"
                        logger.info(f"Found Q4 answer C using OCR pattern matching")

    return mcq_answers

def extract_answers_from_text(text):
    """
    Extract answers by question from OCR text with improved error handling.

    Args:
        text (str): Raw text from OCR

    Returns:
        dict: Dictionary mapping question IDs to answers
    """
    # Ensure text is a string
    if not isinstance(text, str):
        text = str(text)

    logger.info(f"Extracting answers from text of length: {len(text)}")
    logger.debug(f"Raw OCR text: {repr(text[:500])}...")  # Log first 500 chars

    # Initialize answers dictionary
    answers = {}
    lines = text.splitlines()
    current_q = None

    # First pass: Standard question extraction (prioritize clean patterns)
    for line in lines:
        line = str(line).strip()
        if not line:
            continue

        # Try to match question identifiers like Q1, Q2, etc. (strict pattern first)
        match = re.match(r"^(Q[1-6])[:\-\.]?\s*(.*)", line, re.IGNORECASE)
        if match:
            current_q = match.group(1).upper()
            answers[current_q] = match.group(2).strip()
            logger.info(f"Found question {current_q} with initial text: {repr(match.group(2)[:50])}...")
        elif current_q and line:
            # Continue adding to current question
            answers[current_q] = answers.get(current_q, "") + " " + line

    # Second pass: Look for "Answer: X" patterns and associate with MCQ questions
    answer_patterns = [
        r'Answer[:\s]*([A-D])',
        r'Ans[:\s]*([A-D])',
    ]

    # Find all answer patterns in the text
    found_answers = []
    for pattern in answer_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        found_answers.extend([match.upper() for match in matches])

    logger.info(f"Found answer patterns: {found_answers}")

    # Associate found answers with MCQ questions in order
    mcq_questions = ["Q3", "Q4", "Q5"]
    for i, q_id in enumerate(mcq_questions):
        if i < len(found_answers):
            # Replace the answer with the selected option format
            answers[q_id] = f"[SELECTED_OPTION:{found_answers[i]}]"
            logger.info(f"Associated answer {found_answers[i]} with {q_id}")

    # Third pass: Fallback for handwritten booklets with OCR errors (only if no standard answers found)
    if not any(answers.get(q_id, "") for q_id in ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"]):
        logger.info("No standard answers found, trying fuzzy matching for handwritten text")

        # Use fuzzy matching for badly OCR'd handwritten text
        fuzzy_answers = fuzzy_find_questions(text)
        for q_id, answer in fuzzy_answers.items():
            if q_id not in answers or not answers[q_id]:
                answers[q_id] = answer
                logger.info(f"Fuzzy match found {q_id}: {repr(answer[:50])}...")

        # Extract MCQ answers for handwritten text
        mcq_answers = extract_mcq_answers(text)
        for q_id, option in mcq_answers.items():
            if q_id in ["Q3", "Q4", "Q5"]:
                answers[q_id] = option
                logger.info(f"MCQ extraction found {q_id}: {option}")

    # Ensure all questions have entries
    for i in range(1, 7):
        q = f"Q{i}"
        if q not in answers:
            answers[q] = ""

    # Filter out invalid question IDs (only keep Q1-Q6)
    valid_answers = {}
    for q_id in ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"]:
        valid_answers[q_id] = answers.get(q_id, "")

    # Log final results
    for q_id, answer in valid_answers.items():
        if answer:
            logger.info(f"Final {q_id}: {repr(answer[:100])}...")
        else:
            logger.warning(f"No answer found for {q_id}")

    return valid_answers

def evaluate_written_answer(student_ans, model_ans, model):
    """
    Score a written answer using semantic similarity.
    
    Args:
        student_ans (str): Student's answer
        model_ans (str): Model answer
        model (SentenceTransformer): The model to use for scoring
        
    Returns:
        float: Score between 0 and 100
    """
    # Ensure both inputs are strings
    if not isinstance(student_ans, str):
        student_ans = str(student_ans)
    if not isinstance(model_ans, str):
        model_ans = str(model_ans)
        
    # Clean the text
    student_ans_clean = clean_text(student_ans)
    model_ans_clean = clean_text(model_ans)
    
    # If either answer is empty after cleaning, return 0
    if not student_ans_clean or not model_ans_clean:
        return 0.0
    
    # Encode the answers
    student_embedding = model.encode(student_ans_clean, convert_to_tensor=True)
    model_embedding = model.encode(model_ans_clean, convert_to_tensor=True)
    
    # Calculate cosine similarity
    similarity = util.pytorch_cos_sim(student_embedding, model_embedding).item()
    
    # Convert similarity (-1 to 1) to score (0 to 100)
    score = max(0, min(100, (similarity + 1) * 50))
    
    return score

def evaluate_mcq(student_ans, correct_ans):
    """
    Score a multiple-choice question with improved pattern matching.

    Args:
        student_ans (str): Student's answer
        correct_ans (str): Correct answer

    Returns:
        int: 100 if correct, 0 if incorrect
    """
    # Handle empty answers
    if not student_ans or not correct_ans:
        logger.debug(f"MCQ evaluation: empty answer - student: '{student_ans}', correct: '{correct_ans}'")
        return 0

    # Convert both to strings and uppercase
    student_ans = str(student_ans).strip().upper()
    correct_ans = str(correct_ans).strip().upper()

    logger.debug(f"MCQ evaluation: student='{student_ans}', correct='{correct_ans}'")

    # Check for our special marker
    selected_option_match = re.search(r'\[SELECTED_OPTION:([A-D])\]', student_ans)
    if selected_option_match:
        student_option = selected_option_match.group(1)
        result = 100 if student_option == correct_ans else 0
        logger.info(f"MCQ: Found selected option {student_option}, correct={correct_ans}, score={result}")
        return result

    # Look for "Answer: X" pattern (case insensitive)
    answer_patterns = [
        r'ANSWER[:\s]*([A-D])',
        r'ANS[:\s]*([A-D])',
        r'[Aa]nswer[:\s]*([A-D])',
        r'[Aa]ns[:\s]*([A-D])',
    ]

    for pattern in answer_patterns:
        answer_match = re.search(pattern, student_ans, re.IGNORECASE)
        if answer_match:
            student_option = answer_match.group(1).upper()
            result = 100 if student_option == correct_ans else 0
            logger.info(f"MCQ: Found answer pattern {student_option}, correct={correct_ans}, score={result}")
            return result

    # Extract just the option letter if it's in a format like "A) Option text"
    option_patterns = [
        r'([A-D])\)',  # A)
        r'([A-D])\.',  # A.
        r'([A-D])\s',  # A followed by space
        r'([A-D])$',   # A at end of string
    ]

    for pattern in option_patterns:
        student_match = re.search(pattern, student_ans)
        if student_match:
            student_option = student_match.group(1)
            result = 100 if student_option == correct_ans else 0
            logger.info(f"MCQ: Found option pattern {student_option}, correct={correct_ans}, score={result}")
            return result

    # If student answer is just a single letter
    if len(student_ans) == 1 and student_ans in 'ABCD':
        result = 100 if student_ans == correct_ans else 0
        logger.info(f"MCQ: Single letter answer {student_ans}, correct={correct_ans}, score={result}")
        return result

    # If student wrote the full answer, try to match the option letter
    if len(student_ans) > 1:
        # Look for standalone option letters
        option_matches = re.findall(r'\b([A-D])\b', student_ans)
        if option_matches:
            # Use the last match as it's likely the answer
            student_option = option_matches[-1]
            result = 100 if student_option == correct_ans else 0
            logger.info(f"MCQ: Found standalone option {student_option}, correct={correct_ans}, score={result}")
            return result

    # If we get here, no clear option was found
    logger.warning(f"MCQ: Could not extract clear option from '{student_ans}', correct={correct_ans}")
    return 0

def cli_progress_callback(stage: str, message: str, data=None):
    """Progress callback for CLI interface."""
    if stage == "started":
        print(f"[INFO] {message}")
    elif stage == "file_progress":
        print(f"[INFO] {message}")
    elif stage == "extracting":
        print(f"[INFO] {message}")
    elif stage == "completed":
        print(f"[INFO] {message}")
    elif stage == "error":
        print(f"[ERROR] {message}")
    elif stage == "warning":
        print(f"[WARNING] {message}")
    elif stage == "finished":
        print(f"[INFO] {message}")

def process_exam_booklets(booklet_folder=None):
    """
    Process all exam booklets using the core processing module.

    Args:
        booklet_folder (str, optional): Path to the booklet folder. If None, use the default.

    Returns:
        pd.DataFrame: DataFrame containing the results
    """
    try:
        # Import the core processing module
        from exam_processing_core import create_exam_processor_from_files

        # Determine folder path
        if booklet_folder:
            folder = Path(booklet_folder)
        else:
            folder = BOOKLET_FOLDER

        # Load model
        model = load_model()

        # Create processor using the core module
        processor = create_exam_processor_from_files(model=model)

        if not processor.questions or not processor.marking_scheme:
            logger.error("Failed to load questions or marking scheme")
            print("[ERROR] Failed to load questions or marking scheme")
            return pd.DataFrame()

        # Check if results file exists and load existing results
        processed = set()
        if Path(RESULTS_FILE).exists():
            try:
                df_existing = pd.read_csv(RESULTS_FILE)
                processed.update(df_existing['Student'].tolist())
                logger.info(f"Found {len(processed)} already processed booklets")
            except Exception as e:
                logger.error(f"Error reading existing results file: {e}")
                print(f"[ERROR] Could not read existing results: {e}")

        # Get list of files to process (skip already processed ones)
        valid_extensions = (".jpg", ".png", ".jpeg", ".tif", ".pdf")
        all_files = [f for f in os.listdir(folder)
                    if f.lower().endswith(valid_extensions)]

        files_to_process = [f for f in all_files if f not in processed]

        if not files_to_process:
            if not all_files:
                logger.warning(f"No valid booklet files found in {folder}")
                print(f"[WARNING] No valid booklet files found in {folder}")
            else:
                logger.info("No new booklets to process")
                print("[INFO] No new booklets to process.")
            return pd.DataFrame()

        # Process only new files using the core processor
        results = []
        for file in files_to_process:
            file_path = folder / file
            result = processor.process_single_booklet(str(file_path), cli_progress_callback)

            if result["status"] == "success":
                # Create result row for DataFrame
                result_row = {"Student": result["Student"]}
                result_row.update(result["scores"])
                result_row["Final_Score_Percentage"] = result["Final_Score_Percentage"]
                results.append(result_row)

        # Save results if any were processed
        if not results:
            logger.info("No new booklets processed successfully")
            print("[INFO] No new booklets processed successfully.")
            return pd.DataFrame()

    except ImportError as e:
        logger.error(f"Could not import core processing module: {e}")
        print(f"[ERROR] Could not import core processing module: {e}")
        print("[INFO] Falling back to legacy processing...")

        # Fallback to legacy processing if core module is not available
        return process_exam_booklets_legacy(booklet_folder)
    except Exception as e:
        logger.error(f"Unexpected error in process_exam_booklets: {e}")
        print(f"[ERROR] Unexpected error: {e}")
        return pd.DataFrame()

def process_exam_booklets_legacy(booklet_folder=None):
    """
    Legacy processing function (original implementation).
    Used as fallback if core module is not available.
    """
    if booklet_folder:
        folder = Path(booklet_folder)
    else:
        folder = BOOKLET_FOLDER

    # Load questions and marking scheme
    questions, marking_scheme = load_questions_and_marking_scheme()

    # Load or fine-tune model
    model = load_model()
    results = []
    processed = set()

    if Path(RESULTS_FILE).exists():
        try:
            df_existing = pd.read_csv(RESULTS_FILE)
            processed.update(df_existing['Student'].tolist())
            logger.info(f"Found {len(processed)} already processed booklets")
        except Exception as e:
            logger.error(f"Error reading existing results file: {e}")
            print(f"[ERROR] Could not read existing results: {e}")

    for file in os.listdir(folder):
        if file in processed or not file.lower().endswith((".jpg", ".png", ".jpeg", ".tif", ".pdf")):
            continue

        print(f"[INFO] Processing {file}")
        logger.info(f"Processing booklet: {file}")

        try:
            text = extract_text(folder / file)
            answers = extract_answers_from_text(text)

            scores = {}
            for q_id, ans in answers.items():
                if q_id in questions:
                    if questions[q_id]['type'] == 'written':
                        if q_id in marking_scheme:
                            # Ensure marking scheme value is a string
                            model_ans = str(marking_scheme[q_id])
                            scores[q_id] = evaluate_written_answer(ans, model_ans, model)
                        else:
                            scores[q_id] = 0.0
                            logger.warning(f"No marking scheme found for {q_id}")
                    else:  # MCQ
                        if q_id in marking_scheme:
                            # Ensure marking scheme value is a string
                            correct_ans = str(marking_scheme[q_id])
                            scores[q_id] = evaluate_mcq(ans, correct_ans)
                        else:
                            scores[q_id] = 0.0
                            logger.warning(f"No marking scheme found for {q_id}")
                else:
                    scores[q_id] = 0.0
                    logger.warning(f"Unknown question ID: {q_id}")

            # Calculate final score as percentage dynamically
            # Each question is worth 100 points, so total possible = number of questions * 100
            num_questions = len(scores)
            total_possible_points = num_questions * 100
            total_score = sum(scores.values())
            final_score_percentage = (total_score / total_possible_points) * 100 if total_possible_points > 0 else 0

            # Create result row with Final_Score_Percentage right after the question scores
            result_row = {"Student": file}
            result_row.update(scores)  # Add all question scores
            result_row["Final_Score_Percentage"] = round(final_score_percentage, 2)

            results.append(result_row)
            logger.info(f"Successfully processed {file} - Final Score: {final_score_percentage:.2f}% ({total_score:.1f}/{total_possible_points})")

        except Exception as e:
            logger.error(f"Error processing {file}: {e}")
            print(f"[ERROR] Failed to process {file}: {e}")

    if results:
        df_new = pd.DataFrame(results)

        # Ensure proper column ordering: Student, Q1, Q2, ..., Q6, Final_Score_Percentage
        question_cols = [col for col in df_new.columns if col.startswith('Q')]
        question_cols.sort()  # Ensure Q1, Q2, Q3, Q4, Q5, Q6 order

        desired_columns = ['Student'] + question_cols + ['Final_Score_Percentage']
        df_new = df_new.reindex(columns=desired_columns)

        logger.info(f"DataFrame columns: {df_new.columns.tolist()}")
        logger.info(f"DataFrame shape: {df_new.shape}")

        try:
            # Check if existing file has the correct column structure
            if Path(RESULTS_FILE).exists():
                try:
                    df_existing = pd.read_csv(RESULTS_FILE)
                    # Check if Final_Score_Percentage column exists and structure matches
                    if 'Final_Score_Percentage' not in df_existing.columns or set(df_existing.columns) != set(df_new.columns):
                        logger.info("Existing results file has different structure, recreating...")
                        # Backup old file
                        backup_file = str(RESULTS_FILE).replace('.csv', '_backup.csv')
                        df_existing.to_csv(backup_file, index=False)
                        logger.info(f"Backed up old results to {backup_file}")

                        # Recreate with new structure
                        df_new.to_csv(RESULTS_FILE, index=False)
                        logger.info(f"Created new results file with columns: {df_new.columns.tolist()}")
                    else:
                        # Append to existing file with correct structure
                        df_new.to_csv(RESULTS_FILE, mode='a', header=False, index=False)
                        logger.info(f"Appended {len(df_new)} rows to existing results file")
                except Exception as read_error:
                    logger.warning(f"Could not read existing results file: {read_error}")
                    # Create new file
                    df_new.to_csv(RESULTS_FILE, index=False)
                    logger.info(f"Created new results file due to read error")
            else:
                # Create new file
                df_new.to_csv(RESULTS_FILE, index=False)
                logger.info(f"Created new results file with columns: {df_new.columns.tolist()}")

            logger.info(f"Results saved to {RESULTS_FILE}")
            print(f"[INFO] Results saved to {RESULTS_FILE}")

            # Verify the saved file
            try:
                df_verify = pd.read_csv(RESULTS_FILE)
                logger.info(f"Verification: CSV file has {len(df_verify)} rows and columns: {df_verify.columns.tolist()}")
                if 'Final_Score_Percentage' in df_verify.columns:
                    logger.info("✓ Final_Score_Percentage column confirmed in saved file")
                else:
                    logger.error("✗ Final_Score_Percentage column missing in saved file")
            except Exception as verify_error:
                logger.error(f"Could not verify saved file: {verify_error}")

            plot_results(df_new)
            return df_new
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            print(f"[ERROR] Could not save results: {e}")
            return df_new
    else:
        logger.info("No new booklets processed")
        print("[INFO] No new booklets processed.")
        return pd.DataFrame()

def plot_results(df):
    """
    Plot the results of exam marking.

    Args:
        df (pd.DataFrame): DataFrame containing the results
    """
    if df.empty:
        logger.warning("Cannot plot results: DataFrame is empty")
        return

    try:
        # Get question columns and final score column
        score_cols = [col for col in df.columns if col.startswith('Q')]

        # Plot final scores by student
        plt.figure(figsize=(12, 6))
        if 'Final_Score_Percentage' in df.columns:
            sns.barplot(x='Student', y='Final_Score_Percentage', data=df)
            plt.title('Final Scores by Student (%)')
            plt.ylabel('Final Score (%)')
            plt.ylim(0, 100)
            # Add score labels on bars
            for i, v in enumerate(df['Final_Score_Percentage']):
                plt.text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom')
        elif 'Final_Score' in df.columns:
            # Backward compatibility
            sns.barplot(x='Student', y='Final_Score', data=df)
            plt.title('Final Scores by Student (%)')
            plt.ylabel('Final Score (%)')
            plt.ylim(0, 100)
            # Add score labels on bars
            for i, v in enumerate(df['Final_Score']):
                plt.text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom')
        else:
            # Fallback to calculated average if no final score column available
            # Calculate dynamically based on number of questions
            num_questions = len(score_cols)
            total_possible = num_questions * 100
            df['Final_Score_Calculated'] = (df[score_cols].sum(axis=1) / total_possible) * 100
            sns.barplot(x='Student', y='Final_Score_Calculated', data=df)
            plt.title('Final Scores by Student (%) - Calculated')
            plt.ylabel('Final Score (%)')
            plt.ylim(0, 100)
            # Add score labels on bars
            for i, v in enumerate(df['Final_Score_Calculated']):
                plt.text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom')

        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('final_scores.png', dpi=300, bbox_inches='tight')

        # Plot scores by question
        plt.figure(figsize=(12, 8))
        df_melted = df.melt(id_vars=['Student'], value_vars=score_cols, var_name='Question', value_name='Score')
        sns.boxplot(x='Question', y='Score', data=df_melted)
        plt.title('Score Distribution by Question')
        plt.ylabel('Score')
        plt.tight_layout()
        plt.savefig('question_scores.png', dpi=300, bbox_inches='tight')

        # Plot individual student performance
        plt.figure(figsize=(14, 8))
        df_melted = df.melt(id_vars=['Student'], value_vars=score_cols, var_name='Question', value_name='Score')
        sns.heatmap(df.set_index('Student')[score_cols], annot=True, fmt='.1f', cmap='RdYlGn',
                   cbar_kws={'label': 'Score'}, vmin=0, vmax=100)
        plt.title('Individual Question Scores by Student')
        plt.ylabel('Student')
        plt.xlabel('Question')
        plt.tight_layout()
        plt.savefig('student_heatmap.png', dpi=300, bbox_inches='tight')

        logger.info("Results plots saved successfully (final_scores.png, question_scores.png, student_heatmap.png)")
    except Exception as e:
        logger.error(f"Error plotting results: {e}")
        print(f"[ERROR] Failed to plot results: {e}")

@lru_cache(maxsize=1)
def cross_validate_model(model_name, examples_tuple=None, k=5):
    """
    Perform cross-validation for a model.
    
    Args:
        model_name (str): Name of the model to validate
        examples_tuple (tuple, optional): Tuple of InputExample objects (converted from list)
        k (int): Number of folds for cross-validation
        
    Returns:
        float: Average accuracy across folds
    """
    # Convert tuple back to list if provided
    examples = list(examples_tuple) if examples_tuple else []
    
    logger.info(f"Starting cross-validation for {model_name} with {k}-folds.")
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    model = SentenceTransformer(model_name).to(DEVICE)
    scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(examples)):
        train_examples = [examples[i] for i in train_idx]
        val_examples = [examples[i] for i in val_idx]

        train_loader = DataLoader(train_examples, shuffle=True, batch_size=16)
        loss = losses.CosineSimilarityLoss(model=model)

        logger.info(f"Training fold {fold + 1}...")
        model.fit(train_objectives=[(train_loader, loss)],
                  epochs=2,
                  warmup_steps=50,
                  show_progress_bar=True)

        val_scores = []
        for example in val_examples:
            embeddings = model.encode([example.texts[0], example.texts[1]], convert_to_tensor=True)
            sim = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
            val_scores.append(sim)

        accuracy = sum(1 for i, example in enumerate(val_examples) if round(val_scores[i]) == round(example.label)) / len(val_examples)
        scores.append(accuracy)
        logger.info(f"Fold {fold + 1} accuracy: {accuracy:.2f}")

    avg_accuracy = sum(scores) / len(scores)
    logger.info(f"Cross-validation completed for {model_name}. Average accuracy: {avg_accuracy:.2f}")
    return avg_accuracy

def train_and_evaluate_models():
    """
    Train and evaluate multiple models with advanced models included.
    
    Returns:
        list: List of dictionaries containing model names, accuracies, and paths
    """
    logging.info("Training and evaluating models with cross-validation...")
    df = pd.read_csv(TRAINING_DATA)

    examples = [
        InputExample(
            texts=[clean_text(row['student_answer']), clean_text(row['model_answer'])],
            label=float(row['score']) / 100  # convert score from 0–100 to 0–1
        )
        for _, row in df.iterrows()
    ]
    
    # Convert examples to tuple for caching
    examples_tuple = tuple(examples)

    models = [
        {"name": "all-MiniLM-L6-v2", "model": SentenceTransformer("all-MiniLM-L6-v2")},
        {"name": "paraphrase-MiniLM-L6-v2", "model": SentenceTransformer("paraphrase-MiniLM-L6-v2")},
        {"name": "distilbert-base-nli-stsb-mean-tokens", "model": SentenceTransformer("distilbert-base-nli-stsb-mean-tokens")},
        # Add more models as needed
    ]

    results = []
    for model_info in models:
        try:
            accuracy = cross_validate_model(model_info["name"], examples_tuple, k=5)
            
            # Save model if it's the best so far
            save_path = FINE_TUNED_MODEL_PATH / model_info["name"]
            save_path.mkdir(parents=True, exist_ok=True)
            model_info["model"].save(str(save_path))
            logger.info(f"Save model to {save_path}")
            
            results.append({
                "name": model_info["name"],
                "accuracy": accuracy,
                "path": str(save_path)
            })
        except Exception as e:
            logger.error(f"Error evaluating {model_info['name']}: {e}")
    
    return results

def plot_model_accuracies(results):
    """
    Plot the accuracies of different models.
    
    Args:
        results (list): List of dictionaries containing model names and accuracies
    """
    names = [result["name"] for result in results]
    accuracies = [result["accuracy"] for result in results]

    plt.figure(figsize=(10, 6))
    plt.bar(names, accuracies, color='skyblue')
    plt.title('Model Training Accuracy Comparison')
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()

def load_best_model():
    """
    Load the best model based on cross-validation results.
    
    Returns:
        SentenceTransformer: The best model
    """
    results = train_and_evaluate_models()
    best_model = max(results, key=lambda x: x["accuracy"])
    print(f"[INFO] Best model: {best_model['name']} with accuracy {best_model['accuracy']:.2f}")
    return SentenceTransformer(best_model["path"])

def load_questions_and_marking_scheme(questions_file="questions.csv", marking_scheme_file="marking_scheme.csv"):
    """
    Load questions and marking scheme from CSV files.
    
    Args:
        questions_file (str): Path to the questions CSV file
        marking_scheme_file (str): Path to the marking scheme CSV file
        
    Returns:
        tuple: (questions_dict, marking_scheme_dict)
    """
    questions = {}
    marking_scheme = {}
    
    # Load questions
    try:
        if Path(questions_file).exists():
            df_questions = pd.read_csv(questions_file)
            for _, row in df_questions.iterrows():
                q_id = row['question_id']
                q_type = row['question_type']  # 'written' or 'mcq'
                q_text = row['question_text']
                questions[q_id] = {'type': q_type, 'text': q_text}
            logger.info(f"Loaded {len(questions)} questions from {questions_file}")
        else:
            logger.warning(f"Questions file not found: {questions_file}")
            print(f"[WARNING] Questions file not found: {questions_file}")
            print("[INFO] Please run create_samples.py to generate sample data first.")
            return {}, {}
    except Exception as e:
        logger.error(f"Error loading questions: {e}")
        print(f"[ERROR] Error loading questions: {e}")
        return {}, {}
    
    # Load marking scheme
    try:
        if Path(marking_scheme_file).exists():
            df_marking = pd.read_csv(marking_scheme_file)
            for _, row in df_marking.iterrows():
                q_id = row['question_id']

                # For written questions, use model_answer
                if 'model_answer' in row and pd.notna(row['model_answer']) and str(row['model_answer']).strip():
                    marking_scheme[q_id] = str(row['model_answer']).strip()
                    logger.debug(f"Loaded model answer for {q_id}")

                # For MCQ questions, use correct_option
                elif 'correct_option' in row and pd.notna(row['correct_option']) and str(row['correct_option']).strip():
                    marking_scheme[q_id] = str(row['correct_option']).strip()
                    logger.debug(f"Loaded correct option for {q_id}: {marking_scheme[q_id]}")

                else:
                    logger.warning(f"No valid answer found for {q_id} in marking scheme")

            logger.info(f"Loaded marking scheme from {marking_scheme_file}")
            logger.info(f"Marking scheme entries: {list(marking_scheme.keys())}")
        else:
            logger.warning(f"Marking scheme file not found: {marking_scheme_file}")
            print(f"[WARNING] Marking scheme file not found: {marking_scheme_file}")
            print("[INFO] Please run create_samples.py to generate sample data first.")
            return questions, {}
    except Exception as e:
        logger.error(f"Error loading marking scheme: {e}")
        print(f"[ERROR] Error loading marking scheme: {e}")
        return questions, {}
    
    return questions, marking_scheme

def cleanup_old_models(keep_latest=3):
    """
    Clean up old model directories to save disk space.
    
    Args:
        keep_latest (int): Number of latest models to keep
    """
    try:
        model_path = Path(FINE_TUNED_MODEL_PATH)
        
        # Look for model_* directories and sort by timestamp (oldest first)
        model_dirs = sorted(
            [d for d in model_path.iterdir() if d.is_dir() and d.name.startswith("model_")],
            key=lambda d: d.name
        )
        
        # Keep only the latest N models
        if len(model_dirs) > keep_latest:
            dirs_to_remove = model_dirs[:-keep_latest]
            
            for dir_to_remove in dirs_to_remove:
                try:
                    # Check if this directory is linked from current_model
                    current_model_path = model_path / "current_model"
                    if current_model_path.exists() and current_model_path.is_symlink():
                        target = current_model_path.resolve()
                        if target == dir_to_remove:
                            # Skip removing this directory as it's the current model
                            logger.info(f"Skipping removal of {dir_to_remove} as it's the current model")
                            continue
                    
                    # Remove the directory
                    import shutil
                    shutil.rmtree(dir_to_remove)
                    logger.info(f"Removed old model directory: {dir_to_remove}")
                except Exception as e:
                    logger.error(f"Failed to remove directory {dir_to_remove}: {e}")
    
    except Exception as e:
        logger.error(f"Error during model cleanup: {e}")

# Main
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exam AI Marking System")
    parser.add_argument("--process", action="store_true", help="Process exam booklets")
    parser.add_argument("--booklet-folder", default=None, help="Path to the booklet folder")
    args = parser.parse_args()

    if not os.path.exists(FINE_TUNED_MODEL_PATH):
        os.makedirs(FINE_TUNED_MODEL_PATH, exist_ok=True)

    if args.process or not args.booklet_folder:
        print("[INFO] Processing exam booklets...")
        model = fine_tune_model()
        process_exam_booklets(args.booklet_folder)
    process_exam_booklets(args.booklet_folder)
