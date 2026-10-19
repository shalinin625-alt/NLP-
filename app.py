from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
    jsonify,
    send_file
)

import os
import re
import io
import csv
import pickle
import math
from email import policy
from email.parser import BytesParser
from collections import Counter

import pandas as pd
import numpy as np

from werkzeug.utils import secure_filename

# Optional libraries
try:
    from PyPDF2 import PdfReader
except Exception:
    PdfReader = None

try:
    from docx import Document
except Exception:
    Document = None

try:
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        pipeline
    )
except Exception:
    AutoTokenizer = None
    AutoModelForSequenceClassification = None
    pipeline = None


# ============================================================
# APP CONFIGURATION
# ============================================================

app = Flask(__name__)

app.secret_key = "intellireq_secret_key_2026"

UPLOAD_FOLDER = "uploads"
REPORT_FOLDER = "reports"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

# Step 15 - Multi-format input
ALLOWED_EXTENSIONS = {"pdf", "docx", "txt", "csv", "eml"}


# ============================================================
# GLOBAL MODEL VARIABLES
# ============================================================

bert_model = None
bert_tokenizer = None
bert_classifier = None
bert_label_encoder = None

logistic_model = None
logistic_vectorizer = None

svm_model = None
svm_vectorizer = None

random_forest_model = None
random_forest_vectorizer = None


# ============================================================
# LOAD MODEL SAFELY
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

BERT_FOLDER = os.path.join(
    BASE_DIR,
    "bert_requirement_classifier"
)

BERT_ENCODER = os.path.join(
    BASE_DIR,
    "bert_label_encoder.pkl"
)


def safe_pickle_load(path):
    """
    Safely load pickle files.

    Some older project pickle files may be incompatible or corrupted.
    Instead of crashing Flask, this function returns None.
    """
    if not os.path.exists(path):
        return None

    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Could not load {os.path.basename(path)}: {e}")
        return None


def load_models():

    global bert_model
    global bert_tokenizer
    global bert_classifier
    global bert_label_encoder

    global logistic_model
    global logistic_vectorizer

    global svm_model
    global svm_vectorizer

    global random_forest_model
    global random_forest_vectorizer

    # --------------------------------------------------------
    # Logistic Regression
    # --------------------------------------------------------

    logistic_model = safe_pickle_load(
        os.path.join(BASE_DIR, "logistic_regression_model.pkl")
    )

    logistic_vectorizer = safe_pickle_load(
        os.path.join(BASE_DIR, "tfidf_vectorizer.pkl")
    )

    # --------------------------------------------------------
    # SVM
    # --------------------------------------------------------

    svm_model = safe_pickle_load(
        os.path.join(BASE_DIR, "svm_model.pkl")
    )

    svm_vectorizer = safe_pickle_load(
        os.path.join(BASE_DIR, "svm_tfidf_vectorizer.pkl")
    )

    # --------------------------------------------------------
    # Random Forest
    # --------------------------------------------------------

    random_forest_model = safe_pickle_load(
        os.path.join(BASE_DIR, "random_forest_model.pkl")
    )

    random_forest_vectorizer = safe_pickle_load(
        os.path.join(BASE_DIR, "random_forest_vectorizer.pkl")
    )

    # --------------------------------------------------------
    # BERT
    # --------------------------------------------------------

    if (
        AutoTokenizer is not None
        and AutoModelForSequenceClassification is not None
        and os.path.exists(BERT_FOLDER)
    ):

        try:

            bert_tokenizer = AutoTokenizer.from_pretrained(
                BERT_FOLDER
            )

            bert_model = AutoModelForSequenceClassification.from_pretrained(
                BERT_FOLDER
            )

            if pipeline is not None:

                bert_classifier = pipeline(
                    "text-classification",
                    model=bert_model,
                    tokenizer=bert_tokenizer,
                    top_k=None
                )

            bert_label_encoder = safe_pickle_load(
                BERT_ENCODER
            )

            print("BERT model loaded successfully!")

        except Exception as e:

            print("Could not load BERT:", e)

    else:

        print("BERT model folder not found.")


# ============================================================
# LOAD MODELS
# ============================================================

load_models()


# ============================================================
# TEXT EXTRACTION - STEP 15 MULTI-FORMAT INPUT
# ============================================================

def allowed_file(filename):
    """Return True when the uploaded file extension is supported."""
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def extract_pdf_text(filepath):
    if PdfReader is None:
        return ""

    try:
        reader = PdfReader(filepath)
        pages = []

        for page in reader.pages:
            page_text = page.extract_text() or ""
            if page_text.strip():
                pages.append(page_text.strip())

        return "\n".join(pages)

    except Exception as e:
        print("PDF extraction error:", e)
        return ""


def extract_docx_text(filepath):
    if Document is None:
        return ""

    try:
        document = Document(filepath)
        parts = []

        # Normal paragraphs
        for paragraph in document.paragraphs:
            text = paragraph.text.strip()
            if text:
                parts.append(text)

        # Requirement tables are also common in SRS documents
        for table in document.tables:
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if cells:
                    parts.append(" | ".join(cells))

        return "\n".join(parts)

    except Exception as e:
        print("DOCX extraction error:", e)
        return ""


def extract_txt_text(filepath):
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception as e:
        print("TXT extraction error:", e)
        return ""


def extract_csv_text(filepath):
    """
    Extract requirement text from CSV files.

    IntelliReq first looks for common requirement-column names. If no
    known column exists, all non-empty cell values are combined row-wise.
    """
    try:
        dataframe = pd.read_csv(filepath, encoding="utf-8", on_bad_lines="skip")
    except UnicodeDecodeError:
        try:
            dataframe = pd.read_csv(filepath, encoding="latin-1", on_bad_lines="skip")
        except Exception as e:
            print("CSV extraction error:", e)
            return ""
    except Exception as e:
        print("CSV extraction error:", e)
        return ""

    if dataframe.empty:
        return ""

    normalized_columns = {
        str(column).strip().lower().replace("-", "_"): column
        for column in dataframe.columns
    }

    preferred_names = [
        "requirement",
        "requirements",
        "description",
        "requirement_description",
        "user_story",
        "user story",
        "story",
    ]

    selected_column = None
    for name in preferred_names:
        key = name.replace("-", "_")
        if key in normalized_columns:
            selected_column = normalized_columns[key]
            break

    lines = []

    if selected_column is not None:
        for value in dataframe[selected_column].dropna().astype(str):
            value = value.strip()
            if value and value.lower() != "nan":
                lines.append(value)
    else:
        for _, row in dataframe.iterrows():
            values = []
            for value in row.tolist():
                if pd.isna(value):
                    continue
                text = str(value).strip()
                if text and text.lower() != "nan":
                    values.append(text)
            if values:
                lines.append(" ".join(values))

    return "\n".join(lines)


def _decode_email_part(part):
    try:
        payload = part.get_payload(decode=True)
        if payload is None:
            return ""

        charset = part.get_content_charset() or "utf-8"
        return payload.decode(charset, errors="ignore")
    except Exception:
        return ""


def extract_email_text(filepath):
    """Extract Subject and plain-text body from an RFC .eml email file."""
    try:
        with open(filepath, "rb") as f:
            message = BytesParser(policy=policy.default).parse(f)

        parts = []

        subject = str(message.get("subject", "")).strip()
        sender = str(message.get("from", "")).strip()

        if subject:
            parts.append(f"Email subject: {subject}")
        if sender:
            parts.append(f"Email from: {sender}")

        body_chunks = []

        if message.is_multipart():
            for part in message.walk():
                content_type = part.get_content_type()
                disposition = str(part.get("Content-Disposition", "")).lower()

                if "attachment" in disposition:
                    continue

                if content_type == "text/plain":
                    text = _decode_email_part(part).strip()
                    if text:
                        body_chunks.append(text)
        else:
            if message.get_content_type() == "text/plain":
                text = _decode_email_part(message).strip()
                if text:
                    body_chunks.append(text)

        if not body_chunks:
            try:
                fallback = message.get_body(preferencelist=("plain",))
                if fallback is not None:
                    content = fallback.get_content().strip()
                    if content:
                        body_chunks.append(content)
            except Exception:
                pass

        if body_chunks:
            parts.append("\n".join(body_chunks))

        return "\n".join(parts)

    except Exception as e:
        print("Email extraction error:", e)
        return ""


def convert_user_story_to_requirement(story):
    """Convert a common Agile user story into a requirement sentence."""
    story = re.sub(r"\s+", " ", story).strip().rstrip(".")

    pattern = re.compile(
        r"^as\s+an?\s+(.+?),\s*i\s+want\s+(?:to\s+)?(.+?)(?:,?\s*so\s+that\s+(.+))?$",
        re.IGNORECASE,
    )

    match = pattern.match(story)
    if not match:
        return story + ("" if story.endswith(".") else ".")

    actor = match.group(1).strip()
    goal = match.group(2).strip()
    benefit = match.group(3).strip() if match.group(3) else ""

    requirement = f"The system shall allow the {actor} to {goal}"
    if benefit:
        requirement += f" so that {benefit}"

    return requirement.rstrip(".") + "."


def extract_user_stories(text):
    """Find Agile 'As a/an ..., I want ...' statements and convert them."""
    if not text:
        return []

    candidates = re.split(r"\n+|(?<=[.!?])\s+(?=[A-Z])", text)
    stories = []

    for candidate in candidates:
        candidate = re.sub(r"^\s*(?:\d+[.\)]|[-*•])\s*", "", candidate).strip()
        if re.match(r"^as\s+an?\s+.+?,\s*i\s+want\s+", candidate, re.IGNORECASE):
            stories.append(convert_user_story_to_requirement(candidate))

    return stories


def extract_text_from_file(filepath):
    """Dispatch extraction by file extension."""
    extension = os.path.splitext(filepath)[1].lower()

    if extension == ".txt":
        return extract_txt_text(filepath)

    if extension == ".pdf":
        return extract_pdf_text(filepath)

    if extension == ".docx":
        return extract_docx_text(filepath)

    if extension == ".csv":
        return extract_csv_text(filepath)

    if extension == ".eml":
        return extract_email_text(filepath)

    return ""


# ============================================================
# REQUIREMENT EXTRACTION
# ============================================================

def extract_requirements(text):

    if not text:
        return []

    text = text.replace("\r", "\n")

    lines = re.split(
        r"\n+|(?<=[.!?])\s+(?=[A-Z])",
        text
    )

    requirements = []

    for line in lines:

        line = line.strip()

        # Remove numbering
        line = re.sub(
            r"^\s*(?:\d+[\.\)]|[-*•])\s*",
            "",
            line
        )

        if len(line) < 10:
            continue

        requirements.append(line)

    return requirements


# ============================================================
# TEXT PREPROCESSING
# ============================================================

def preprocess_text(text):

    if not text:
        return ""

    text = text.lower()

    text = re.sub(
        r"[^a-zA-Z0-9\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ============================================================
# KEYWORD DETECTION
# ============================================================

AMBIGUOUS_WORDS = [
    "fast",
    "quick",
    "easy",
    "simple",
    "user-friendly",
    "soon",
    "efficient",
    "adequate",
    "appropriate",
    "reasonable",
    "good",
    "secure",
    "high performance",
    "low cost",
    "etc",
    "approximately"
]


SECURITY_KEYWORDS = [
    "password",
    "authentication",
    "authorization",
    "login",
    "security",
    "encrypt",
    "encryption",
    "secure",
    "token",
    "access control",
    "permission",
    "privacy",
    "firewall",
    "vulnerability"
]


PERFORMANCE_KEYWORDS = [
    "response time",
    "latency",
    "milliseconds",
    "seconds",
    "requests per second",
    "throughput",
    "performance",
    "load",
    "concurrent",
    "response",
    "scalable",
    "availability"
]


def find_keywords(text, keywords):

    text_lower = text.lower()

    found = []

    for keyword in keywords:

        if keyword.lower() in text_lower:

            found.append(keyword)

    return found


# ============================================================
# REQUIREMENT TYPE CLASSIFICATION
# ============================================================

def rule_based_classification(text):

    lower = text.lower()

    security_score = sum(
        1 for word in SECURITY_KEYWORDS
        if word in lower
    )

    performance_score = sum(
        1 for word in PERFORMANCE_KEYWORDS
        if word in lower
    )

    usability_words = [
        "easy",
        "simple",
        "user-friendly",
        "accessible",
        "interface",
        "usability"
    ]

    usability_score = sum(
        1 for word in usability_words
        if word in lower
    )

    if security_score > 0:

        return "Security", 0.90

    if performance_score > 0:

        return "Performance", 0.90

    if usability_score > 0:

        return "Usability", 0.85

    return "Functional", 0.80


# ============================================================
# BERT CLASSIFICATION
# ============================================================

def bert_predict(text):

    if bert_classifier is None:

        return None, 0.0

    try:

        result = bert_classifier(text)

        if isinstance(result, list):

            if len(result) > 0 and isinstance(result[0], list):
                result = result[0]

        if not result:

            return None, 0.0

        best = max(
            result,
            key=lambda x: x.get("score", 0)
        )

        label = best.get("label", "")

        score = float(
            best.get("score", 0)
        )

        # Convert LABEL_0 etc.
        label_map = {
            "LABEL_0": "Functional",
            "LABEL_1": "Performance",
            "LABEL_2": "Security",
            "LABEL_3": "Usability"
        }

        if label in label_map:

            label = label_map[label]

        elif bert_label_encoder is not None:

            try:

                number = int(
                    label.replace("LABEL_", "")
                )

                label = bert_label_encoder.inverse_transform(
                    [number]
                )[0]

            except Exception:
                pass

        return label, score

    except Exception as e:

        print("BERT prediction error:", e)

        return None, 0.0


# ============================================================
# QUALITY SCORE
# ============================================================

def calculate_quality(text):

    score = 100

    lower = text.lower()

    # Ambiguity
    ambiguity_count = sum(
        1
        for word in AMBIGUOUS_WORDS
        if word in lower
    )

    score -= min(
        ambiguity_count * 5,
        25
    )

    # Weak wording
    weak_words = [
        "should",
        "may",
        "might",
        "could"
    ]

    weak_count = sum(
        1
        for word in weak_words
        if re.search(
            r"\b" + re.escape(word) + r"\b",
            lower
        )
    )

    score -= min(
        weak_count * 5,
        15
    )

    # Very short requirement
    if len(text.split()) < 6:

        score -= 10

    # Missing measurable details
    measurable_words = [
        "seconds",
        "milliseconds",
        "%",
        "maximum",
        "minimum",
        "within",
        "at least",
        "no more than"
    ]

    if not any(
        word in lower
        for word in measurable_words
    ):

        score -= 3

    return max(
        0,
        min(100, score)
    )


# ============================================================
# COMPLETENESS
# ============================================================

def calculate_completeness(text):

    lower = text.lower()

    score = 100

    if len(text.split()) < 6:
        score -= 20

    if not any(
        word in lower
        for word in [
            "shall",
            "must",
            "will",
            "should"
        ]
    ):
        score -= 15

    if not any(
        word in lower
        for word in [
            "system",
            "application",
            "user",
            "service"
        ]
    ):
        score -= 10

    return max(
        0,
        min(100, score)
    )


# ============================================================
# RISK LEVEL
# ============================================================

def calculate_risk(
    quality,
    security,
    performance,
    ambiguous
):

    if security:

        return "High"

    if quality < 60:

        return "High"

    if ambiguous and quality < 80:

        return "Medium"

    if performance and quality < 75:

        return "Medium"

    return "Low"


# ============================================================
# PRIORITY
# ============================================================

def calculate_priority(
    risk,
    security,
    performance
):

    if security:

        return "Critical"

    if risk == "High":

        return "High"

    if performance:

        return "High"

    if risk == "Medium":

        return "Medium"

    return "Low"


# ============================================================
# AI REQUIREMENT REWRITING
# ============================================================

def improve_requirement(text):

    improved = text.strip()

    replacements = {
        "should": "shall",
        "must": "shall",
        "easy": "easy to use",
        "fast": "within 2 seconds",
        "quick": "within 2 seconds",
        "soon": "within 2 seconds",
        "good": "meet the defined quality criteria",
        "secure": "using industry-standard security controls"
    }

    for old, new in replacements.items():

        improved = re.sub(
            r"\b" + re.escape(old) + r"\b",
            new,
            improved,
            flags=re.IGNORECASE
        )

    if not improved.endswith("."):

        improved += "."

    return improved


def generate_suggestions(
    text,
    quality,
    ambiguous
):

    suggestions = []

    if ambiguous:

        suggestions.append(
            "Replace ambiguous words with measurable values."
        )

    if "should" in text.lower():

        suggestions.append(
            "Replace weak words such as 'should' with 'shall'."
        )

    if not re.search(
        r"\b\d+\b|seconds|milliseconds|%",
        text.lower()
    ):

        suggestions.append(
            "Add measurable performance or acceptance criteria."
        )

    if quality >= 90:

        suggestions.append(
            "Requirement is already well defined."
        )

    return suggestions


# ============================================================
# CONTRADICTION DETECTION
# ============================================================

def normalize_for_comparison(text):

    text = preprocess_text(text)

    text = re.sub(
        r"\bshall\b",
        "",
        text
    )

    text = re.sub(
        r"\bmust\b",
        "",
        text
    )

    text = re.sub(
        r"\bnot\b",
        "",
        text
    )

    return text.strip()


def detect_contradictions(requirements):

    contradictions = set()

    normalized = [
        normalize_for_comparison(r["requirement"])
        for r in requirements
    ]

    for i in range(len(requirements)):

        for j in range(i + 1, len(requirements)):

            a = requirements[i]["requirement"].lower()
            b = requirements[j]["requirement"].lower()

            a_has_not = bool(
                re.search(r"\bnot\b", a)
            )

            b_has_not = bool(
                re.search(r"\bnot\b", b)
            )

            a_clean = normalize_for_comparison(a)
            b_clean = normalize_for_comparison(b)

            similarity = (
                a_clean == b_clean
                or (
                    len(set(a_clean.split()) & set(b_clean.split()))
                    >= max(
                        4,
                        min(
                            len(a_clean.split()),
                            len(b_clean.split())
                        ) * 0.65
                    )
                )
            )

            if similarity and a_has_not != b_has_not:

                contradictions.add(i)
                contradictions.add(j)

    return contradictions


# ============================================================
# DUPLICATE DETECTION
# ============================================================

def detect_duplicates(requirements):

    duplicates = set()

    seen = {}

    for index, req in enumerate(requirements):

        normalized = preprocess_text(
            req["requirement"]
        )

        if normalized in seen:

            duplicates.add(index)
            duplicates.add(seen[normalized])

        else:

            seen[normalized] = index

    return duplicates


# ============================================================
# TEST CASE GENERATOR
# ============================================================

def generate_test_cases(requirement):

    text = requirement["requirement"]

    req_id = requirement["requirement_id"]

    cases = []

    base = text.lower()

    # Positive
    cases.append({
        "test_case_id": f"TC-{req_id.replace('REQ-', '')}-01",
        "requirement_id": req_id,
        "title": "Valid Requirement Execution",
        "type": "Positive",
        "preconditions": "System is available and required data is valid.",
        "steps": [
            "Open the application.",
            "Provide valid input.",
            "Execute the requested operation."
        ],
        "expected_result": "The system shall perform the requested operation successfully.",
        "priority": requirement["priority"]
    })

    # Negative
    cases.append({
        "test_case_id": f"TC-{req_id.replace('REQ-', '')}-02",
        "requirement_id": req_id,
        "title": "Invalid Input Handling",
        "type": "Negative",
        "preconditions": "System is available.",
        "steps": [
            "Open the application.",
            "Provide invalid or incomplete input.",
            "Submit the request."
        ],
        "expected_result": "The system shall reject invalid input and display an appropriate error message.",
        "priority": "High"
    })

    # Boundary
    cases.append({
        "test_case_id": f"TC-{req_id.replace('REQ-', '')}-03",
        "requirement_id": req_id,
        "title": "Boundary Condition Test",
        "type": "Boundary",
        "preconditions": "System is available.",
        "steps": [
            "Enter the minimum valid value.",
            "Execute the operation.",
            "Enter the maximum valid value.",
            "Execute the operation again."
        ],
        "expected_result": "The system shall correctly handle values at the defined boundaries.",
        "priority": "Medium"
    })

    # Security
    if requirement["security"]:

        cases.append({
            "test_case_id": f"TC-{req_id.replace('REQ-', '')}-04",
            "requirement_id": req_id,
            "title": "Security Validation",
            "type": "Security",
            "preconditions": "Security controls are enabled.",
            "steps": [
                "Attempt the operation using unauthorized or malicious input.",
                "Submit the request."
            ],
            "expected_result": "The system shall prevent unauthorized access and protect sensitive information.",
            "priority": "Critical"
        })

    # Performance
    if requirement["performance"]:

        cases.append({
            "test_case_id": f"TC-{req_id.replace('REQ-', '')}-05",
            "requirement_id": req_id,
            "title": "Performance Validation",
            "type": "Performance",
            "preconditions": "System is running under expected load.",
            "steps": [
                "Send the required request.",
                "Measure the response time.",
                "Repeat the request under concurrent load."
            ],
            "expected_result": "The system shall meet the specified performance and response-time requirements.",
            "priority": "High"
        })

    return cases


def generate_all_test_cases(requirements):

    all_cases = []

    for requirement in requirements:

        cases = generate_test_cases(requirement)

        requirement["test_cases"] = cases

        requirement["test_case_count"] = len(cases)

        all_cases.extend(cases)

    return all_cases


# ============================================================
# ANALYZE SINGLE REQUIREMENT
# ============================================================

def analyze_requirement(text, requirement_id):

    ambiguous_words = find_keywords(
        text,
        AMBIGUOUS_WORDS
    )

    security_keywords = find_keywords(
        text,
        SECURITY_KEYWORDS
    )

    performance_keywords = find_keywords(
        text,
        PERFORMANCE_KEYWORDS
    )

    ambiguous = len(ambiguous_words) > 0

    security = len(security_keywords) > 0

    performance = len(performance_keywords) > 0

    # BERT first
    bert_type, bert_confidence = bert_predict(text)

    if bert_type:

        requirement_type = bert_type
        confidence = round(
            bert_confidence * 100,
            2
        )

        classification_source = "BERT"

    else:

        requirement_type, confidence = rule_based_classification(
            text
        )

        confidence = round(
            confidence * 100,
            2
        )

        classification_source = "Rule-Based"

    quality = calculate_quality(text)

    completeness = calculate_completeness(text)

    risk = calculate_risk(
        quality,
        security,
        performance,
        ambiguous
    )

    priority = calculate_priority(
        risk,
        security,
        performance
    )

    improved = improve_requirement(text)

    suggestions = generate_suggestions(
        text,
        quality,
        ambiguous
    )

    return {
        "requirement_id": requirement_id,
        "requirement": text,

        "requirement_type": requirement_type,
        "type": requirement_type,

        "confidence": confidence,
        "classification_source": classification_source,

        "ambiguity_words": ambiguous_words,
        "ambiguous": ambiguous,

        "security_keywords": security_keywords,
        "security": security,

        "performance_keywords": performance_keywords,
        "performance": performance,

        "quality_score": quality,
        "quality": quality,

        "completeness": completeness,

        "risk": risk,
        "risk_level": risk,

        "priority": priority,

        "suggestion": (
            suggestions[0]
            if suggestions
            else "Requirement looks good."
        ),

        "suggestions": suggestions,

        "improved_requirement": improved,

        "contradiction": False,
        "is_contradiction": False,

        "duplicate": False,
        "is_duplicate": False,

        "processed_text": preprocess_text(text),

        "test_cases": [],
        "test_case_count": 0
    }


# ============================================================
# UPDATE RELATIONSHIPS
# ============================================================

def update_relationships(requirements):

    contradiction_indexes = detect_contradictions(
        requirements
    )

    duplicate_indexes = detect_duplicates(
        requirements
    )

    for i, requirement in enumerate(requirements):

        contradiction = i in contradiction_indexes

        duplicate = i in duplicate_indexes

        requirement["contradiction"] = contradiction
        requirement["is_contradiction"] = contradiction

        requirement["duplicate"] = duplicate
        requirement["is_duplicate"] = duplicate

    return requirements


# ============================================================
# SUMMARY
# ============================================================

def create_summary(requirements):

    total = len(requirements)

    if total == 0:

        return {
            "total": 0,
            "average_quality": 0,

            "high_risk": 0,
            "medium_risk": 0,
            "low_risk": 0,

            "critical": 0,
            "high_priority": 0,
            "medium_priority": 0,
            "low_priority": 0,

            "contradictions": 0,
            "duplicates": 0,

            "security": 0,
            "performance": 0,

            "test_cases": 0,

            "categories": {
                "Functional": 0,
                "Performance": 0,
                "Security": 0,
                "Usability": 0
            }
        }

    quality_values = [
        float(
            r.get(
                "quality_score",
                r.get("quality", 0)
            )
        )
        for r in requirements
    ]

    return {

        "total": total,

        "average_quality": round(
            sum(quality_values) / total,
            2
        ),

        "high_risk": sum(
            1 for r in requirements
            if r.get("risk_level", r.get("risk")) == "High"
        ),

        "medium_risk": sum(
            1 for r in requirements
            if r.get("risk_level", r.get("risk")) == "Medium"
        ),

        "low_risk": sum(
            1 for r in requirements
            if r.get("risk_level", r.get("risk")) == "Low"
        ),

        "critical": sum(
            1 for r in requirements
            if r.get("priority") == "Critical"
        ),

        "high_priority": sum(
            1 for r in requirements
            if r.get("priority") == "High"
        ),

        "medium_priority": sum(
            1 for r in requirements
            if r.get("priority") == "Medium"
        ),

        "low_priority": sum(
            1 for r in requirements
            if r.get("priority") == "Low"
        ),

        "contradictions": sum(
            1 for r in requirements
            if r.get("contradiction", False)
        ),

        "duplicates": sum(
            1 for r in requirements
            if r.get("duplicate", False)
        ),

        "security": sum(
            1 for r in requirements
            if r.get("security", False)
        ),

        "performance": sum(
            1 for r in requirements
            if r.get("performance", False)
        ),

        "test_cases": sum(
            r.get("test_case_count", 0)
            for r in requirements
        ),

        "categories": {

            "Functional": sum(
                1 for r in requirements
                if r.get(
                    "requirement_type",
                    r.get("type")
                ) == "Functional"
            ),

            "Performance": sum(
                1 for r in requirements
                if r.get(
                    "requirement_type",
                    r.get("type")
                ) == "Performance"
            ),

            "Security": sum(
                1 for r in requirements
                if r.get(
                    "requirement_type",
                    r.get("type")
                ) == "Security"
            ),

            "Usability": sum(
                1 for r in requirements
                if r.get(
                    "requirement_type",
                    r.get("type")
                ) == "Usability"
            )
        }
    }


# ============================================================
# HOME
# ============================================================

@app.route("/")
def index():

    requirements = session.get(
        "requirements",
        []
    )

    summary = create_summary(
        requirements
    )

    return render_template(
        "index.html",
        requirements=requirements,
        summary=summary
    )


# ============================================================
# UPLOAD - STEP 15 MULTI-FORMAT INPUT
# ============================================================

@app.route(
    "/upload",
    methods=["POST"]
)
def upload():

    file = request.files.get("file")

    if not file or file.filename == "":
        return redirect(url_for("index"))

    if not allowed_file(file.filename):
        return (
            "Unsupported file format. Please upload PDF, DOCX, TXT, CSV, or EML.",
            400,
        )

    filename = secure_filename(file.filename)

    if not filename:
        return "Invalid filename.", 400

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    file.save(filepath)

    text = extract_text_from_file(filepath)

    if not text or not text.strip():
        return (
            "No readable text was found in the uploaded file.",
            400,
        )

    # Detect and normalize Agile user stories first.
    user_story_requirements = extract_user_stories(text)

    # Continue using the existing IntelliReq requirement extractor.
    raw_requirements = extract_requirements(text)

    # Replace raw user-story lines with their requirement-style versions
    # and avoid duplicate copies of the same text.
    converted_story_map = {}
    for raw in raw_requirements:
        if re.match(
            r"^as\s+an?\s+.+?,\s*i\s+want\s+",
            raw,
            re.IGNORECASE
        ):
            converted_story_map[raw] = convert_user_story_to_requirement(raw)

    normalized_requirements = []
    seen = set()

    for raw in raw_requirements:
        candidate = converted_story_map.get(raw, raw).strip()
        key = preprocess_text(candidate)
        if candidate and key and key not in seen:
            seen.add(key)
            normalized_requirements.append(candidate)

    # Some email/CSV formatting can cause user stories to be missed by the
    # generic extractor, so merge separately detected stories as a fallback.
    for story in user_story_requirements:
        key = preprocess_text(story)
        if story and key and key not in seen:
            seen.add(key)
            normalized_requirements.append(story)

    requirements = []

    for index, requirement in enumerate(
        normalized_requirements,
        start=1
    ):
        req_id = f"REQ-{index:03d}"
        requirements.append(
            analyze_requirement(
                requirement,
                req_id
            )
        )

    requirements = update_relationships(requirements)

    # Step 8: automatically generate test cases for every requirement.
    all_test_cases = generate_all_test_cases(requirements)

    extension = os.path.splitext(filename)[1].lower().lstrip(".")

    session["requirements"] = requirements
    session["test_cases"] = all_test_cases
    session["filename"] = filename
    session["input_metadata"] = {
        "filename": filename,
        "format": extension.upper(),
        "character_count": len(text),
        "user_story_count": len(user_story_requirements),
        "requirement_count": len(requirements),
    }
    session.modified = True

    return redirect(url_for("results"))


# ============================================================
# RESULTS
# ============================================================

@app.route("/results")
def results():

    requirements = session.get(
        "requirements",
        []
    )

    summary = create_summary(
        requirements
    )

    return render_template(
        "results.html",
        requirements=requirements,
        summary=summary
    )


# ============================================================
# TRACEABILITY
# ============================================================

@app.route("/traceability")
def traceability():

    requirements = session.get(
        "requirements",
        []
    )

    # THIS FIXES:
    # jinja2.exceptions.UndefinedError:
    # 'summary' is undefined

    summary = create_summary(
        requirements
    )

    traceability_data = []

    for req in requirements:

        traceability_data.append({

            "requirement_id":
                req.get(
                    "requirement_id",
                    "REQ-000"
                ),

            "requirement":
                req.get(
                    "requirement",
                    ""
                ),

            "requirement_type":
                req.get(
                    "requirement_type",
                    req.get(
                        "type",
                        "Functional"
                    )
                ),

            "quality_score":
                req.get(
                    "quality_score",
                    req.get(
                        "quality",
                        0
                    )
                ),

            "risk":
                req.get(
                    "risk_level",
                    req.get(
                        "risk",
                        "Low"
                    )
                ),

            "priority":
                req.get(
                    "priority",
                    "Medium"
                ),

            "security":
                req.get(
                    "security",
                    False
                ),

            "performance":
                req.get(
                    "performance",
                    False
                ),

            "ambiguity":
                req.get(
                    "ambiguous",
                    False
                ),

            "contradiction":
                req.get(
                    "contradiction",
                    False
                ),

            "duplicate":
                req.get(
                    "duplicate",
                    False
                ),

            "test_case_count":
                req.get(
                    "test_case_count",
                    0
                ),

            "improved_requirement":
                req.get(
                    "improved_requirement",
                    req.get(
                        "requirement",
                        ""
                    )
                )
        })

    return render_template(
        "traceability.html",

        requirements=requirements,

        traceability_data=traceability_data,

        summary=summary
    )


# ============================================================
# TEST CASES
# ============================================================

@app.route("/test-cases")
def test_cases():

    requirements = session.get(
        "requirements",
        []
    )

    # Generate if missing
    if requirements:

        all_test_cases = generate_all_test_cases(
            requirements
        )

        session["requirements"] = requirements

        session["test_cases"] = all_test_cases

    else:

        all_test_cases = []

    summary = create_summary(
        requirements
    )

    return render_template(
        "test_cases.html",
        requirements=requirements,
        test_cases=all_test_cases,
        summary=summary
    )


# ============================================================
# REANALYZE
# ============================================================

@app.route(
    "/reanalyze",
    methods=["POST"]
)
def reanalyze():

    requirement_id = request.form.get(
        "requirement_id"
    )

    new_text = request.form.get(
        "requirement",
        ""
    ).strip()

    requirements = session.get(
        "requirements",
        []
    )

    if not new_text:

        return redirect(
            url_for("results")
        )

    for index, req in enumerate(
        requirements
    ):

        if req.get(
            "requirement_id"
        ) == requirement_id:

            requirements[index] = analyze_requirement(
                new_text,
                requirement_id
            )

            break

    requirements = update_relationships(
        requirements
    )

    generate_all_test_cases(
        requirements
    )

    session["requirements"] = requirements

    session["test_cases"] = [
        case
        for req in requirements
        for case in req.get(
            "test_cases",
            []
        )
    ]

    session.modified = True

    return redirect(
        url_for("results")
    )


# ============================================================
# MODEL COMPARISON
# ============================================================

@app.route("/model-comparison")
def model_comparison():

    models = [

        {
            "name": "BERT",
            "type": "Deep Learning",
            "accuracy": 100,
            "status": (
                "Loaded"
                if bert_model
                else "Unavailable"
            )
        },

        {
            "name": "Logistic Regression",
            "type": "Machine Learning",
            "accuracy": 99,
            "status": (
                "Loaded"
                if logistic_model
                else "Unavailable"
            )
        },

        {
            "name": "SVM",
            "type": "Machine Learning",
            "accuracy": 99,
            "status": (
                "Loaded"
                if svm_model
                else "Unavailable"
            )
        },

        {
            "name": "Random Forest",
            "type": "Machine Learning",
            "accuracy": 99,
            "status": (
                "Loaded"
                if random_forest_model
                else "Unavailable"
            )
        }
    ]

    return render_template(
        "model_comparison.html",
        models=models
    )


# ============================================================
# CSV EXPORT
# ============================================================

@app.route("/export/csv")
def export_csv():

    requirements = session.get(
        "requirements",
        []
    )

    output = io.StringIO()

    writer = csv.writer(
        output
    )

    writer.writerow([
        "Requirement ID",
        "Requirement",
        "Type",
        "Confidence",
        "Quality Score",
        "Completeness",
        "Risk",
        "Priority",
        "Security",
        "Performance",
        "Ambiguous",
        "Contradiction",
        "Duplicate",
        "Test Cases",
        "Improved Requirement"
    ])

    for r in requirements:

        writer.writerow([

            r.get("requirement_id"),

            r.get("requirement"),

            r.get(
                "requirement_type",
                r.get("type")
            ),

            r.get("confidence"),

            r.get(
                "quality_score",
                r.get("quality")
            ),

            r.get("completeness"),

            r.get(
                "risk_level",
                r.get("risk")
            ),

            r.get("priority"),

            "Yes"
            if r.get("security")
            else "No",

            "Yes"
            if r.get("performance")
            else "No",

            "Yes"
            if r.get("ambiguous")
            else "No",

            "Yes"
            if r.get("contradiction")
            else "No",

            "Yes"
            if r.get("duplicate")
            else "No",

            r.get(
                "test_case_count",
                0
            ),

            r.get(
                "improved_requirement"
            )
        ])

    output.seek(0)

    return send_file(
        io.BytesIO(
            output.getvalue().encode(
                "utf-8-sig"
            )
        ),
        mimetype="text/csv",
        as_attachment=True,
        download_name="IntelliReq_Report.csv"
    )


# ============================================================
# EXCEL EXPORT
# ============================================================

@app.route("/export/excel")
def export_excel():

    requirements = session.get(
        "requirements",
        []
    )

    rows = []

    for r in requirements:

        rows.append({

            "Requirement ID":
                r.get("requirement_id"),

            "Requirement":
                r.get("requirement"),

            "Type":
                r.get(
                    "requirement_type",
                    r.get("type")
                ),

            "Confidence":
                r.get("confidence"),

            "Quality Score":
                r.get(
                    "quality_score",
                    r.get("quality")
                ),

            "Completeness":
                r.get("completeness"),

            "Risk":
                r.get(
                    "risk_level",
                    r.get("risk")
                ),

            "Priority":
                r.get("priority"),

            "Security":
                r.get("security"),

            "Performance":
                r.get("performance"),

            "Ambiguous":
                r.get("ambiguous"),

            "Contradiction":
                r.get("contradiction"),

            "Duplicate":
                r.get("duplicate"),

            "Test Cases":
                r.get("test_case_count", 0),

            "Improved Requirement":
                r.get(
                    "improved_requirement"
                )
        })

    dataframe = pd.DataFrame(
        rows
    )

    output = io.BytesIO()

    with pd.ExcelWriter(
        output,
        engine="openpyxl"
    ) as writer:

        dataframe.to_excel(
            writer,
            index=False,
            sheet_name="Requirements"
        )

    output.seek(0)

    return send_file(
        output,
        mimetype=(
            "application/vnd.openxmlformats-"
            "officedocument.spreadsheetml.sheet"
        ),
        as_attachment=True,
        download_name="IntelliReq_Report.xlsx"
    )


# ============================================================
# PDF REPORT
# ============================================================

@app.route("/download_report")
def download_report():

    requirements = session.get(
        "requirements",
        []
    )

    summary = create_summary(
        requirements
    )

    try:

        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import (
            SimpleDocTemplate,
            Paragraph,
            Spacer,
            Table,
            TableStyle
        )
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.enums import TA_CENTER

        output = io.BytesIO()

        document = SimpleDocTemplate(
            output,
            pagesize=A4
        )

        styles = getSampleStyleSheet()

        title_style = styles["Title"]

        title_style.alignment = TA_CENTER

        elements = []

        elements.append(
            Paragraph(
                "IntelliReq - Requirements Analysis Report",
                title_style
            )
        )

        elements.append(
            Spacer(1, 20)
        )

        elements.append(
            Paragraph(
                f"Total Requirements: {summary['total']}",
                styles["Normal"]
            )
        )

        elements.append(
            Paragraph(
                f"Average Quality: {summary['average_quality']}%",
                styles["Normal"]
            )
        )

        elements.append(
            Paragraph(
                f"High Risk: {summary['high_risk']}",
                styles["Normal"]
            )
        )

        elements.append(
            Paragraph(
                f"Contradictions: {summary['contradictions']}",
                styles["Normal"]
            )
        )

        elements.append(
            Paragraph(
                f"Duplicates: {summary['duplicates']}",
                styles["Normal"]
            )
        )

        elements.append(
            Spacer(1, 20)
        )

        data = [[
            "ID",
            "Type",
            "Quality",
            "Risk",
            "Priority"
        ]]

        for r in requirements:

            data.append([

                r.get("requirement_id"),

                r.get(
                    "requirement_type",
                    r.get("type")
                ),

                f"{r.get('quality_score', 0)}%",

                r.get(
                    "risk_level",
                    r.get("risk")
                ),

                r.get("priority")
            ])

        table = Table(
            data,
            repeatRows=1
        )

        table.setStyle(
            TableStyle([
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.grey
                ),
                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, 0),
                    colors.white
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.black
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "TOP"
                )
            ])
        )

        elements.append(table)

        document.build(elements)

        output.seek(0)

        return send_file(
            output,
            mimetype="application/pdf",
            as_attachment=True,
            download_name="IntelliReq_Report.pdf"
        )

    except Exception as e:

        return f"PDF generation error: {e}", 500


# ============================================================
# API SUMMARY
# ============================================================

@app.route("/api/summary")
def api_summary():

    requirements = session.get(
        "requirements",
        []
    )

    return jsonify(
        create_summary(
            requirements
        )
    )


# ============================================================
# API REQUIREMENTS
# ============================================================

@app.route("/api/requirements")
def api_requirements():

    return jsonify(
        session.get(
            "requirements",
            []
        )
    )


# ============================================================
# API MODELS
# ============================================================

@app.route("/api/models")
def api_models():

    return jsonify({

        "BERT": bool(
            bert_model
        ),

        "Logistic Regression": bool(
            logistic_model
            and logistic_vectorizer
        ),

        "SVM": bool(
            svm_model
            and svm_vectorizer
        ),

        "Random Forest": bool(
            random_forest_model
            and random_forest_vectorizer
        )
    })


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/health")
def health():

    return jsonify({

        "status": "healthy",

        "bert_loaded": bool(
            bert_model
        ),

        "requirements":
            len(
                session.get(
                    "requirements",
                    []
                )
            )
    })


# ============================================================
# CLEAR SESSION
# ============================================================

@app.route("/clear")
def clear():

    session.clear()

    return redirect(
        url_for("index")
    )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )