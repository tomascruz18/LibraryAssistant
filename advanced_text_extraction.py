from pathlib import Path
import subprocess
import fitz
import re


def extract_text(pdf_path):

    doc = fitz.open(pdf_path)

    text = ""

    for page in doc:
        text += page.get_text()

    return text

def clean_text(text):
    # Merge broken words split by line breaks
    text = re.sub(r"-\n", "", text)

    # Replace newlines with spaces
    text = text.replace("\n", " ")

    # Collapse repeated whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def pdf_to_text(pdf_path, method=None):

    text = extract_text(pdf_path)

    # If very little text exists,
    # assume the PDF is scanned.

    if len(text.strip()) < 100 or method == "OCR":

        print("Running OCR...")

        output = Path(pdf_path).with_stem(Path(pdf_path).stem + "_ocr")

        result = subprocess.run(
            [
                "ocrmypdf",
                "--skip-text",
                "--deskew",
                "--rotate-pages",
                pdf_path,
                str(output),
            ],
            capture_output=True,
            text=True,
        )

        # If OCRmyPDF reported an error but still produced the file,
        # continue anyway.
        if result.returncode != 0:
            if output.exists():
                print("OCR completed with warnings.")
            else:
                raise RuntimeError(result.stderr)

        text = extract_text(output)

    return clean_text(text)
pdf_1 = r"C:\Users\tr-mo\Zotero\storage\FNRZ3I4J\Allamaprabhu et al. - 2011 - Improved Prediction of Flow Separation in Thrust Optimized Parabolic Nozzles with FLUENT.pdf"
pdf_2 = r"C:\Users\tr-mo\Zotero\storage\FKTLDIEH\19780016336.pdf"

text = pdf_to_text(pdf_1)
print(text[:3000])

text = pdf_to_text(pdf_2, method="OCR")
print(text[12000:17000])