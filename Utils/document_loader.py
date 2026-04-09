import pdfplumber
import docx

def extract_text_with_pages(uploaded_file):

    file_name = uploaded_file.name.lower()

    # --------------------
    # PDF FILE (with pages)
    # --------------------
    if file_name.endswith(".pdf"):

        pages = []

        with pdfplumber.open(uploaded_file) as pdf:

            for i, page in enumerate(pdf.pages):

                text = page.extract_text()

                if text:
                    pages.append({
                        "text": text,
                        "page": i + 1
                    })

        return pages


    # --------------------
    # DOCX FILE
    # --------------------
    elif file_name.endswith(".docx"):

        doc = docx.Document(uploaded_file)

        text = "\n".join([para.text for para in doc.paragraphs])

        return [{
            "text": text,
            "page": "N/A"
        }]


    # --------------------
    # TXT FILE
    # --------------------
    elif file_name.endswith(".txt"):

        text = uploaded_file.read().decode("utf-8")

        return [{
            "text": text,
            "page": "N/A"
        }]

    else:
        return []