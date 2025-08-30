import fitz  # PyMuPDF

def pdf_to_text(pdf_path, txt_path):
    doc = fitz.open(pdf_path)
    text = ""

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"✅ Converted '{pdf_path}' to '{txt_path}' successfully.")

if __name__ == "__main__":
    pdf_path = r"D:\Voice_RAG\rag\PG-Manual.pdf"   
    txt_path = r"D:\Voice_RAG\rag\pg_manual.txt"  

    pdf_to_text(pdf_path, txt_path)
