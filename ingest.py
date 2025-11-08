# backend/ingest.py
import os
import json
import pdfplumber
from pdf2image import convert_from_path
import pytesseract
import pandas as pd
import camelot

def extract_pages(pdf_path):
    """Return list of page dicts: {page_number, text, tables: [DataFrame], images_text (OCR if any)}"""
    pages = []
    with pdfplumber.open(r"C:\Users\HP\RAG_Chatboat\data\bajaj_finserv_factsheet_Oct.pdf") as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            tables = []
            # try camelot first
            try:
                tables_raw = camelot.read_pdf(pdf_path, pages=str(i))
                for t in tables_raw:
                    df = t.df
                    if df.shape[0] > 1:
                        # heuristically set first row as header
                        header = df.iloc[0].tolist()
                        data = df.iloc[1:].copy()
                        data.columns = header
                        tables.append(data)
            except Exception:
                # fallback to pdfplumber
                try:
                    for tbl in page.extract_tables():
                        if not tbl:
                            continue
                        df = pd.DataFrame(tbl[1:], columns=tbl[0]) if len(tbl) > 1 else pd.DataFrame(tbl)
                        tables.append(df)
                except Exception:
                    pass

            images_text = ""
            try:
                pil_pages = convert_from_path(pdf_path, first_page=i, last_page=i)
                if pil_pages:
                    images_text = pytesseract.image_to_string(pil_pages[0])
            except Exception:
                images_text = ""

            pages.append({
                "page_number": i,
                "text": text,
                "tables": [t.to_dict(orient='records') if hasattr(t, 'to_dict') else t for t in tables],
                "images_text": images_text,
            })
    return pages

def pages_to_snippets(pages, max_chars=800):
    snippets = []
    for p in pages:
        page_num = p['page_number']
        text = (p.get('text') or '').strip()
        if not text and p.get('images_text'):
            text = p.get('images_text')
        paras = [s.strip() for s in text.split('\n') if s.strip()]
        buffer = ''
        for para in paras:
            if len(buffer) + len(para) + 1 > max_chars:
                if buffer:
                    snippets.append({'page': page_num, 'text': buffer})
                buffer = para
            else:
                buffer = (buffer + ' ' + para).strip()
        if buffer:
            snippets.append({'page': page_num, 'text': buffer})
        # add tables as CSV snippets
        for tbl in p.get('tables', []):
            try:
                df = pd.DataFrame(tbl)
                csv = df.to_csv(index=False)
                snippets.append({'page': page_num, 'text': csv, 'type':'table'})
            except Exception:
                snippets.append({'page': page_num, 'text': str(tbl), 'type':'table'})
    return snippets

def save_snippets(snippets, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'snippets.json'), 'w', encoding='utf-8') as f:
        json.dump(snippets, f, ensure_ascii=False, indent=2)
