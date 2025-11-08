# backend/answer_engine.py
"""
Rule-based answer engine that uses retrieved snippets and deterministic parsing to answer queries WITHOUT any external LLM or API.
Supports:
- Top N holdings of <Fund>
- X-year return of <Fund>
- Compare equity vs debt allocation
- AUM month-over-month change
- Which equity fund has highest 3-year return
- YTM / Macaulay / Avg Maturity for Money Market Fund
Returns: (answer_text, citations_list)
"""

import re
import pandas as pd
from backend.utils import cagr, parse_percent, pct_change, avg_return_from_list, compute_macaulay_duration

def find_tables(snippets):
    tables = []
    for s in snippets:
        is_table = s.get('type') == 'table' or (',' in s['text'] and '\n' in s['text'])
        if is_table:
            tables.append(s)
    return tables

def parse_csv_snippet(snippet_text):
    try:
        from io import StringIO
        df = pd.read_csv(StringIO(snippet_text))
        return df
    except Exception:
        return None

def extract_numeric_from_text(text):
    nums = re.findall(r"[0-9]+(?:\.[0-9]+)?%|[0-9,]+(?:\.[0-9]+)?", text)
    return nums

def answer_question(query, retrieved_snippets):
    q = query.lower()
    citations = []

    # Top N holdings
    m = re.search(r"top\s*(\d+)\s*holdings of (.+)", q)
    if m:
        n = int(m.group(1))
        fund = m.group(2).strip()
        tables = find_tables(retrieved_snippets)
        for t in tables:
            df = parse_csv_snippet(t['text'])
            if df is None:
                continue
            cols = [c.lower() for c in df.columns]
            if any('hold' in c for c in cols) and any('weight' in c or '%' in c or 'allocation' in c for c in cols):
                weight_col = [c for c in df.columns if 'weight' in c.lower() or '%' in c or 'allocation' in c]
                holding_col = [c for c in df.columns if 'hold' in c.lower() or 'name' in c.lower() or 'scheme' in c.lower()]
                if weight_col and holding_col:
                    wc = weight_col[0]; hc = holding_col[0]
                    try:
                        df[wc+'_clean'] = df[wc].astype(str).str.replace('%','').str.replace(',','').astype(float)
                    except Exception:
                        continue
                    topn = df.sort_values(by=wc+'_clean', ascending=False).head(n)
                    rows = topn[[hc, wc]].to_dict(orient='records')
                    citations.append({'page': t['page'], 'snippet': t['text'][:400]})
                    ans = f"Top {n} holdings of {fund}: " + ", ".join([f\"{r[hc]} ({r[wc]})\" for r in rows])
                    return ans, citations
        return "Not stated in the document.", []

    # X-year return of fund
    m = re.search(r"(\d+)[- ]*year return of (.+)", q)
    if m:
        years = int(m.group(1)); fund = m.group(2).strip()
        for s in retrieved_snippets:
            txt = s['text'].lower()
            if fund in txt and (f"{years}-year" in txt or f"{years} year" in txt or f"{years} yr" in txt):
                nums = extract_numeric_from_text(txt)
                pct = [n for n in nums if n.endswith('%')]
                if pct:
                    citations.append({'page': s['page'], 'snippet': txt[:400]})
                    return f"{years}-year return of {fund} is {pct[0]}", citations
        tables = find_tables(retrieved_snippets)
        for t in tables:
            df = parse_csv_snippet(t['text'])
            if df is None: continue
            for col in df.columns:
                try:
                    matches = df[df[col].astype(str).str.lower().str.contains(fund)]
                    if not matches.empty:
                        target_cols = [c for c in df.columns if str(years) in c or ('year' in c.lower() and str(years) in c)]
                        if not target_cols:
                            target_cols = [c for c in df.columns if '3' in c and ('year' in c.lower() or 'yr' in c.lower())]
                        if target_cols:
                            val = matches.iloc[0][target_cols[0]]
                            citations.append({'page': t['page'], 'snippet': t['text'][:400]})
                            return f"{years}-year return of {fund} is {val}", citations
                except Exception:
                    continue
        return "Not stated in the document.", []

    # Compare equity vs debt allocation
    if 'compare' in q and 'equity' in q and 'debt' in q:
        tables = find_tables(retrieved_snippets)
        for t in tables:
            txt = t['text'].lower()
            if 'equity' in txt and 'debt' in txt:
                df = parse_csv_snippet(t['text'])
                if df is None:
                    citations.append({'page': t['page'], 'snippet': t['text'][:400]})
                    return f"Allocation table found on page {t['page']}. Please inspect the snippet for equity vs debt numbers.", citations
                for idx, row in df.iterrows():
                    rowtext = ' '.join([str(x).lower() for x in row.values])
                    if 'equity' in rowtext and 'debt' in rowtext:
                        citations.append({'page': t['page'], 'snippet': t['text'][:400]})
                        return f"Allocation row containing both Equity and Debt found on page {t['page']}. Row: {row.to_dict()}", citations
                cols = [c.lower() for c in df.columns]
                if 'equity' in cols and 'debt' in cols:
                    citations.append({'page': t['page'], 'snippet': t['text'][:400]})
                    return f"Equity column and Debt column values found on page {t['page']}. Inspect snippet.", citations
        return "Not stated in the document.", []

    # AUM change vs last month
    if 'aum' in q and ('last month' in q or 'previous month' in q or 'month over month' in q or 'mom' in q):
        for s in retrieved_snippets:
            txt = s['text'].lower()
            if 'aum' in txt and ('month' in txt or 'oct' in txt or 'sep' in txt or 'month-on-month' in txt or 'm-o-m' in txt):
                nums = extract_numeric_from_text(txt)
                if len(nums) >= 2:
                    a1 = nums[0]; a2 = nums[1]
                    citations.append({'page': s['page'], 'snippet': txt[:400]})
                    return f"AUM change found: {a1} vs {a2} (see page {s['page']})", citations
        return "Not stated in the document.", []

    # Which equity fund has highest 3-year return
    if ('which' in q or 'which of' in q) and 'highest' in q and ('3-year' in q or '3 year' in q):
        tables = find_tables(retrieved_snippets)
        for t in tables:
            df = parse_csv_snippet(t['text'])
            if df is None: continue
            cols = [c.lower() for c in df.columns]
            target_cols = [c for c in df.columns if '3' in c and ('year' in c.lower() or 'yr' in c.lower())]
            name_cols = [c for c in df.columns if any(x in c.lower() for x in ['fund','scheme','name'])]
            if target_cols and name_cols:
                tc = target_cols[0]; nc = name_cols[0]
                try:
                    df[tc+'_clean'] = df[tc].astype(str).str.replace('%','').str.replace(',','').astype(float)
                    row = df.loc[df[tc+'_clean'].idxmax()]
                    return f\"{row[nc]} has the highest 3-year return: {row[tc]} [Pg {t['page']}]\", [{'page': t['page'], 'snippet': t['text'][:400]}]
                except Exception:
                    continue
        return "Not stated in the document.", []

    # YTM / Macaulay / Avg maturity for money market
    if 'ytm' in q or 'macaulay' in q or 'average maturity' in q or 'avg maturity' in q:
        tables = find_tables(retrieved_snippets)
        for t in tables:
            txt = t['text'].lower()
            if 'money market' in txt or 'moneymarket' in txt or 'money-market' in txt:
                ytm = None; mac = None; avgm = None
                if 'ytm' in txt:
                    nums = extract_numeric_from_text(txt)
                    if nums: ytm = nums[0]
                m = re.search(r'macaulay[^0-9%\n]*([0-9,.]+)', txt)
                if m: mac = m.group(1)
                m2 = re.search(r'(average maturity|avg maturity)[^0-9%\n]*([0-9,.]+)', txt)
                if m2: avgm = m2.group(2)
                if any([ytm, mac, avgm]):
                    citations.append({'page': t['page'], 'snippet': t['text'][:400]})
                    parts = []
                    if ytm: parts.append(f"YTM: {ytm}")
                    if mac: parts.append(f"Macaulay Duration: {mac}")
                    if avgm: parts.append(f"Average Maturity: {avgm}")
                    return '; '.join(parts), citations
        return "Not stated in the document.", []

    # Fallback: return top retrieved snippets
    citations = [{'page': s['snippet']['page'], 'snippet': s['snippet']['text'][:400]} for s in retrieved_snippets]
    return ("I couldn't find a deterministic answer from parsed tables/snippets. Here are the top retrieved snippets for your question.", citations)
