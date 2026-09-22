# download_news.py
import os
import requests
import pandas as pd


def get_sec_cik(ticker_symbol):
    url = "https://www.sec.gov/files/company_tickers.json"
    headers = {"User-Agent": "GlobalQuantFund research_analytics_2026@outlook.com"}
    try:
        res = requests.get(url, headers=headers, timeout=10)
        if res.status_code == 200:
            data = res.json()
            for item in data.values():
                if item["ticker"].upper() == ticker_symbol.upper():
                    return str(item["cik_str"]).zfill(10), item["title"]
    except Exception as e:
        print(f"[WARN] 获取 SEC CIK 失败: {e}")
    return None, ticker_symbol


def fetch_and_calculate_dynamic_metrics(cik):
    """
    聚合 SEC Revenues 科目，按会计截止日提取顶层合并总营收，并动态计算 QoQ/YoY
    """
    headers = {"User-Agent": "GlobalQuantFund research_analytics_2026@outlook.com"}
    concept_tags = [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet"
    ]

    all_raw_units = []
    for tag in concept_tags:
        url = f"https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/us-gaap/{tag}.json"
        try:
            res = requests.get(url, headers=headers, timeout=10)
            if res.status_code == 200:
                data = res.json()
                units = data.get("units", {}).get("USD", [])
                if units:
                    all_raw_units.extend(units)
        except Exception:
            continue

    if not all_raw_units:
        print("[WARN] SEC 接口未返回营业收入科目。")
        return []

    raw_records = []
    for item in all_raw_units:
        form = item.get("form")
        fp = item.get("fp", "")
        fy = item.get("fy")
        val = item.get("val")
        filed = item.get("filed")
        start = item.get("start")
        end = item.get("end")

        if form in ["10-Q", "10-K"] and filed and val and start and end:
            raw_records.append({
                "filed": filed,
                "start": start,
                "end": end,
                "val": val,
                "form": form,
                "fp": fp,
                "fy": fy
            })

    df = pd.DataFrame(raw_records)
    if df.empty:
        return []

    df['start'] = pd.to_datetime(df['start'])
    df['end'] = pd.to_datetime(df['end'])
    df['filed'] = pd.to_datetime(df['filed'])
    df['days'] = (df['end'] - df['start']).dt.days
    df['val_b'] = df['val'] / 1e9

    results = []

    # 1. 单季处理 (10-Q): 限制 70~115 天
    df_q = df[(df['form'] == '10-Q') & (df['days'] >= 70) & (df['days'] <= 115)]
    if not df_q.empty:
        idx_max = df_q.groupby('end')['val_b'].idxmax()
        df_q = df_q.loc[idx_max].sort_values('end').reset_index(drop=True)

        for i, row in df_q.iterrows():
            val_b = row['val_b']
            fy = row['fy']
            fp = row['fp']
            filed_dt = row['filed']

            qoq_str = ""
            if i > 0:
                prev_val = df_q.iloc[i - 1]['val_b']
                if prev_val > 0:
                    qoq = (val_b - prev_val) / prev_val * 100
                    qoq_str = f"，单季环比增长: {qoq:+.1f}%"

            desc = f"{fy} {fp} 单季总营收: {val_b:.2f} 亿美元{qoq_str}"
            results.append({"filed_dt": filed_dt, "desc": desc})

    # 2. 全年处理 (10-K): 限制 >300 天
    df_a = df[(df['form'] == '10-K') & (df['days'] > 300)]
    if not df_a.empty:
        idx_max_a = df_a.groupby('end')['val_b'].idxmax()
        df_a = df_a.loc[idx_max_a].sort_values('end').reset_index(drop=True)

        for i, row in df_a.iterrows():
            val_b = row['val_b']
            fy = row['fy']
            filed_dt = row['filed']

            yoy_str = ""
            if i > 0:
                prev_val = df_a.iloc[i - 1]['val_b']
                if prev_val > 0:
                    yoy = (val_b - prev_val) / prev_val * 100
                    yoy_str = f"，全财年同比增长: {yoy:+.1f}%"

            desc = f"{fy} 全财年累计总营收: {val_b:.2f} 亿美元{yoy_str}"
            results.append({"filed_dt": filed_dt, "desc": desc})

    print(f"[INFO] 成功自 SEC 动态解析到 {len(results)} 个历史财期的顶层合并财务指标。")
    return results


def decode_sec_items(items_str):
    item_map = {
        "2.02": "季度业绩与财务状况发布",
        "1.01": "重大商业协议签署",
        "5.02": "董事与核心高管任免变动",
        "5.07": "股东大会表决结果公布",
        "7.01": "Reg FD 公开业务说明会",
        "8.01": "重要监管与业务事项披露 (含出口管制政策)"
    }
    if not items_str or not isinstance(items_str, str):
        return "公司常规运营事项"
    decoded = []
    for code, desc in item_map.items():
        if code in items_str:
            decoded.append(desc)
    return " | ".join(decoded) if decoded else f"事项代码: {items_str}"


def download_dynamic_news(ticker="105.NVDA", start_date="2023-01-01", end_date="2024-04-01"):
    symbol = ticker.split('.')[-1] if '.' in ticker else ticker
    print("=" * 50)
    print(f"[START] 动态生成标的 [{symbol}] 量化基本面档案 [{start_date} 至 {end_date}]")
    print("=" * 50)

    cik, company_name = get_sec_cik(symbol)
    if not cik:
        print(f"[ERROR] 无法获取 [{symbol}] 的 CIK 代码。")
        return

    fin_records = fetch_and_calculate_dynamic_metrics(cik)

    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    headers = {"User-Agent": "GlobalQuantFund research_analytics_2026@outlook.com"}

    try:
        res = requests.get(url, headers=headers, timeout=15)
        if res.status_code != 200:
            print(f"[ERROR] SEC 接口异常: {res.status_code}")
            return

        data = res.json()
        recent = data.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        filing_dates = recent.get("filingDate", [])
        primary_docs = recent.get("primaryDocDescription", [])
        items_list = recent.get("items", [])

        date_aggregated = {}
        target_forms = {"8-K", "10-Q", "10-K"}
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        for i in range(len(forms)):
            form_type = forms[i]
            if form_type in target_forms:
                f_date_str = filing_dates[i]
                f_date = pd.to_datetime(f_date_str)

                if start_dt <= f_date <= end_dt:
                    items_raw = items_list[i] if i < len(items_list) else ""
                    decoded_desc = decode_sec_items(items_raw)
                    doc_desc = primary_docs[i] if i < len(primary_docs) and primary_docs[i] else form_type

                    if f_date_str not in date_aggregated:
                        date_aggregated[f_date_str] = {
                            "dt": f_date,
                            "forms": set(),
                            "items": set(),
                            "doc_desc": doc_desc
                        }
                    date_aggregated[f_date_str]["forms"].add(form_type)
                    if decoded_desc != "公司常规运营事项":
                        date_aggregated[f_date_str]["items"].add(decoded_desc)

        records = []
        for f_date_str, info in date_aggregated.items():
            f_dt = info["dt"]
            forms_str = "/".join(sorted(info["forms"]))
            items_summary = " | ".join(info["items"]) if info["items"] else "定期财务报告归档"

            # 严格消除未来信息：财务指标归档日期必须早于或等于当前事件日期 (0 <= (f_dt - fin_filed_dt).days <= 7)
            matched_fin_text = ""
            for fin in fin_records:
                diff_days = (f_dt - fin["filed_dt"]).days
                if 0 <= diff_days <= 7:
                    matched_fin_text = f"【量化财务指标: {fin['desc']}】"
                    break

            headline = f"{company_name} 官方披露 (Form {forms_str}: {items_summary})"
            summary = f"SEC官方归档: {items_summary}。{matched_fin_text} 详细文档: {info['doc_desc']}"

            records.append({
                "datetime": f_date_str,
                "headline": headline,
                "summary": summary
            })

        df = pd.DataFrame(records)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values(by='datetime').reset_index(drop=True)

        os.makedirs("data", exist_ok=True)
        save_path = f"data/{symbol}_news.csv"
        df.to_csv(save_path, index=False)

        print(f"[SUCCESS] 动态量化基本面档案已生成: {save_path}，有效事件数: {len(df)} 条")

    except Exception as e:
        print(f"[ERROR] 执行失败: {e}")


if __name__ == "__main__":
    download_dynamic_news(ticker="105.NVDA", start_date="2023-01-01", end_date="2024-04-01")