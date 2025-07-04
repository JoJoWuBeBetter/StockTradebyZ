from __future__ import annotations
import argparse
import datetime as dt
import json
import logging
import random
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional
import akshare as ak
import pandas as pd
import tushare as ts
from mootdx.quotes import Quotes
from tqdm import tqdm

# 导入限流库
from ratelimit import limits, sleep_and_retry

warnings.filterwarnings("ignore")
# --------------------------- 全局日志配置 --------------------------- #
LOG_FILE = Path("fetch.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger("fetch_mktcap")
# 屏蔽第三方库多余 INFO 日志
for noisy in ("httpx", "urllib3", "_client", "akshare"):
    logging.getLogger(noisy).setLevel(logging.WARNING)


# --------------------------- 市值快照 --------------------------- #
def _get_mktcap_ak() -> pd.DataFrame:
    """实时快照，返回列：code, mktcap（单位：元）"""
    for attempt in range(1, 4):
        try:
            df = ak.stock_zh_a_spot_em()
            break
        except Exception as e:
            logger.warning("AKShare 获取市值快照失败(%d/3): %s", attempt, e)
            time.sleep(backoff := random.uniform(1, 3) * attempt)
    else:
        raise RuntimeError("AKShare 连续三次拉取市值快照失败！")
    df = df[["代码", "总市值"]].rename(columns={"代码": "code", "总市值": "mktcap"})
    df["mktcap"] = pd.to_numeric(df["mktcap"], errors="coerce")
    return df


# --------------------------- 股票池筛选 --------------------------- #
def get_constituents(
        min_cap: float,
        max_cap: float,
        small_player: bool,
        mktcap_df: Optional[pd.DataFrame] = None,
) -> List[str]:
    df = mktcap_df if mktcap_df is not None else _get_mktcap_ak()
    cond = (df["mktcap"] >= min_cap) & (df["mktcap"] <= max_cap)
    if small_player:
        cond &= ~df["code"].str.startswith(("300", "301", "688", "8", "4"))
    codes = df.loc[cond, "code"].str.zfill(6).tolist()
    # 附加股票池 appendix.json
    try:
        with open("appendix.json", "r", encoding="utf-8") as f:
            appendix_codes = json.load(f)["data"]
    except FileNotFoundError:
        appendix_codes = []
    codes = list(dict.fromkeys(appendix_codes + codes))  # 去重保持顺序
    logger.info("筛选得到 %d 只股票", len(codes))
    return codes


# --------------------------- 历史 K 线抓取 --------------------------- #
COLUMN_MAP_HIST_AK = {
    "日期": "date",
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
    "成交额": "amount",
    "换手率": "turnover",
}
_FREQ_MAP = {
    0: "5m",
    1: "15m",
    2: "30m",
    3: "1h",
    4: "day",
    5: "week",
    6: "mon",
    7: "1m",
    8: "1m",
    9: "day",
    10: "3mon",
    11: "year",
}

# ---------- Tushare 工具函数 ---------- #
# Tushare 免费用户每分钟 200 次调用限制
# 为了保险起见，我们设置一个略低的阈值，比如每 60 秒 180 次调用
TS_CALLS_PER_MINUTE = 180
TS_ONE_MINUTE = 60


# 使用 @sleep_and_retry 装饰器，当达到限制时会自动暂停，并在一段时间后重试
# @limits(calls=TS_CALLS_PER_MINUTE, period=TS_ONE_MINUTE) 装饰器定义了限流规则
@sleep_and_retry
@limits(calls=TS_CALLS_PER_MINUTE, period=TS_ONE_MINUTE)
def _tushare_pro_bar_limited(**kwargs):
    """
    一个包装 ts.pro_bar 的函数，并应用了限流。
    所有的并发调用都会共享这个限流规则。
    """
    return ts.pro_bar(**kwargs)


def _to_ts_code(code: str) -> str:
    return f"{code.zfill(6)}.SH" if code.startswith(("60", "68", "9")) else f"{code.zfill(6)}.SZ"


def _get_kline_tushare(code: str, start: str, end: str, adjust: str) -> pd.DataFrame:
    ts_code = _to_ts_code(code)
    adj_flag = None if adjust == "" else adjust
    for attempt in range(1, 4):
        try:
            # 调用我们封装好的、带限流功能的函数
            df = _tushare_pro_bar_limited(
                ts_code=ts_code,
                adj=adj_flag,
                start_date=start,
                end_date=end,
                freq="D",
            )
            break
        except Exception as e:
            logger.warning("Tushare 拉取 %s 失败(%d/3): %s", code, attempt, e)
            # Tushare自身的限流会导致大部分异常是 HTTP 4xx/5xx，
            # 但这里是 @sleep_and_retry 在处理，所以这里更多的可能是网络错误或Tushare服务本身的问题
            time.sleep(random.uniform(1, 2) * attempt)
    else:
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.rename(columns={"trade_date": "date", "vol": "volume"})[
        ["date", "open", "close", "high", "low", "volume"]
    ].copy()
    df["date"] = pd.to_datetime(df["date"])
    df[[c for c in df.columns if c != "date"]] = df[[c for c in df.columns if c != "date"]].apply(
        pd.to_numeric, errors="coerce"
    )
    return df.sort_values("date").reset_index(drop=True)


# ---------- AKShare 工具函数 ---------- #
def _get_kline_akshare(code: str, start: str, end: str, adjust: str) -> pd.DataFrame:
    for attempt in range(1, 4):
        try:
            df = ak.stock_zh_a_hist(
                symbol=code,
                period="daily",
                start_date=start,
                end_date=end,
                adjust=adjust,
            )
            break
        except Exception as e:
            logger.warning("AKShare 拉取 %s 失败(%d/3): %s", code, attempt, e)
            time.sleep(random.uniform(1, 2) * attempt)
    else:
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    df = (
        df[list(COLUMN_MAP_HIST_AK)]
        .rename(columns=COLUMN_MAP_HIST_AK)
        .assign(date=lambda x: pd.to_datetime(x["date"]))
    )
    df[[c for c in df.columns if c != "date"]] = df[[c for c in df.columns if c != "date"]].apply(
        pd.to_numeric, errors="coerce"
    )
    df = df[["date", "open", "close", "high", "low", "volume"]]
    return df.sort_values("date").reset_index(drop=True)


# ---------- Mootdx 工具函数 ---------- #
def _get_kline_mootdx(code: str, start: str, end: str, adjust: str, freq_code: int) -> pd.DataFrame:
    symbol = code.zfill(6)
    freq = _FREQ_MAP.get(freq_code, "day")
    client = Quotes.factory(market="std")
    try:
        df = client.bars(symbol=symbol, frequency=freq, adjust=adjust or None)
    except Exception as e:
        logger.warning("Mootdx 拉取 %s 失败: %s", code, e)
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.rename(
        columns={"datetime": "date", "open": "open", "high": "high", "low": "low", "close": "close", "vol": "volume"}
    )
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    start_ts = pd.to_datetime(start, format="%Y%m%d")
    end_ts = pd.to_datetime(end, format="%Y%m%d")
    df = df[(df["date"].dt.date >= start_ts.date()) & (df["date"].dt.date <= end_ts.date())].copy()
    df = df.sort_values("date").reset_index(drop=True)
    return df[["date", "open", "close", "high", "low", "volume"]]


# ---------- 通用接口 ---------- #
def get_kline(
        code: str,
        start: str,
        end: str,
        adjust: str,
        datasource: str,
        freq_code: int = 4,
) -> pd.DataFrame:
    if datasource == "tushare":
        return _get_kline_tushare(code, start, end, adjust)
    elif datasource == "akshare":
        return _get_kline_akshare(code, start, end, adjust)
    elif datasource == "mootdx":
        return _get_kline_mootdx(code, start, end, adjust, freq_code)
    else:
        raise ValueError("datasource 仅支持 'tushare', 'akshare' 或 'mootdx'")


# ---------- 数据校验 ---------- #
def validate(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)
    if df["date"].isna().any():
        raise ValueError("存在缺失日期！")
    if (df["date"] > pd.Timestamp.today()).any():
        raise ValueError("数据包含未来日期，可能抓取错误！")
    return df


def drop_dup_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[:, ~df.columns.duplicated()]


# ---------- 单只股票抓取 ---------- #
def fetch_one(
        code: str,
        start: str,
        end: str,
        out_dir: Path,
        incremental: bool,
        datasource: str,
        freq_code: int,
):
    csv_path = out_dir / f"{code}.csv"
    # 增量更新：若本地已有数据则从最后一天开始
    if incremental and csv_path.exists():
        try:
            existing = pd.read_csv(csv_path, parse_dates=["date"])
            last_date = existing["date"].max()
            # 检查 `last_date` 是否为 NaT (Not a Time)
            if pd.isna(last_date):
                # 如果 `last_date` 无效，则从头下载
                logger.warning("%s CSV文件日期列存在问题或为空，将重新下载", csv_path)
            elif last_date.date() >= pd.to_datetime(end, format="%Y%m%d").date():
                logger.debug("%s 已是最新，无需更新", code)
                return
            else:
                start = (last_date + pd.Timedelta(days=1)).strftime("%Y%m%d")  # 从下一天开始更新
        except Exception:
            logger.exception("读取 %s 失败，将重新下载", csv_path)
            # 保持 start 为原始值，进行全量下载
            pass  # Already handled by starting from original `start`

    for attempt in range(1, 4):
        try:
            new_df = get_kline(code, start, end, "qfq", datasource, freq_code)
            if new_df.empty:
                logger.debug("%s 无新数据或拉取失败，跳过", code)
                break  # 如果没有数据，或者拉取失败导致空，则无需重试
            new_df = validate(new_df)

            if csv_path.exists() and incremental:
                old_df = pd.read_csv(
                    csv_path,
                    parse_dates=["date"],
                    index_col=False
                )
                old_df = drop_dup_columns(old_df)
                new_df = drop_dup_columns(new_df)
                # 合并新旧数据，去重并排序
                combined_df = (
                    pd.concat([old_df, new_df], ignore_index=True)
                    .drop_duplicates(subset="date")
                    .sort_values("date")
                    .reset_index(drop=True)
                )
                # 确保日期连续，并处理重复日期导致的数据缺失，但这里不处理
                # 如果新数据覆盖了旧数据，则新数据优先
                new_df = combined_df
            new_df.to_csv(csv_path, index=False)
            logger.debug("%s 数据已更新至 %s", code, csv_path.name)
            break
        except Exception:
            logger.exception("%s 第 %d 次抓取失败", code, attempt)
            time.sleep(random.uniform(1, 3) * attempt)  # 指数退避
    else:
        logger.error("%s 三次抓取均失败，已跳过！", code)


# ---------- 主入口 ---------- #
def main():
    parser = argparse.ArgumentParser(description="按市值筛选 A 股并抓取历史 K 线")
    parser.add_argument("--datasource", choices=["tushare", "akshare", "mootdx"], default="tushare",
                        help="历史 K 线数据源")
    parser.add_argument("--frequency", type=int, choices=list(_FREQ_MAP.keys()), default=4, help="K线频率编码，参见说明")
    parser.add_argument("--exclude-gem", type=bool, default=True,
                        help="True则排除创业板/科创板/北交所")  # argparse boolean arguments need `type=bool` and proper defaults
    parser.add_argument("--min-mktcap", type=float, default=5e9, help="最小总市值（含），单位：元")
    parser.add_argument("--max-mktcap", type=float, default=float("+inf"), help="最大总市值（含），单位：元，默认无限制")
    parser.add_argument("--start", default="20190101", help="起始日期 YYYYMMDD 或 'today'")
    parser.add_argument("--end", default="today", help="结束日期 YYYYMMDD 或 'today'")
    parser.add_argument("--out", default="./data", help="输出目录")
    parser.add_argument("--workers", type=int, default=3, help="并发线程数")
    args = parser.parse_args()
    # ---------- Token 处理 ---------- #
    if args.datasource == "tushare":
        ts_token = "81ce83803878702239464506c90e6b344eb329e01a17d0bc350b796e"  # 在这里补充你的 Tushare Token
        if ts_token == "YOUR_TUSHARE_TOKEN" or not ts_token.strip():
            logger.error("请在代码中设置 Tushare Token！")
            sys.exit(1)
        ts.set_token(ts_token)
        global pro  # 声明 pro 为全局变量
        pro = ts.pro_api()
    # ---------- 日期解析 ---------- #
    start = dt.date.today().strftime("%Y%m%d") if args.start.lower() == "today" else args.start
    end = dt.date.today().strftime("%Y%m%d") if args.end.lower() == "today" else args.end
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    # ---------- 市值快照 & 股票池 ---------- #
    mktcap_df = _get_mktcap_ak()
    codes_from_filter = get_constituents(
        args.min_mktcap,
        args.max_mktcap,
        args.exclude_gem,
        mktcap_df=mktcap_df,
    )
    # 加上本地已有的股票，确保旧数据也能更新
    local_codes = [p.stem for p in out_dir.glob("*.csv")]
    codes = sorted(set(codes_from_filter) | set(local_codes))
    if not codes:
        logger.error("筛选结果为空，请调整参数！")
        sys.exit(1)
    logger.info(
        "开始抓取 %d 支股票 | 数据源:%s | 频率:%s | 日期:%s → %s",
        len(codes),
        args.datasource,
        _FREQ_MAP[args.frequency],
        start,
        end,
    )
    # ---------- 多线程抓取 ---------- #
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                fetch_one,
                code,
                start,
                end,
                out_dir,
                True,  # incremental 设置为 True
                args.datasource,
                args.frequency,
            )
            for code in codes
        ]
        for _ in tqdm(as_completed(futures), total=len(futures), desc="下载进度"):
            pass
    logger.info("全部任务完成，数据已保存至 %s", out_dir.resolve())


if __name__ == "__main__":
    main()
