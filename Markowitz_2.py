"""
Package Import
"""
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import quantstats as qs
import gurobipy as gp
import warnings
import argparse
import sys

"""
Project Setup
"""
warnings.simplefilter(action="ignore", category=FutureWarning)

assets = [
    "SPY",
    "XLB",
    "XLC",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLRE",
    "XLU",
    "XLV",
    "XLY",
]

# Initialize Bdf and df
Bdf = pd.DataFrame()
for asset in assets:
    raw = yf.download(asset, start="2012-01-01", end="2024-04-01", auto_adjust = False)
    Bdf[asset] = raw['Adj Close']

df = Bdf.loc["2019-01-01":"2024-04-01"]

"""
Strategy Creation

Create your own strategy, you can add parameter but please remain "price" and "exclude" unchanged
"""


class MyPortfolio:
    """
    NOTE: You can modify the initialization function
    """

    def __init__(self, price, exclude, lookback=50, gamma=0):
        self.price = price
        self.returns = price.pct_change().fillna(0)
        self.exclude = exclude
        self.lookback = lookback
        self.gamma = gamma

    def calculate_weights(self):
        # Get the assets by excluding the specified column
        assets = self.price.columns[self.price.columns != self.exclude]

        # Calculate the portfolio weights
        self.portfolio_weights = pd.DataFrame(
            index=self.price.index, columns=self.price.columns
        )

        """
        TODO: Complete Task 4 Below
        """
        
        # --- 雙重市場體制切換 (Dual-Regime Momentum) ---
        # 核心：根據大盤 (SPY) 的狀態，切換 "進攻" 與 "防守" 兩套完全不同的邏輯。
        # 1. 牛市 (SPY > 年線): 追逐動能 (Momentum)，全力 Beat SPY。
        # 2. 熊市 (SPY < 年線): 躲進低波 (Min Vol)，全力保 Sharpe > 1。
        
        # 1. 準備資料
        # SPY 用來判斷大盤，其他 Assets 用來交易
        spy = self.price['SPY']
        target_price = self.price[assets]
        
        # 2. 判斷市場狀態 (Regime)
        # 使用 200 日均線 (年線) 作為牛熊分界
        sma_200 = spy.rolling(200).mean()
        is_bull = (spy > sma_200) # True=牛市, False=熊市
        
        # ----------------------------------------
        # 3. 計算兩套劇本的權重
        # ----------------------------------------
        
        # === 劇本 A: 牛市進攻 (Bull Strategy) ===
        # 邏輯：買過去半年 (126天) 漲最多的前 3 名，等權重分配
        mom_ret = target_price.pct_change(126)
        mom_rank = mom_ret.rank(axis=1, ascending=False)
        # 選前 3 名
        bull_signal = (mom_rank <= 3).astype(float)
        # 等權重 (1/3)
        bull_weights = bull_signal.div(3)
        
        # === 劇本 B: 熊市防守 (Bear Strategy) ===
        # 邏輯：買過去一個月 (20天) 波動最小的前 3 名，倒數波動率加權
        # 使用短週期波動率 (20天) 以快速反應恐慌
        short_vol = target_price.pct_change().rolling(20).std()
        vol_rank = short_vol.rank(axis=1, ascending=True) # 波動越小排名越前
        # 選前 3 名 (最穩的)
        bear_signal = (vol_rank <= 3).astype(float)
        
        # 倒數波動率加權 (在穩的裡面，買更穩的)
        inv_vol = 1.0 / (short_vol + 1e-9)
        bear_target = inv_vol * bear_signal
        bear_weights = bear_target.div(bear_target.sum(axis=1), axis=0)
        
        # ----------------------------------------
        # 4. 根據狀態進行切換 (Switching)
        # ----------------------------------------
        # 建立訊號矩陣 (將 is_bull 擴展到所有資產)
        # 這種寫法比迴圈快很多
        regime_signal = pd.DataFrame(index=self.price.index, columns=assets)
        for col in assets:
            regime_signal[col] = is_bull
            
        # 如果是 Bull，用 bull_weights；如果是 Bear，用 bear_weights
        # 使用 where(condition, other): condition 為 True 保留原值，False 填入 other
        # 注意: 這裡 regime_signal 為 True/False，我們需要將其轉為 float 做運算或直接用 where
        
        # 更直觀的寫法：
        # final = (bull_weights * 1) + (bear_weights * 0)  <-- 牛市
        # final = (bull_weights * 0) + (bear_weights * 1)  <-- 熊市
        
        final_weights = bull_weights * regime_signal.astype(float) + \
                        bear_weights * (~regime_signal).astype(float)
        
        # 5. Shift(1) 避免預視偏差
        final_weights = final_weights.shift(1)
        
        # 6. 填補空窗期 (Start-up Filling)
        # 前 200 天沒有 SMA 訊號，預設為 "牛市" (因為 2012 是漲勢)
        # 我們用 "全資產等權重" 填補，最中庸
        num_assets = len(assets)
        equal_weights = pd.DataFrame(1.0 / num_assets, index=self.price.index, columns=assets)
        
        final_weights = final_weights.combine_first(equal_weights)
        
        # 7. 填入結果
        self.portfolio_weights[assets] = final_weights
        self.portfolio_weights.fillna(0.0, inplace=True)
        
        """
        TODO: Complete Task 4 Above
        """

        self.portfolio_weights.ffill(inplace=True)
        self.portfolio_weights.fillna(0, inplace=True)

    def calculate_portfolio_returns(self):
        # Ensure weights are calculated
        if not hasattr(self, "portfolio_weights"):
            self.calculate_weights()

        # Calculate the portfolio returns
        self.portfolio_returns = self.returns.copy()
        assets = self.price.columns[self.price.columns != self.exclude]
        self.portfolio_returns["Portfolio"] = (
            self.portfolio_returns[assets]
            .mul(self.portfolio_weights[assets])
            .sum(axis=1)
        )

    def get_results(self):
        # Ensure portfolio returns are calculated
        if not hasattr(self, "portfolio_returns"):
            self.calculate_portfolio_returns()

        return self.portfolio_weights, self.portfolio_returns


if __name__ == "__main__":
    # Import grading system (protected file in GitHub Classroom)
    from grader_2 import AssignmentJudge
    parser = argparse.ArgumentParser(
        description="Introduction to Fintech Assignment 3 Part 12"
    )

    parser.add_argument(
        "--score",
        action="append",
        help="Score for assignment",
    )

    parser.add_argument(
        "--allocation",
        action="append",
        help="Allocation for asset",
    )

    parser.add_argument(
        "--performance",
        action="append",
        help="Performance for portfolio",
    )

    parser.add_argument(
        "--report", action="append", help="Report for evaluation metric"
    )

    parser.add_argument(
        "--cumulative", action="append", help="Cumulative product result"
    )

    args = parser.parse_args()

    judge = AssignmentJudge()
    
    # All grading logic is protected in grader_2.py
    judge.run_grading(args)
    