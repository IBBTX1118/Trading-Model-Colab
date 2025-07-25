# 檔名: 03_backtesting.py
# 版本: 1.0 (正式版：使用 Backtrader)

"""
此腳本為策略回測階段。

它會讀取由 02_feature_engineering.py 產生的、帶有特徵的 Parquet 檔案，
然後使用 Backtrader 框架來執行一個簡單的交易策略，並輸出績效報告與圖表。
"""

import backtrader as bt
import pandas as pd
from pathlib import Path

# ==============================================================================
# 1. 建立自訂數據饋送類別 (Custom Data Feed)
# ==============================================================================
# 我們繼承 bt.feeds.PandasData，並告訴它我們有哪些額外的特徵欄位
class PandasDataWithFeatures(bt.feeds.PandasData):
    """
    一個自訂的數據饋送，用來讓 Backtrader 識別我們預先計算好的特徵。
    """
    # 告訴 Backtrader，我們的數據中有這些額外的「數據線 (Line)」
    lines = (
        'SMA_20', 'SMA_50', 'EMA_20', 'EMA_50', 'RSI_14',
        'MACD', 'SIGNAL', # finta 的 MACD 輸出欄位
        'BB_UPPER', 'BB_MIDDLE', 'BB_LOWER', # finta 的布林帶輸出欄位
        'ATR_14', 'OBV',
    )

    # 將這些數據線與 DataFrame 中的欄位對應起來
    # -1 表示 Backtrader 會自動按名稱尋找對應的欄位
    params = (
        ('SMA_20', -1), ('SMA_50', -1), ('EMA_20', -1), ('EMA_50', -1),
        ('RSI_14', -1), ('MACD', -1), ('SIGNAL', -1), ('BB_UPPER', -1),
        ('BB_MIDDLE', -1), ('BB_LOWER', -1), ('ATR_14', -1), ('OBV', -1),
    )


# ==============================================================================
# 2. 建立策略類別
# ==============================================================================
class SmaCrossStrategy(bt.Strategy):
    """一個簡單的雙均線交叉策略"""
    params = (('short_sma', 20), ('long_sma', 50),)

    def __init__(self):
        # 現在可以直接透過 .lines.xxx 的方式來存取我們自訂的數據線
        # 注意：這裡的名稱必須與 PandasDataWithFeatures 中 lines 定義的完全一致
        self.short_sma = self.data.lines.SMA_20
        self.long_sma = self.data.lines.SMA_50
        
        # 使用 Backtrader 內建的交叉指標，讓邏輯更簡潔
        self.crossover = bt.indicators.CrossOver(self.short_sma, self.long_sma)

    def next(self):
        # 如果已有倉位
        if self.position:
            # 如果出現死亡交叉 (短期均線下穿長期均線)，則平倉
            if self.crossover < 0:
                self.log('平倉 SELL')
                self.close()
        # 如果沒有倉位
        else:
            # 如果出現黃金交叉 (短期均線上穿長期均線)，則進場
            if self.crossover > 0:
                self.log('進場 BUY')
                self.buy()

    def log(self, txt, dt=None):
        """策略的日誌記錄功能"""
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()}, {txt}')


# ==============================================================================
# 3. 主程式執行區塊
# ==============================================================================
if __name__ == '__main__':
    cerebro = bt.Cerebro() # 建立 Cerebro 引擎

    # --- 數據加載 ---
    # 您可以修改這個路徑來回測不同的商品或時間週期
    data_path = Path("Output_Feature_Engineering/EURUSD_sml/EURUSD_sml_H4_features.parquet")
    
    print(f"正在加載數據: {data_path}")
    df = pd.read_parquet(data_path)
    df.index = pd.to_datetime(df.index) # 確保時間索引格式正確

    # 關鍵：使用我們自訂的 PandasDataWithFeatures
    data_feed = PandasDataWithFeatures(dataname=df)
    cerebro.adddata(data_feed)

    # --- 策略與參數設定 ---
    cerebro.addstrategy(SmaCrossStrategy)
    cerebro.broker.setcash(10000.0) # 設定初始資金
    cerebro.broker.setcommission(commission=0.001) # 設定交易佣金 (例如千分之一)

    # --- 績效分析工具 ---
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe_ratio', timeframe=bt.TimeFrame.Days)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trade_analyzer')

    # --- 執行與結果打印 ---
    print('開始回測...')
    results = cerebro.run()
    strat = results[0]
    print('回測結束。')

    # 打印績效報告
    print(f"\n{'='*30} 績效報告 {'='*30}")
    print(f"初始資金: 10000.00")
    print(f"最終資產: {cerebro.broker.getvalue():.2f}")
    
    analysis = strat.analyzers.trade_analyzer.get_analysis()
    sharpe = strat.analyzers.sharpe_ratio.get_analysis()
    drawdown = strat.analyzers.drawdown.get_analysis()

    if hasattr(analysis, 'total') and analysis.total.total > 0:
        print(f"總交易次數: {analysis.total.total}")
        print(f"勝率: {analysis.won.total / analysis.total.total * 100:.2f}%")
    else:
        print("總交易次數: 0")

    print(f"夏普比率 (年化): {sharpe.get('sharperatio', 'N/A')}")
    print(f"最大回撤: {drawdown.max.drawdown:.2f}%")
    print(f"{'='*72}\n")

    # 繪製績效圖
    # 在 Colab 中，iplot=False 會將圖表儲存為 plot.png 檔案
    print('正在生成圖表...')
    cerebro.plot(style='candlestick', iplot=False)
    print('圖表已儲存為 plot.png')
