# 檔名: 03_backtesting.py
# 版本: 2.0 (使用自訂 Data Feed)

import backtrader as bt
import pandas as pd
from pathlib import Path

# ==============================================================================
# 1. 建立自訂數據饋送類別 (Custom Data Feed)
# ==============================================================================
# 我們繼承 bt.feeds.PandasData，並告訴它我們有哪些額外的欄位
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
        self.short_sma = self.data.lines.SMA_20
        self.long_sma = self.data.lines.SMA_50
        
        # 也可以用 self.p.xxx 的方式讓參數更靈活，但直接存取更清晰
        # short_sma_name = 'SMA_' + str(self.p.short_sma)
        # self.short_sma = getattr(self.data.lines, short_sma_name)

    def next(self):
        if self.position:
            if self.short_sma < self.long_sma:
                self.log('平倉 SELL')
                self.close()
        else:
            if self.short_sma > self.long_sma:
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
    cerebro = bt.Cerebro()

    # 讀取帶有特徵的數據
    data_path = Path("Output_Feature_Engineering/EURUSD_sml/EURUSD_sml_H4_features.parquet")
    df = pd.read_parquet(data_path)
    df.index = pd.to_datetime(df.index)

    # 關鍵：使用我們自訂的 PandasDataWithFeatures，而不是標準的 bt.feeds.PandasData
    data_feed = PandasDataWithFeatures(dataname=df)
    cerebro.adddata(data_feed)

    cerebro.addstrategy(SmaCrossStrategy)

    cerebro.broker.setcash(10000.0)
    cerebro.broker.setcommission(commission=0.001)

    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe_ratio')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trade_analyzer')

    print('開始回測...')
    results = cerebro.run()
    strat = results[0]
    print('回測結束。')

    # 打印績效
    print(f"\n{'='*30} 績效報告 {'='*30}")
    print(f"初始資金: 10000.00")
    print(f"最終資產: {cerebro.broker.getvalue():.2f}")
    
    analysis = strat.analyzers.trade_analyzer.get_analysis()
    sharpe = strat.analyzers.sharpe_ratio.get_analysis()
    drawdown = strat.analyzers.drawdown.get_analysis()

    if analysis.total.total > 0:
        print(f"總交易次數: {analysis.total.total}")
        print(f"勝率: {analysis.won.total / analysis.total.total * 100:.2f}%")
    else:
        print("總交易次數: 0")

    print(f"夏普比率: {sharpe.get('sharperatio', 'N/A')}")
    print(f"最大回撤: {drawdown.max.drawdown:.2f}%")
    print(f"{'='*70}\n")

    # 繪製績效圖
    cerebro.plot(style='candlestick', iplot=False)
