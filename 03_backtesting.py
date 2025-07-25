# 檔名: 03_backtesting.py
# 版本: 3.2 (最終修正版)

import matplotlib
# 關鍵修正：在導入 backtrader 之前，強制設定 matplotlib 使用無 GUI 的 'Agg' 後端
matplotlib.use('Agg')

import backtrader as bt
import pandas as pd
from pathlib import Path

# ==============================================================================
# 1. 建立自訂數據饋送類別 (Custom Data Feed)
# ==============================================================================
# 這個版本精確匹配 finta 函式庫的輸出欄位名稱
class PandasDataWithFeatures(bt.feeds.PandasData):
    lines = (
        'SMA_20', 'SMA_50', 'EMA_20', 'EMA_50', 'RSI_14',
        'MACD', 'SIGNAL',  # finta 的 MACD 輸出欄位
        'BB_UPPER', 'BB_MIDDLE', 'BB_LOWER',  # finta 的布林帶輸出欄位
        'ATR_14', 'OBV',
    )
    params = (
        ('SMA_20', -1), ('SMA_50', -1), ('EMA_20', -1), ('EMA_50', -1),
        ('RSI_14', -1), ('MACD', -1), ('SIGNAL', -1), ('BB_UPPER', -1),
        ('BB_MIDDLE', -1), ('BB_LOWER', -1), ('ATR_14', -1), ('OBV', -1),
    )


# ==============================================================================
# 2. 建立策略類別
# ==============================================================================
class SmaCrossStrategy(bt.Strategy):
    params = (('short_sma_period', 20), ('long_sma_period', 50),)

    def __init__(self):
        short_sma_name = f"SMA_{self.p.short_sma_period}"
        long_sma_name = f"SMA_{self.p.long_sma_period}"
        self.short_sma = getattr(self.data.lines, short_sma_name)
        self.long_sma = getattr(self.data.lines, long_sma_name)
        self.crossover = bt.indicators.CrossOver(self.short_sma, self.long_sma)

    def next(self):
        if not self.position:
            if self.crossover > 0:
                self.log('進場 BUY')
                self.buy()
        else:
            if self.crossover < 0:
                self.log('平倉 SELL')
                self.close()

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()}, {txt}')

# ==============================================================================
# 3. 主程式執行區塊
# ==============================================================================
if __name__ == '__main__':
    cerebro = bt.Cerebro() # 已修正拼寫錯誤
    data_path = Path("Output_Feature_Engineering/MarketData_with_Features/EURUSD_sml/EURUSD_sml_H4_features.parquet")

    print(f"正在加載數據: {data_path}")
    df = pd.read_parquet(data_path)
    df.index = pd.to_datetime(df.index)

    data_feed = PandasDataWithFeatures(dataname=df)
    cerebro.adddata(data_feed)

    cerebro.addstrategy(SmaCrossStrategy)
    cerebro.broker.setcash(10000.0)
    cerebro.broker.setcommission(commission=0.001)

    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe_ratio', timeframe=bt.TimeFrame.Days)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trade_analyzer')

    print('開始回測...')
    results = cerebro.run()
    strat = results[0]
    print('回測結束。')

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


    print("\n[INFO] Backtest calculation complete.")
    print("[INFO] Skipping built-in plotting due to environment conflicts.")
    print("[INFO] You can now proceed to the next step for manual plotting or analysis.")
