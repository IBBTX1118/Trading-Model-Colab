import backtrader as bt
import pandas as pd
from pathlib import Path

# 1. 建立策略類別 (這裡是您交易邏輯的核心)
class SmaCrossStrategy(bt.Strategy):
    params = (('short_sma', 20), ('long_sma', 50),)

class SmaCrossStrategy(bt.Strategy):
    params = (('short_sma', 20), ('long_sma', 50),)

    def __init__(self):
        # 建立指標的完整欄位名稱
        short_sma_name = 'SMA_' + str(self.p.short_sma)
        long_sma_name = 'SMA_' + str(self.p.long_sma)

        # 使用 getattr 這種更穩定的方式來按名稱獲取數據線
        self.short_sma = getattr(self.data.lines, short_sma_name)
        self.long_sma = getattr(self.data.lines, long_sma_name)

    def next(self):
        # 如果已有倉位，則暫不操作
        if self.position:
            # 可以加入平倉邏輯，例如長期均線反轉
            if self.short_sma < self.long_sma:
                self.close()
        # 如果沒有倉位，判斷進場
        else:
            if self.short_sma > self.long_sma:
                self.buy()

# 3. 主程式區塊
if __name__ == '__main__':
    cerebro = bt.Cerebro() # 建立大腦

    # 讀取帶有特徵的數據
    data_path = Path("Output_Feature_Engineering/MarketData_with_Features/EURUSD_sml_H4_features.parquet")
    df = pd.read_parquet(data_path)
    df.index = pd.to_datetime(df.index) # 確保時間索引格式正確

    # 將 DataFrame 餵給 Backtrader
    data_feed = bt.feeds.PandasData(dataname=df)
    cerebro.adddata(data_feed)

    # 將策略加入大腦
    cerebro.addstrategy(SmaCrossStrategy)

    # 設定初始資金
    cerebro.broker.setcash(10000.0)
    # 設定交易佣金 (例如 0.1%)
    cerebro.broker.setcommission(commission=0.001)

    # 加入績效分析工具
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe_ratio')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')

    # 執行回測
    print('開始回測...')
    results = cerebro.run()
    strat = results[0]
    print('回測結束。')

    # 打印績效
    print(f"最終資產: {cerebro.broker.getvalue():.2f}")
    print(f"夏普比率: {strat.analyzers.sharpe_ratio.get_analysis().get('sharperatio', 'N/A')}")
    print(f"最大回撤: {strat.analyzers.drawdown.get_analysis().max.drawdown:.2f}%")

    # 繪製績效圖
    cerebro.plot(style='candlestick')
