# 檔名: 03_backtesting.py
# 版本: 4.1 (最終修正版：手動建構唐奇安通道)

import matplotlib
matplotlib.use('Agg')

import backtrader as bt
import pandas as pd
from pathlib import Path

# ==============================================================================
# 1. 自訂數據饋送類別 (維持不變)
# ==============================================================================
class PandasDataWithFeatures(bt.feeds.PandasData):
    lines = ('SMA_20', 'SMA_50', 'EMA_20', 'EMA_50', 'RSI_14', 'MACD', 
             'SIGNAL', 'BB_UPPER', 'BB_MIDDLE', 'BB_LOWER', 'ATR_14', 'OBV',)
    params = (('SMA_20', -1), ('SMA_50', -1), ('EMA_20', -1), ('EMA_50', -1), 
              ('RSI_14', -1), ('MACD', -1), ('SIGNAL', -1), ('BB_UPPER', -1), 
              ('BB_MIDDLE', -1), ('BB_LOWER', -1), ('ATR_14', -1), ('OBV', -1),)

# ==============================================================================
# 2. 策略類別：唐奇安 ATR 突破策略 (已修正)
# ==============================================================================
class DonchianATRStrategy(bt.Strategy):
    params = (
        ('donchian_period', 20),
        ('atr_period', 14),
        ('stop_loss_atr', 1.5),
        ('risk_percent', 0.01),
    )

    def __init__(self):
        self.atr = self.data.lines.ATR_14
        
        # 【關鍵修正】使用 Highest 和 Lowest 指標來手動建構唐奇安通道
        self.donchian_high = bt.indicators.Highest(
            self.data.high, period=self.p.donchian_period
        )
        self.donchian_low = bt.indicators.Lowest(
            self.data.low, period=self.p.donchian_period
        )
        
        self.stop_loss_order = None

    def next(self):
        if self.stop_loss_order or self.position:
            return

        if self.data.close[0] > self.donchian_high[-1]:
            # --- 修正後的邏輯順序 ---
            # 1. 先計算止損價格
            stop_price = self.data.close[0] - self.p.stop_loss_atr * self.atr[0]
            # 2. 接著計算每單位風險
            risk_per_unit = self.data.close[0] - stop_price
            # 3. 然後才檢查每單位風險是否有效 (必須大於 0)
            if risk_per_unit <= 0:
                return
            # 4. 最後才計算倉位大小
            cash_to_risk = self.broker.getvalue() * self.p.risk_percent
            size = cash_to_risk / risk_per_unit
            
            self.log(f'進場 BUY, Size: {size:.2f}, Price: {self.data.close[0]:.5f}')
            self.buy(size=size)

    def notify_order(self, order):
        # ... (此函數內容與之前版本相同) ...
        if order.status in [order.Submitted, order.Accepted]:
            return
        if order.status in [order.Completed]:
            if order.isbuy():
                stop_price = order.executed.price - self.p.stop_loss_atr * self.atr[0]
                self.log(f'BUY EXECUTED, Price: {order.executed.price:.5f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
                self.log(f'設定止損單在: {stop_price:.5f}')
                self.stop_loss_order = self.sell(exectype=bt.Order.Stop, price=stop_price)
            elif order.issell():
                self.log(f'SELL EXECUTED, Price: {order.executed.price:.5f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('訂單 Canceled/Margin/Rejected')
        if order.status != order.Partial:
             self.stop_loss_order = None

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()}, {txt}')

# ==============================================================================
# 3. 主程式執行區塊 (與之前版本相同)
# ==============================================================================
if __name__ == '__main__':
    # --- 迴圈測試設定 ---
    symbols_to_test = ['EURUSD']
    timeframes_to_test = ['H1', 'H4', 'D1']

    for symbol in symbols_to_test:
        for timeframe in timeframes_to_test:
            cerebro = bt.Cerebro()
            data_path = Path(f"Output_Feature_Engineering/MarketData_with_Features/{symbol}_sml/{symbol}_sml_{timeframe}_features.parquet")
            
            print(f"\n{'='*25} 開始回測: {symbol} - {timeframe} {'='*25}")
            if not data_path.exists():
                print(f"數據檔案不存在: {data_path}，跳過。")
                continue

            df = pd.read_parquet(data_path)
            df.index = pd.to_datetime(df.index)

            data_feed = PandasDataWithFeatures(dataname=df)
            cerebro.adddata(data_feed)

            cerebro.addstrategy(DonchianATRStrategy)
            cerebro.broker.setcash(10000.0)
            cerebro.broker.setcommission(commission=0.001)

            cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe_ratio', timeframe=bt.TimeFrame.Days)
            cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
            cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trade_analyzer')

            results = cerebro.run()
            strat = results[0]
            
            # --- 打印績效報告 ---
            print(f"\n----------- 績效報告: {symbol} - {timeframe} -----------")
            print(f"初始資金: 10000.00, 最終資產: {cerebro.broker.getvalue():.2f}")
            
            analysis = strat.analyzers.trade_analyzer.get_analysis()
            if hasattr(analysis, 'total') and analysis.total.total > 0:
                print(f"總交易次數: {analysis.total.total}, 勝率: {analysis.won.total / analysis.total.total * 100:.2f}%")
            else:
                print("總交易次數: 0")
            
            print(f"夏普比率 (年化): {strat.analyzers.sharpe_ratio.get_analysis().get('sharperatio', 'N/A')}")
            print(f"最大回撤: {strat.analyzers.drawdown.get_analysis().max.drawdown:.2f}%")
            print(f"{'-'*60}\n")
            
            # --- 儲存圖表 ---
            figure = cerebro.plot(style='candlestick', iplot=False)[0][0]
            plot_filename = f'backtest_result_{symbol}_{timeframe}.png'
            figure.savefig(plot_filename)
            print(f"圖表已儲存為 {plot_filename}")

    print(f"\n{'='*30} 所有回測執行完畢 {'='*30}")
