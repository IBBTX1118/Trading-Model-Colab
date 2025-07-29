# 檔名: 03_backtesting.py
# 版本: 4.0 (策略升級：唐奇安 ATR 突破策略)

import matplotlib
matplotlib.use('Agg')

import backtrader as bt
import pandas as pd
from pathlib import Path

# ==============================================================================
# 1. 自訂數據饋送類別 (與之前相同)
# ==============================================================================
class PandasDataWithFeatures(bt.feeds.PandasData):
    lines = ('SMA_20', 'SMA_50', 'EMA_20', 'EMA_50', 'RSI_14', 'MACD', 
             'SIGNAL', 'BB_UPPER', 'BB_MIDDLE', 'BB_LOWER', 'ATR_14', 'OBV',)
    params = (('SMA_20', -1), ('SMA_50', -1), ('EMA_20', -1), ('EMA_50', -1), 
              ('RSI_14', -1), ('MACD', -1), ('SIGNAL', -1), ('BB_UPPER', -1), 
              ('BB_MIDDLE', -1), ('BB_LOWER', -1), ('ATR_14', -1), ('OBV', -1),)

# ==============================================================================
# 2. 策略類別：唐奇安 ATR 突破策略
# ==============================================================================
class DonchianATRStrategy(bt.Strategy):
    """
    實現報告中的唐奇安通道突破策略，並結合 ATR 進行動態倉位與止損管理。
    """
    params = (
        ('donchian_period', 20),
        ('atr_period', 14),
        ('stop_loss_atr', 1.5), # 止損設為 1.5 倍 ATR
        ('risk_percent', 0.01),  # 單筆交易風險為總資金的 1%
    )

    def __init__(self):
        # 引用預先計算好的 ATR 指標
        self.atr = self.data.lines.ATR_14
        
        # 使用 Backtrader 內建指標計算唐奇安通道
        self.donchian = bt.indicators.DonchianChannel(
            self.data, period=self.p.donchian_period
        )
        
        # 追蹤掛單
        self.stop_loss_order = None

    def next(self):
        # 如果有止損單正在掛單，或已有倉位，則不執行任何操作
        if self.stop_loss_order or self.position:
            return

        # 進場邏輯：當收盤價突破唐奇安通道上軌
        if self.data.close[0] > self.donchian.dch[-1]:
            # --- 動態倉位計算 ---
            stop_price = self.data.close[0] - self.p.stop_loss_atr * self.atr[0]
            risk_per_unit = self.data.close[0] - stop_price
            
            if risk_per_unit <= 0: # 避免除以零的錯誤
                return
                
            cash_to_risk = self.broker.getvalue() * self.p.risk_percent
            size = cash_to_risk / risk_per_unit
            
            self.log(f'進場 BUY, Size: {size:.2f}, Price: {self.data.close[0]:.5f}')
            self.buy(size=size)

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # 掛單已提交或被接受，不需處理
            return

        # 檢查訂單是否已完成
        if order.status in [order.Completed]:
            if order.isbuy():
                # --- ATR 止損設定 ---
                stop_price = order.executed.price - self.p.stop_loss_atr * self.atr[0]
                self.log(f'BUY EXECUTED, Price: {order.executed.price:.5f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
                self.log(f'設定止損單在: {stop_price:.5f}')
                self.stop_loss_order = self.sell(exectype=bt.Order.Stop, price=stop_price)

            elif order.issell():
                self.log(f'SELL EXECUTED, Price: {order.executed.price:.5f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('訂單 Canceled/Margin/Rejected')

        # 訂單處理完畢後，重設追蹤
        if order.status != order.Partial:
             self.stop_loss_order = None

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()}, {txt}')

# ==============================================================================
# 3. 主程式執行區塊 (升級為可迴圈測試)
# ==============================================================================
if __name__ == '__main__':
    # --- 定義要測試的商品與時間週期 ---
    symbols_to_test = ['EURUSD']
    timeframes_to_test = ['H1', 'H4', 'D1']

    for symbol in symbols_to_test:
        for timeframe in timeframes_to_test:
            # --- 每次迴圈都建立一個全新的 Cerebro 引擎 ---
            cerebro = bt.Cerebro()
            
            data_path = Path(f"Output_Feature_Engineering/MarketData_with_Features/{symbol}_sml/{symbol}_sml_{timeframe}_features.parquet")
            
            print(f"\n{'='*25} 開始回測: {symbol} - {timeframe} {'='*25}")
            print(f"正在加載數據: {data_path}")

            if not data_path.exists():
                print(f"數據檔案不存在，跳過此輪回測。")
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
            print(f"初始資金: 10000.00")
            print(f"最終資產: {cerebro.broker.getvalue():.2f}")
            
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
