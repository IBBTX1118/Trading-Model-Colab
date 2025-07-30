# 檔名: 04_backtesting_MA_Cross.py
# 版本: 1.1 (基礎模型：修正 dataname 拼寫錯誤)

"""
此腳本為一個全新的基礎模型，基於雙指數移動平均線 (EMA) 交叉策略。

功能：
1. 實現一個完整的多空交易邏輯。
2. 使用 ATR 進行動態止損。
3. 使用固定風險百分比進行倉位大小管理。
4. 最終產出單一商品的完整績效報告。
"""

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
# 2. 策略類別：雙均線交叉策略
# ==============================================================================
class MACrossoverStrategy(bt.Strategy):
    params = (
        ('short_ema_period', 20),
        ('long_ema_period', 50),
        ('stop_loss_atr', 2.0), # 止損設為 2 倍 ATR
        ('risk_percent', 0.02),  # 單筆交易風險為總資金的 2%
    )

    def __init__(self):
        # 從數據中引用預先計算好的 EMA 和 ATR
        short_ema_name = f"EMA_{self.p.short_ema_period}"
        long_ema_name = f"EMA_{self.p.long_ema_period}"
        
        self.short_ema = getattr(self.data.lines, short_ema_name)
        self.long_ema = getattr(self.data.lines, long_ema_name)
        self.atr = self.data.lines.ATR_14
        
        # 使用 Backtrader 內建的交叉指標
        self.crossover = bt.indicators.CrossOver(self.short_ema, self.long_ema)

    def next(self):
        # 如果已有倉位，則不執行新的開倉邏輯
        if self.position:
            return

        # 黃金交叉，且無倉位 -> 買入
        if self.crossover > 0:
            stop_price = self.data.close[0] - self.p.stop_loss_atr * self.atr[0]
            risk_per_unit = self.data.close[0] - stop_price
            
            if risk_per_unit > 0:
                cash_to_risk = self.broker.getvalue() * self.p.risk_percent
                size = int(cash_to_risk / risk_per_unit)
                self.buy(size=size, exectype=bt.Order.Market, slprice=stop_price)

        # 死亡交叉，且無倉位 -> 賣出
        elif self.crossover < 0:
            stop_price = self.data.close[0] + self.p.stop_loss_atr * self.atr[0]
            risk_per_unit = stop_price - self.data.close[0]

            if risk_per_unit > 0:
                cash_to_risk = self.broker.getvalue() * self.p.risk_percent
                size = int(cash_to_risk / risk_per_unit)
                self.sell(size=size, exectype=bt.Order.Market, slprice=stop_price)

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED, Price: {order.executed.price:.5f}, Size: {order.executed.size}, Cost: {order.executed.value:.2f}')
            elif order.issell():
                self.log(f'SELL EXECUTED, Price: {order.executed.price:.5f}, Size: {order.executed.size}, Cost: {order.executed.value:.2f}')
        
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f'訂單 Canceled/Margin/Rejected: {order.getstatusname()}')

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()}, {txt}')

# ==============================================================================
# 3. 主程式執行區塊
# ==============================================================================
if __name__ == '__main__':
    cerebro = bt.Cerebro()

    # --- 數據加載 (我們先專注於單一商品，確保模型可執行) ---
    data_path = Path("Output_Feature_Engineering/MarketData_with_Features/EURUSD_sml/EURUSD_sml_H4_features.parquet")
    
    print(f"正在加載數據: {data_path}")
    df = pd.read_parquet(data_path)
    df.index = pd.to_datetime(df.index)

    # 【關鍵修正】將 datename 修正為 dataname
    data_feed = PandasDataWithFeatures(dataname=df)
    cerebro.adddata(data_feed)

    cerebro.addstrategy(MACrossoverStrategy)
    
    initial_cash = 10000.0
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(commission=0.001)

    # --- 加入分析工具 ---
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe_ratio', timeframe=bt.TimeFrame.Days)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trade_analyzer')
    cerebro.addanalyzer(bt.analyzers.SQN, _name='sqn')

    print('開始回測...')
    results = cerebro.run()
    strat = results[0]
    print('回測結束。')

    # --- 產出完整的績效報告 ---
    final_value = cerebro.broker.getvalue()
    pnl = final_value - initial_cash
    
    analysis_trade = strat.analyzers.trade_analyzer.get_analysis()
    analysis_sharpe = strat.analyzers.sharpe_ratio.get_analysis()
    analysis_drawdown = strat.analyzers.drawdown.get_analysis()
    analysis_sqn = strat.analyzers.sqn.get_analysis()

    print(f"\n{'='*30} 績效報告 {'='*30}")
    print(f"回測標的: {data_path.name}")
    print(f"初始資金: {initial_cash:,.2f}")
    print(f"最終資產: {final_value:,.2f}")
    print(f"淨損益: {pnl:,.2f}")
    print(f"總報酬率: {(pnl / initial_cash) * 100:.2f}%")
    
    print("-" * 72)
    
    total_trades = analysis_trade.total.total if hasattr(analysis_trade, 'total') else 0
    if total_trades > 0:
        win_rate = (analysis_trade.won.total / total_trades * 100) if hasattr(analysis_trade, 'won') else 0
        loss_rate = (analysis_trade.lost.total / total_trades * 100) if hasattr(analysis_trade, 'lost') else 0
        avg_win = analysis_trade.won.pnl.average if hasattr(analysis_trade, 'won') else 0
        avg_loss = analysis_trade.lost.pnl.average if hasattr(analysis_trade, 'lost') else 0
        
        print(f"總交易次數: {total_trades}")
        print(f"勝率: {win_rate:.2f}%")
        print(f"敗率: {loss_rate:.2f}%")
        print(f"平均獲利: {avg_win:.2f}")
        print(f"平均虧損: {avg_loss:.2f}")
        print(f"賺賠比 (Avg Win / Avg Loss): {abs(avg_win / avg_loss):.2f}" if avg_loss != 0 else "inf")
    else:
        print("總交易次數: 0")

    print("-" * 72)
    
    print(f"夏普比率 (年化): {analysis_sharpe.get('sharperatio', 'N/A')}")
    print(f"最大回撤: {analysis_drawdown.max.drawdown:.2f}%")
    print(f"系統品質指標 (SQN): {analysis_sqn.get('sqn', 'N/A'):.2f}")
    print(f"{'='*72}\n")
