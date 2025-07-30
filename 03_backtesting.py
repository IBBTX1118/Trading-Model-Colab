# 檔名: 03_backtesting.py
# 版本: 6.2 (最終穩定版：修正獲利因子計算)

"""
此腳本為策略回測的最終穩定版。

功能：
1. 使用 multiprocessing 並行處理，加速多商品回測。
2. 迴圈測試多個指定商品。
3. 完全移除繪圖功能，避免任何潛在的環境衝突。
4. 修正了訂單狀態管理與績效計算邏輯，確保程式穩健運行。
5. 最終以表格形式，彙總並打印所有商品的回測績效報告。
"""

import backtrader as bt
import pandas as pd
from pathlib import Path
import multiprocessing
import time

# ==============================================================================
# 1. 自訂數據饋送類別
# ==============================================================================
class PandasDataWithFeatures(bt.feeds.PandasData):
    lines = ('SMA_20', 'SMA_50', 'EMA_20', 'EMA_50', 'RSI_14', 'MACD', 
             'SIGNAL', 'BB_UPPER', 'BB_MIDDLE', 'BB_LOWER', 'ATR_14', 'OBV',)
    params = (('SMA_20', -1), ('SMA_50', -1), ('EMA_20', -1), ('EMA_50', -1), 
              ('RSI_14', -1), ('MACD', -1), ('SIGNAL', -1), ('BB_UPPER', -1), 
              ('BB_MIDDLE', -1), ('BB_LOWER', -1), ('ATR_14', -1), ('OBV', -1),)

# ==============================================================================
# 2. 策略類別
# ==============================================================================
class DonchianATRStrategy(bt.Strategy):
    params = (
        ('donchian_period', 20),
        ('atr_period', 14),
        ('stop_loss_atr', 1.5),
        ('risk_percent', 0.01),
        ('verbose', False),
    )

    def __init__(self):
        self.atr = self.data.lines.ATR_14
        self.donchian_high = bt.indicators.Highest(self.data.high, period=self.p.donchian_period)
        self.buy_order = None
        self.stop_loss_order = None

    def next(self):
        if self.buy_order or self.stop_loss_order or self.position:
            return

        if self.data.close[0] > self.donchian_high[-1]:
            stop_price = self.data.close[0] - self.p.stop_loss_atr * self.atr[0]
            risk_per_unit = self.data.close[0] - stop_price
            
            if risk_per_unit <= 0: return
                
            cash_to_risk = self.broker.getvalue() * self.p.risk_percent
            size = cash_to_risk / risk_per_unit
            
            self.buy_order = self.buy(size=int(size))

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                stop_price = order.executed.price - self.p.stop_loss_atr * self.atr[0]
                self.stop_loss_order = self.sell(exectype=bt.Order.Stop, price=stop_price, size=order.executed.size)
        
        if order == self.buy_order:
            self.buy_order = None
        elif order == self.stop_loss_order:
            self.stop_loss_order = None

# ==============================================================================
# 3. 獨立的回測執行函數
# ==============================================================================
def run_backtest(symbol):
    cerebro = bt.Cerebro(stdstats=False)

    data_path = Path(f"Output_Feature_Engineering/MarketData_with_Features/{symbol}_sml/{symbol}_sml_H4_features.parquet")
    
    if not data_path.exists():
        print(f"數據檔案不存在: {data_path}，跳過 {symbol}。")
        return None

    df = pd.read_parquet(data_path)
    df.index = pd.to_datetime(df.index)

    data_feed = PandasDataWithFeatures(dataname=df)
    data_feed._name = symbol
    cerebro.adddata(data_feed)

    cerebro.addstrategy(DonchianATRStrategy)
    
    initial_cash = 10000.0
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(commission=0.001)

    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe_ratio', timeframe=bt.TimeFrame.Days)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trade_analyzer')

    results = cerebro.run()
    strat = results[0]
    
    final_value = cerebro.broker.getvalue()
    pnl = final_value - initial_cash
    
    analysis_trade = strat.analyzers.trade_analyzer.get_analysis()
    analysis_sharpe = strat.analyzers.sharpe_ratio.get_analysis()
    analysis_drawdown = strat.analyzers.drawdown.get_analysis()

    total_trades = analysis_trade.total.total if hasattr(analysis_trade, 'total') else 0
    win_rate = (analysis_trade.won.total / total_trades * 100) if total_trades > 0 and hasattr(analysis_trade, 'won') else 0
    
    # 【關鍵修正】更穩健的獲利因子計算方式
    if hasattr(analysis_trade, 'pnl') and hasattr(analysis_trade.pnl, 'gross'):
        gross_profit = analysis_trade.pnl.gross.total
        # 使用 .get() 安全地獲取虧損值，如果不存在則預設為 0
        gross_loss = abs(analysis_trade.pnl.net.get('lost', 0.0))
        if gross_loss > 0:
            profit_factor = gross_profit / gross_loss
        else:
            profit_factor = float('inf')  # 如果沒有虧損，獲利因子為無限大
    else:
        profit_factor = 0.0

    performance_dict = {
        "商品 (Symbol)": symbol,
        "最終資產 (Final Value)": f"{final_value:,.2f}",
        "淨損益 (Net PnL)": f"{pnl:,.2f}",
        "總報酬率 (Total Return %)": f"{(pnl / initial_cash) * 100:.2f}",
        "夏普比率 (Annualized Sharpe)": f"{analysis_sharpe.get('sharperatio', 0):.2f}",
        "最大回撤 (Max Drawdown %)": f"{analysis_drawdown.max.drawdown:.2f}",
        "總交易次數 (Total Trades)": total_trades,
        "勝率 (Win Rate %)": f"{win_rate:.2f}",
        "獲利因子 (Profit Factor)": f"{profit_factor:.2f}",
    }
    return performance_dict

# ==============================================================================
# 4. 主程式執行區塊
# ==============================================================================
if __name__ == '__main__':
    symbols_to_test = ['EURUSD', 'USDJPY', 'GBPUSD', 'AUDUSD']
    
    print(f"準備對 {len(symbols_to_test)} 個商品進行並行回測...")
    start_time = time.time()

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.map(run_backtest, symbols_to_test)

    end_time = time.time()
    print(f"所有回測執行完畢，總耗時: {end_time - start_time:.2f} 秒")

    valid_results = [res for res in results if res is not None]
    
    if valid_results:
        performance_df = pd.DataFrame(valid_results)
        performance_df.set_index("商品 (Symbol)", inplace=True)
        
        print("\n" + "="*35 + " 綜合績效報告 " + "="*35)
        print(performance_df)
        print("="*105)
