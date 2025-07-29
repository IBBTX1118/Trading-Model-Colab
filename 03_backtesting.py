# 檔名: 03_backtesting.py
# 版本: 5.0 (並行多商品回測版)

"""
此腳本為策略回測的升級版。

功能：
1. 使用 multiprocessing 並行處理，加速多商品回測。
2. 迴圈測試多個指定商品 (EURUSD, USDJPY, GBPUSD, AUDUSD) 的 H4 數據。
3. 移除逐筆交易日誌與繪圖功能，專注於績效計算。
4. 最終以表格形式，彙總並打印所有商品的回測績效報告。
"""

import backtrader as bt
import pandas as pd
from pathlib import Path
import multiprocessing
import time

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
# 2. 策略類別 (移除 log 功能以保持輸出乾淨)
# ==============================================================================
class DonchianATRStrategy(bt.Strategy):
    params = (
        ('donchian_period', 20),
        ('atr_period', 14),
        ('stop_loss_atr', 1.5),
        ('risk_percent', 0.01),
        ('verbose', False), # 新增參數，控制是否打印交易日誌
    )

    def __init__(self):
        self.atr = self.data.lines.ATR_14
        self.donchian_high = bt.indicators.Highest(self.data.high, period=self.p.donchian_period)
        self.stop_loss_order = None

    def next(self):
        if self.stop_loss_order or self.position:
            return

        if self.data.close[0] > self.donchian_high[-1]:
            stop_price = self.data.close[0] - self.p.stop_loss_atr * self.atr[0]
            risk_per_unit = self.data.close[0] - stop_price
            
            if risk_per_unit <= 0: return
                
            cash_to_risk = self.broker.getvalue() * self.p.risk_percent
            size = cash_to_risk / risk_per_unit
            
            self.buy(size=int(size))

    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                stop_price = order.executed.price - self.p.stop_loss_atr * self.atr[0]
                self.stop_loss_order = self.sell(exectype=bt.Order.Stop, price=stop_price, size=order.executed.size)
                self.log(f'BUY EXECUTED, Price: {order.executed.price:.5f}')
            elif order.issell():
                self.log(f'SELL EXECUTED, Price: {order.executed.price:.5f}')
        
        if order.status not in [order.Submitted, order.Accepted, order.Partial]:
             self.stop_loss_order = None

    def log(self, txt, dt=None):
        # 只有在 verbose 設為 True 時才打印日誌
        if self.p.verbose:
            dt = dt or self.datas[0].datetime.date(0)
            print(f'{dt.isoformat()}, {self.data._name}, {txt}')

# ==============================================================================
# 3. 獨立的回測執行函數 (為並行處理做準備)
# ==============================================================================
def run_backtest(symbol):
    """
    對單一商品執行回測，並回傳績效字典。
    """
    cerebro = bt.Cerebro(stdstats=False) # 關閉預設的統計打印

    data_path = Path(f"Output_Feature_Engineering/MarketData_with_Features/{symbol}_sml/{symbol}_sml_H4_features.parquet")
    
    if not data_path.exists():
        print(f"數據檔案不存在: {data_path}，跳過 {symbol}。")
        return None

    df = pd.read_parquet(data_path)
    df.index = pd.to_datetime(df.index)

    data_feed = PandasDataWithFeatures(dataname=df)
    data_feed._name = symbol # 為數據命名，方便日誌追蹤
    cerebro.adddata(data_feed)

    cerebro.addstrategy(DonchianATRStrategy)
    
    initial_cash = 10000.0
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(commission=0.001)

    # 加入更豐富的分析工具
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe_ratio', timeframe=bt.TimeFrame.Days, annualization=252)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trade_analyzer')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns', timeframe=bt.TimeFrame.Days)

    # 執行回測
    results = cerebro.run()
    strat = results[0]
    
    # --- 整理並回傳績效報告 ---
    final_value = cerebro.broker.getvalue()
    pnl = final_value - initial_cash
    
    analysis_trade = strat.analyzers.trade_analyzer.get_analysis()
    analysis_sharpe = strat.analyzers.sharpe_ratio.get_analysis()
    analysis_drawdown = strat.analyzers.drawdown.get_analysis()
    analysis_returns = strat.analyzers.returns.get_analysis()

    performance_dict = {
        "商品 (Symbol)": symbol,
        "最終資產 (Final Value)": f"{final_value:,.2f}",
        "淨損益 (Net PnL)": f"{pnl:,.2f}",
        "總報酬率 (Total Return %)": f"{(pnl / initial_cash) * 100:.2f}",
        "夏普比率 (Annualized Sharpe)": f"{analysis_sharpe.get('sharperatio', 0):.2f}",
        "最大回撤 (Max Drawdown %)": f"{analysis_drawdown.max.drawdown:.2f}",
        "總交易次數 (Total Trades)": analysis_trade.total.total if hasattr(analysis_trade, 'total') else 0,
        "勝率 (Win Rate %)": (analysis_trade.won.total / analysis_trade.total.total * 100) if hasattr(analysis_trade, 'won') and analysis_trade.total.total > 0 else 0,
        "獲利因子 (Profit Factor)": (analysis_trade.pnl.gross.total / abs(analysis_trade.pnl.net.lost or 1)) if hasattr(analysis_trade, 'pnl') else 0,
    }
    return performance_dict

# ==============================================================================
# 4. 主程式執行區塊 (使用並行處理)
# ==============================================================================
if __name__ == '__main__':
    # --- 定義要測試的商品列表 ---
    symbols_to_test = ['EURUSD', 'USDJPY', 'GBPUSD', 'AUDUSD']
    
    print(f"準備對 {len(symbols_to_test)} 個商品進行並行回測...")
    start_time = time.time()

    # --- 使用 multiprocessing.Pool 建立 CPU 核心數量的進程池 ---
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        # 使用 pool.map 將 run_backtest 函數應用到每一個商品上
        results = pool.map(run_backtest, symbols_to_test)

    end_time = time.time()
    print(f"所有回測執行完畢，總耗時: {end_time - start_time:.2f} 秒")

    # --- 整理並打印最終的績效報告表格 ---
    # 過濾掉執行失敗的結果 (例如找不到檔案)
    valid_results = [res for res in results if res is not None]
    
    if valid_results:
        performance_df = pd.DataFrame(valid_results)
        performance_df.set_index("商品 (Symbol)", inplace=True)
        
        # 格式化部分欄位的輸出
        performance_df['勝率 (Win Rate %)'] = performance_df['勝率 (Win Rate %)'].apply(lambda x: f"{x:.2f}")
        performance_df['獲利因子 (Profit Factor)'] = performance_df['獲利因子 (Profit Factor)'].apply(lambda x: f"{x:.2f}")
        
        print("\n" + "="*35 + " 綜合績效報告 " + "="*35)
        print(performance_df)
        print("="*85)
