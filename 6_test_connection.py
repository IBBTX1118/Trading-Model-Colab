# 檔名: 6_test_connection.py
# 描述: 專門用於測試 Python 與 OANDA MT5 模擬帳戶的連線。

import MetaTrader5 as mt5
from datetime import datetime

# --- ↓↓↓ 請在這裡填入您的帳戶資訊 ↓↓↓ ---
# 從您的截圖中獲取的資訊
mt5_login = "1600014313"
mt5_server = "OANDA-Japan MT5 Demo"
# 請務必手動輸入您申請時設定的密碼
mt5_password = "#0#bbeQ69"


def connect_to_mt5():
    """初始化並連接到 MetaTrader 5"""
    print("正在初始化 MetaTrader 5...")

    # 嘗試初始化 MT5 終端
    if not mt5.initialize(login=mt5_login, password=mt5_password, server=mt5_server):
        print(f"MT5 初始化失敗, 錯誤代碼 = {mt5.last_error()}")
        print("請檢查：")
        print("1. OANDA MT5 桌面應用程式是否正在運行？")
        print("2. 帳號、密碼、伺服器名稱是否完全正確？")
        return False

    print(f"MT5 初始化成功！(版本: {mt5.version()})")
    return True


def get_account_info():
    """獲取並顯示帳戶資訊"""
    if not mt5.account_info():
        print("無法獲取帳戶資訊。")
        return

    # 獲取帳戶資訊對象
    info = mt5.account_info()._asdict()

    print("\n--- 帳戶資訊 ---")
    for key, value in info.items():
        print(f"{key:<15}: {value}")
    print("------------------")


def shutdown_mt5():
    """關閉與 MetaTrader 5 的連接"""
    print("\n正在關閉 MT5 連接...")
    mt5.shutdown()
    print("MT5 連接已關閉。")


if __name__ == "__main__":
    # --- 執行流程 ---
    # 1. 確保 OANDA MT5 桌面應用程式已打開並登入

    # 2. 連接到 MT5
    if connect_to_mt5():
        # 3. 獲取帳戶資訊
        get_account_info()

        # 4. 關閉連接
        shutdown_mt5()
