# 檔名: 6_test_connection.py
# 描述: 專門用於測試 Python 與 OANDA MT5 模擬帳戶的連線。(v1.1 修復版)

import MetaTrader5 as mt5

# from datetime import datetime # <--- 修改點 1：移除了未被使用的 import

# --- ↓↓↓ 請在這裡填入您的帳戶資訊 ↓↓↓ ---
# 從您的截圖中獲取的資訊
MT5_LOGIN = 1600014313  # <--- 修改點 2：將登入名改為整數 (移除引號)，並改為大寫命名
MT5_SERVER = "OANDA-Demo-1"  # <--- 修改點 3：變數名改為大寫
# 請務必手動輸入您申請時設定的密碼
MT5_PASSWORD = "2130C2f3a966@"  # <--- 修改點 4：變數名改為大寫


def connect_to_mt5():
    """初始化並連接到 MetaTrader 5"""
    print("正在初始化 MetaTrader 5...")

    # 嘗試初始化 MT5 終端
    # 使用修正後的變數
    if not mt5.initialize(login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER):
        print(f"MT5 初始化失敗, 錯誤代碼 = {mt5.last_error()}")
        print("請檢查：")
        print("1. OANDA MT5 桌面應用程式是否正在運行？")
        print("2. 帳號、密碼、伺服器名稱是否完全正確？")
        return False

    print(f"MT5 初始化成功！(版本: {mt5.version()})")
    return True


def get_account_info():
    """獲取並顯示帳戶資訊"""
    account_info = mt5.account_info()
    if account_info is None:
        print("無法獲取帳戶資訊。")
        return

    # 獲取帳戶資訊對象
    info = account_info._asdict()

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
