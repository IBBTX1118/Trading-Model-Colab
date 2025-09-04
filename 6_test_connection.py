# 檔名: 6_test_connection.py
# 描述: 專門用於測試 Python 與 OANDA MT5 模擬帳戶的連線。
# 版本: 1.2 (採用與腳本1相同的「搭便車」連線模式)

import MetaTrader5 as mt5

# --- ↓↓↓ 帳戶資訊（修改後不再用於直接登入）↓↓↓ ---
# 由於改為使用「搭便車」模式連線，以下資訊不再直接傳入 mt5.initialize()
# 您可以保留它們作為參考，但腳本不會使用它們來登入。
MT5_LOGIN = 1600014313
MT5_SERVER = "OANDA-Demo-1"
MT5_PASSWORD = "2130C23966@" # 密碼不再需要


def connect_to_mt5():
    """初始化並連接到 MetaTrader 5 (使用搭便車模式)"""
    print("正在初始化 MetaTrader 5 (搭便車模式)...")

    # ==============================================================================
    # ★★★ 核心修改點 ★★★
    # 移除 mt5.initialize() 中的 login, password, server 參數
    # 這會讓它自動尋找並連接到已登入的 MT5 桌面應用程式
    # ==============================================================================
    if not mt5.initialize():
        print(f"MT5 初始化失敗, 錯誤代碼 = {mt5.last_error()}")
        print("\n請檢查：")
        # ★★★ 修改點：更新錯誤提示 ★★★
        print("1. OANDA MT5 桌面應用程式是否已開啟，並且『成功登入』到您的帳戶？")
        print("2. 如果 MT5 是 64 位元版本，請確保您的 Python 也是 64 位元。")
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

    # ★★★ 修改點：增加連線帳戶驗證 ★★★
    print("\n--- 已連線的帳戶資訊 ---")
    if info['login'] == MT5_LOGIN:
        print(f"✅ 成功連接到預期的帳戶: {info['login']}")
    else:
        print(f"⚠️  警告：腳本連接到的帳戶 ({info['login']}) 與設定檔中的帳戶 ({MT5_LOGIN}) 不符！")
        print("   請確認您 MT5 桌面程式登入的是正確的帳戶。")

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
