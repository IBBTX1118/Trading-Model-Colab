# 檔名: 6_test_connection.py
# 描述: 專門用於測試 Python 與 OANDA MT5 模擬帳戶的連線。
# 版本: 1.3 (使用您提供的最終路徑)

import MetaTrader5 as mt5

# --- 參考資訊 ---
MT5_LOGIN = 1600014313
MT5_SERVER = "OANDA-Demo-1"

def connect_to_mt5():
    """初始化並連接到 MetaTrader 5 (指定路徑)"""
    print("正在初始化 MetaTrader 5 (指定路徑模式)...")

    # ==============================================================================
    # ★★★ 核心修改點 ★★★
    # 已將您提供的路徑填入下方。
    # 這是您電腦上 terminal64.exe 的【完整】檔案路徑。
    # ==============================================================================
    mt5_path = r"C:\Users\10007793\AppData\Roaming\OANDA MetaTrader 5\terminal64.exe"

    # 在 initialize() 中加入 path 參數
    if not mt5.initialize(path=mt5_path):
        print(f"MT5 初始化失敗, 錯誤代碼 = {mt5.last_error()}")
        print("\n請檢查：")
        print("1. 上方指定的 mt5_path 路徑是否完全正確？（看起來是正確的）")
        print("2. OANDA MT5 桌面應用程式是否已開啟並成功登入？")
        print("3. 是否已用系統管理員身分執行 MT5 和 VS Code？")
        return False

    print("✅ MT5 初始化成功！")
    print(f"   版本: {mt5.version()}")
    print(f"   已連接到路徑: {mt5.terminal_info().path}")
    return True

def get_account_info():
    """獲取並顯示帳戶資訊"""
    account_info = mt5.account_info()
    if account_info is None:
        print("無法獲取帳戶資訊。")
        return
    info = account_info._asdict()
    print("\n--- 已連線的帳戶資訊 ---")
    if info['login'] == MT5_LOGIN:
        print(f"✅ 成功連接到預期的帳戶: {info['login']}")
    else:
        print(f"⚠️  警告：腳本連接到的帳戶 ({info['login']}) 與設定檔中的帳戶 ({MT5_LOGIN}) 不符！")
    for key, value in info.items():
        print(f"{key:<15}: {value}")
    print("------------------")

def shutdown_mt5():
    """關閉與 MetaTrader 5 的連接"""
    print("\n正在關閉 MT5 連接...")
    mt5.shutdown()
    print("MT5 連接已關閉。")


if __name__ == "__main__":
    if connect_to_mt5():
        get_account_info()
        shutdown_mt5()
