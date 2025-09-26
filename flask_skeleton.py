
from flask import Flask, jsonify, render_template_string

# 初始化 Flask 應用
app = Flask(__name__)

# --- 路由定義 ---

@app.route('/')
def index():
    """首頁路由，回傳一個簡單的 HTML 頁面。"""
    # 這是一個基礎的 HTML 模板，可以直接在 Python 檔案中編寫。
    # 在真實專案中，通常會使用 render_template() 函式來讀取獨立的 .html 檔案。
    html_template = """
    <!DOCTYPE html>
    <html lang="zh-TW">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Flask 骨架</title>
        <style>
            body { font-family: sans-serif; text-align: center; margin-top: 50px; }
            h1 { color: #333; }
            p { color: #666; }
            code { background-color: #f4f4f4; padding: 2px 6px; border-radius: 4px; }
        </style>
    </head>
    <body>
        <h1>歡迎來到 Flask 骨架應用！</h1>
        <p>這是一個最小化的 Flask 應用程式，可作為您開發的起點。</p>
        <p>嘗試訪問 <code>/api/hello</code> 來查看一個簡單的 JSON 回應。</p>
    </body>
    </html>
    """
    return render_template_string(html_template)

@app.route('/api/hello')
def hello_api():
    """一個簡單的 API 端點，回傳 JSON 格式的資料。"""
    # jsonify 會自動將 Python 的字典轉換為 JSON 格式的回應
    return jsonify({
        "message": "Hello, from the Flask API!",
        "status": "success"
    })

# --- 啟動伺服器 ---

# 確保這段程式碼只在直接執行此檔案時運行
# (而不是在被其他檔案 import 時運行)
if __name__ == '__main__':
    # app.run() 會啟動一個本地的開發伺服器
    # debug=True 會讓伺服器在程式碼變更後自動重啟，並在發生錯誤時顯示詳細的除錯訊息
    app.run(debug=True, port=5001) # 使用 5001 port 以免與其他應用衝突

