import requests

#  token(PRIVATE) và chat_id của bot(được tạo sẵn) trên telegram
BOT_TOKEN = ""
# https://api.telegram.org/bot[BOT_TOKEN]/getUpdates : dùng để lấy chat_id
# CHAT_ID = "-1002607649012" # Group chat đã thêm bot
CHAT_ID = "6139934019" # Chat trực tiếp với bot

def send_telegram():
    url_photo = f"https://api.telegram.org/bot[BOT_TOKEN]/sendPhoto"
 
    # tin nhắn văn bản
    message = "🚨🚨🚨 Có người xâm nhập!"

    # Gửi ảnh
    try:
        with open("Alert_nofications/alert.png", "rb") as photo:
            files = {"photo": photo}
            data = {"chat_id": CHAT_ID, "caption": message}
            requests.post(url_photo, data=data, files=files)
    except FileNotFoundError:
        print("❌ Không tìm thấy file alert.png để gửi.")