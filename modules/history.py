from datetime import datetime

def save_history(news, result, confidence, risk):

    with open("history.txt", "a", encoding="utf-8") as file:

        file.write("=" * 50 + "\n")
        file.write(f"Date : {datetime.now()}\n")
        file.write(f"News : {news}\n")
        file.write(f"Result : {result}\n")
        file.write(f"Confidence : {confidence:.2f}\n")
        file.write(f"Risk : {risk}\n\n")