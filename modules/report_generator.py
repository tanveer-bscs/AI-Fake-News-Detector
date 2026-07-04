from datetime import datetime

def generate_report(news, result, confidence, risk):

    with open("reports/report.txt", "w", encoding="utf-8") as file:

        file.write("=" * 50 + "\n")
        file.write("AI Fake News Detection Report\n")
        file.write("=" * 50 + "\n\n")

        file.write(f"Date : {datetime.now()}\n\n")

        file.write("News:\n")
        file.write(news + "\n\n")

        file.write(f"Prediction : {result}\n")
        file.write(f"Confidence Score : {confidence:.2f}\n")
        file.write(f"Risk Level : {risk}\n")