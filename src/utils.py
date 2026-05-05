def generate_html_report():
    html = f"""
    <html>
    <head><title>Fraud Detection Report</title></head>
    <body>
        <h1>Fraud Detection System Report</h1>

        <h2>Confusion Matrix</h2>
        <img src="../images/confusion_matrix.png" width="400">

        <h2>Project Summary</h2>
        <p>This model detects fraudulent transactions using ML.</p>
    </body>
    </html>
    """

    with open("reports/report.html", "w") as f:
        f.write(html)