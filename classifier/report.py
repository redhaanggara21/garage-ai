from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

def generate_pdf_report(result, filename="training_report.pdf"):
    c = canvas.Canvas(filename, pagesize=letter)
    c.drawString(100, 750, "Training Report")
    c.drawString(100, 720, f"Final Training Accuracy: {result['train_acc']:.4f}")
    c.drawString(100, 700, f"Final Validation Accuracy: {result['val_acc']:.4f}")
    c.drawString(100, 680, f"Final Training Loss: {result['train_loss']:.4f}")
    c.drawString(100, 660, f"Final Validation Loss: {result['val_loss']:.4f}")
    c.save()
