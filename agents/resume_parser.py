import pdfplumber

def read_pdf(path):
       text = ""
       with pdfplumber.open(path) as pdf:
           for page in pdf.pages:
               text += (page.extract_text() or "") + "\n"
       return text

if __name__ == "__main__":
       text = read_pdf("data/sample.pdf")
       print(text)