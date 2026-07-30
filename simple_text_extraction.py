import fitz

doc = fitz.open(r"C:\Users\tr-mo\Zotero\storage\FNRZ3I4J\Allamaprabhu et al. - 2011 - Improved Prediction of Flow Separation in Thrust Optimized Parabolic Nozzles with FLUENT.pdf")

text = ""

for page in doc:
    text += page.get_text()

print(text[:1000])