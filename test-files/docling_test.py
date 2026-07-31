from docling.document_converter import DocumentConverter

source = "/Users/zhue3/Documents/GitHub/sbml-file-generator-wizard/Il6_Manuscript.pdf"
converter = DocumentConverter()
doc = converter.convert(source).document
print(doc.export_to_markdown())
