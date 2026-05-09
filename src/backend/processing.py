from markitdown import MarkItDown

def file_converter(uploaded_files: list):
    converter = MarkItDown(enable_plugins=False)
    return [converter.convert(file) for file in uploaded_files]