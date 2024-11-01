from langchain_core.runnables.config import RunnableConfig
# LOCAL IMPORTS.
from graph import create_graph, compile_graph, print_stream

# api handling imports.
# from fastapi import FastAPI,File, UploadFile
# from fastapi.responses import JSONResponse
# from fastapi.middleware.cors import CORSMiddleware
# import os
# from pathlib import Path
# from pypdf import PdfReader


# app = FastAPI()

# DATA_FOLDER = Path("testfolder")
# DATA_FOLDER.mkdir(exist_ok=True)

# @app.post("/upload")
# async def upload_file(file: UploadFile = File(...)):
#     if file.content_type != "application/pdf":
#         return JSONResponse(status_code=400, content={"message": "Invalid file type. Please upload a PDF."})

#     file_location = DATA_FOLDER / file.filename
#     with open(file_location, "wb") as f:
#         f.write(await file.read())

#     # Validate the PDF and read all pages
#     try:
#         with open(file_location, "rb") as pdf_file:
#             reader = PdfReader(pdf_file)
#             # Iterate through all pages to ensure the PDF is fully readable
#             for page_num, page in enumerate(reader.pages):
#                 text = page.extract_text()
#                 print(f"Text from page {page_num + 1}: {text}")
#     except Exception as e:
#         # Delete the file if it's not a valid PDF
#         os.remove(file_location)
#         return JSONResponse(status_code=400, content={"message": "The file is not a valid PDF."})

#     return JSONResponse(content={"message": "File uploaded and verified successfully!"})





config = RunnableConfig(recursion_limit=50)
print(config)

if __name__ == "__main__":
    # creating graph workflow instance and then compiling it.
    verbose = True
    builder = create_graph()
    graph = compile_graph(builder)

    # print the mermaid diagram of the graph.
    print(graph.get_graph().draw_mermaid())
    
    while True:
        user_input = input("############# User: ")
        print_stream(graph.stream({"messages": [("user", user_input)]}, stream_mode="values", config=config))
