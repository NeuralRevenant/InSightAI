import io
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import StreamingResponse
from PIL import Image
from ..agentic.mcploop import MCPLoop
from .schemas import VQAResponse
from ..config import HOST, PORT

app = FastAPI(title="InSightAI VQA")

mcp = MCPLoop()


@app.post("/vqa", response_model=VQAResponse)
async def vqa(question: str = Form(...), image: UploadFile = File(...)):
    img_bytes = await image.read()
    pil_img   = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    final, trace = mcp.infer(pil_img, question)
    return VQAResponse(
        answer   = final["answer"],
        reasoning= [t["dsl"] for t in trace]       # expose each OS query used
    )
