import os
import sys
sys.path.append(os.path.dirname(__file__))
from fastapi import FastAPI
from run_korquality import main 
from pydantic import BaseModel

app = FastAPI()

class Item(BaseModel):
    doc_id: str

class Args:
    def __init__(self, doc_id):
        self.doc_id = doc_id
        self.input_dir = os.path.join(os.getenv("WORK_DIR"), "input")
        self.output_dir = os.path.join(os.getenv("WORK_DIR"), "output")
        self.meta_dir = os.path.join(os.getenv("WORK_DIR"), "meta")
        self.model_dir = os.path.join(os.getenv("WORK_DIR"), "kobert")
        self.topk_json_file = os.path.join(os.getenv("WORK_DIR"), "filter", f"{doc_id}.json")
        self.checklist_name = os.path.join(self.meta_dir, "checklist.csv")
        self.tmp_dir = os.path.join(os.getenv("WORK_DIR"), "tmp")
        if not os.path.exists(self.tmp_dir):
            os.mkdir(self.tmp_dir)
    
@app.post("/predict/")
async def run_all(item: Item):
    args = Args(item.doc_id)
    output_file_path = main(args)
    
    return {"doc_id": item.doc_id}