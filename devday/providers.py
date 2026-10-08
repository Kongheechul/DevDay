"""Only this boundary sends data externally. Replay makes no API calls."""
import base64
import json
import os
from pathlib import Path
from typing import Protocol
from .contracts import ROIPlan, Diagnosis

class ModelProvider(Protocol):
    mode: str
    def propose_rois(self, metadata, images, correction=None) -> ROIPlan: ...
    def diagnose(self, evidence, images) -> Diagnosis: ...

class OpenAIProvider:
    mode='openai_live'
    def __init__(self, model=None, client=None):
        self.model=model or os.environ.get('DEVDAY_MODEL','gpt-6-astra')
        if client is None:
            if not os.environ.get('OPENAI_API_KEY'): raise RuntimeError('Set OPENAI_API_KEY; use replay/demo for offline verification')
            from openai import OpenAI
            client=OpenAI(timeout=180,max_retries=0)
        self.client=client
        self.calls=[]

    def _call(self, stage, payload, images, schema):
        content=[{'type':'input_text','text':json.dumps(payload,ensure_ascii=False,allow_nan=False)}]
        for label,path in images:
            p=Path(path)
            mime='image/png' if p.suffix.lower()=='.png' else 'image/jpeg'
            content.extend([{'type':'input_text','text':label}, {'type':'input_image','image_url':f'data:{mime};base64,'+base64.b64encode(p.read_bytes()).decode(),'detail':'high'}])
        prompt=(Path(__file__).parent/'prompts'/f'{stage}.txt').read_text(encoding='utf-8')
        response=self.client.responses.parse(model=self.model,instructions=prompt,input=[{'role':'user','content':content}],text_format=schema,store=False,max_output_tokens=7000)
        usage=getattr(response,'usage',None)
        self.calls.append({'stage':stage,'response_id':getattr(response,'id',None),'model':self.model,'usage':usage.model_dump() if usage else None})
        if getattr(response,'status',None)!='completed' or response.output_parsed is None:
            raise RuntimeError(f'{stage} response refused, incomplete or unparseable; no measurement diagnosis substituted')
        return response.output_parsed

    def propose_rois(self, metadata, images, correction=None):
        return self._call('roi',{'videos':metadata,'coordinate_units':'normalized_0_to_1','correction':correction},images,ROIPlan)

    def diagnose(self, evidence, images):
        return self._call('diagnosis',evidence,images,Diagnosis)

class ReplayProvider:
    """Use recorded/fixture responses. Never presented as live model inference."""
    mode='offline_replay'
    def __init__(self, roi_json, diagnosis_json):
        self.roi_json=Path(roi_json);self.diagnosis_json=Path(diagnosis_json);self.calls=[]
    def propose_rois(self, metadata, images, correction=None):
        self.calls.append({'stage':'roi','source':'replay'})
        return ROIPlan.model_validate_json(self.roi_json.read_text(encoding='utf-8'))
    def diagnose(self, evidence, images):
        self.calls.append({'stage':'diagnosis','source':'replay'})
        return Diagnosis.model_validate_json(self.diagnosis_json.read_text(encoding='utf-8'))
