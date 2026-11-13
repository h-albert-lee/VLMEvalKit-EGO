"""Shared processor setup for preflight and GPU inference."""
from PIL import Image

def make_processor(modelinfo):
    from transformers import AutoProcessor
    processor=AutoProcessor.from_pretrained(modelinfo['id'],revision=modelinfo['revision'])
    if not getattr(processor,'chat_template',None):
        processor.chat_template=getattr(processor.tokenizer,'chat_template',None)
    if not processor.chat_template:
        raise ValueError('Pinned model has no processor or tokenizer chat template')
    return processor

def encode_input(processor,inp):
    images=[]
    for path in inp['images']:
        with Image.open(path) as image:images.append(image.convert('RGB'))
    messages=[{'role':'user','content':[{'type':'image','image':im} for im in images]+[{'type':'text','text':inp['prompt']}]}]
    # Passing unsupported template kwargs to older models routes them to the
    # image processor in Transformers 5.20. Only reasoning templates need this.
    kwargs={'enable_thinking':False} if 'enable_thinking' in str(processor.chat_template) else {}
    return processor.apply_chat_template(messages,add_generation_prompt=True,tokenize=True,return_dict=True,return_tensors='pt',**kwargs)
