from transformers import BitsAndBytesConfig
from langchain_huggingface import HuggingFacePipeline

def initiate_language_model_huggingface():
    print("Loading Language Model...")

    # quantization_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype="float16",
    #     bnb_4bit_use_double_quant=True,
    # )

    lm = HuggingFacePipeline.from_model_id(
        model_id="Qwen/Qwen2-7B-Instruct",
        task="text-generation",
        pipeline_kwargs=dict(
            max_new_tokens=512,
        ),
    )
    
    print("Language Model loaded successfully!")
    return lm  


lm = initiate_language_model_huggingface()