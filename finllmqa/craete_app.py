from finllmqa.api.app import *

if __name__ == '__main__':
    # # create llm app
    # # Load LLM
    # tokenizer = AutoTokenizer.from_pretrained(
    #     TOKENIZER_PATH, trust_remote_code=True)
    # model = AutoModel.from_pretrained(
    #     MODEL_PATH, trust_remote_code=True).cuda().eval()
    #
    # # load Embedding
    # embedding_model = SentenceTransformer(EMBEDDING_PATH, device="cuda")
    # uvicorn.run(llm_app, host='0.0.0.0', port=8000, workers=1)

    # create autogen app
    uvicorn.run(autogen_app, host='0.0.0.0', port=8006, workers=1)
