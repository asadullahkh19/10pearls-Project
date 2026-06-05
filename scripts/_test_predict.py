import os, sys
sys.path.insert(0, os.getcwd())
from src.inference_pipeline.predict import predict_next_72h
import traceback

try:
    df = predict_next_72h('london')
    print('OK_ROWS', len(df))
    print(df.head().to_dict(orient='records'))
except Exception as e:
    print('ERROR', e)
    traceback.print_exc()
