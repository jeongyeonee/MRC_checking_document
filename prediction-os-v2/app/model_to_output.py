import argparse
import os
import pandas as pd
from utils import load_json

def pred_to_output(args):
    # Load prediction file
    pred = load_json(os.path.join(args.tmp_dir, args.prediction_file))

    # Load checklist file
    df_check = pd.read_csv(args.checklist_name)

    pred = pd.DataFrame(pred.items(), columns=['id','answer'])
    if args.processing:
        pred.answer = pred.answer.apply(lambda x: "" if len(x)<=3 else x)
    pred['question_id'] = pred.id.apply(lambda x:'_'.join(x.split('_')[-2:]))
    pred['관련 요구 사항'] = pred.id.apply(lambda x:x.split('_')[1])
    pred['판별 근거'] = pred.answer.apply(lambda x: "" if (len(x)<=3) or (x=="empty") else x)
    pred['answer'] = pred['판별 근거'].apply(lambda x: 1 if x!='' else 0)
    pred['관련 요구 사항'] = pred.apply(lambda x: "" if x['판별 근거']=="" else x['관련 요구 사항'], axis=1)
    pred = pred[['question_id','answer', '관련 요구 사항', '판별 근거']]

    output = pred.pivot_table(index='question_id', values='answer', aggfunc='sum').reset_index()
    output['answer'] = output.answer.apply(lambda x: 1 if x !=0 else 0)

    include = pd.merge(output[output.answer==1], pred, how='left', on=['question_id', 'answer'])
    not_include = output[output.answer==0]

    result = pd.concat([include, not_include]).sort_values(['question_id']).reset_index(drop=True)
    result = pd.merge(result, df_check[['question_id', 'sub_question']], on='question_id', how='left')
    
    save_name = os.path.join(args.output_dir, f"{args.doc_id}.csv")
    result.to_csv(save_name, index=False, encoding='utf-8')
    
    return save_name
