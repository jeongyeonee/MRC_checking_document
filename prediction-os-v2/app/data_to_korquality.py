import argparse
import os
import glob
import pandas as pd
import json
from utils import load_json


def data_to_input(args):
    f_json = load_json(args.topk_json_file)
    df_check = pd.read_csv(args.checklist_name)
    df_data = pd.DataFrame([load_json(d) for d in glob.glob(f'{args.input_dir}/*.json')])
    df_data.doc_id == df_data.doc_id.apply(str)
    df_data = df_data[df_data.doc_id==str(args.doc_id)].reset_index(drop=True)
    korq = {'data':[
                {'title':args.doc_id,
                }]
            }
    if 'doc_id' in f_json:
        del f_json['doc_id']

    req_qid = {}
    for k, vs in f_json.items():
        for v in vs:
            v = v.split('_')[-1].split('.')[0]
            req_qid[v] = req_qid.get(v, []) + [k]

    para = []
    for req, qid in req_qid.items():
        context = {'context_id':req}
        context['context'] = df_data[df_data.pid==req].context.values[0]
        context['qas'] = []
        for q in qid:
            qa = {}
            qa['id'] = f'{args.doc_id}_{req}_{q}'
            qa['question'] = df_check[df_check.question_id==q].sub_question.values[0]
            qa['answers'] = []
            context['qas'].append(qa)
        para.append(context)
    korq['data'][0]['paragraphs'] = para

    save_name = f"korquality_{args.doc_id}.json"
    with open(os.path.join(args.tmp_dir, save_name), "w", encoding="utf-8") as json_file:
        json.dump(korq, json_file, ensure_ascii=False, indent="\t")
        
    return save_name
