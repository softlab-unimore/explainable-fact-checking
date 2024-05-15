import json
import os

from explainable_fact_checking import FeverousModelAdapter
import explainable_fact_checking as xfc

if __name__ == "__main__":
    fc_model = FeverousModelAdapter()
    predictor = fc_model.predict
    AB_path = '/homes/bussotti/feverous_work/feverousdata/AB/'
    # input_file = os.path.join('/homes/bussotti/feverous_work/feverousdata/', 'feverous_dev_challenges_sentencesandtable.jsonl')
    # output_file = os.path.join(AB_path, 'feverous_dev_ST_01.jsonl')
    # input_file = '/homes/bussotti/feverous_work/feverousdata/feverous_dev_challenges_sentencesonly.jsonl'
    # output_file = os.path.join(AB_path, 'feverous_dev_SO_01.jsonl')
    for input_file_name in [
        'feverous_dev_ST_01.jsonl',
        'ex_AB_00.jsonl',
        'original_TO_01.jsonl',
        'feverous_dev_SO_01.jsonl'
    ]:
        input_file = os.path.join(AB_path, input_file_name)
        output_file = os.path.join(AB_path, input_file_name.replace('.jsonl', '_evidence_type_map.pickle'))
        with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
            record_list = [json.loads(line) for line in f_in]
            evidence_type_map = xfc.xfc_utils.map_evidence_types(record_list)
            f_out.write(json.dumps(evidence_type_map) + '\n')

        # output_file = os.path.join(AB_path, input_file_name.replace('.jsonl', '_pred_only_claim.json'))
        # # xfc.xfc_utils.save_prediciton_without_evidence(input_file_path, output_file=output_file, model=fc_model)

    # input_file = os.path.join(AB_path, 'original_TO_01.jsonl')
    # output_file = os.path.join(AB_path, 'original_TO_01_formatted.jsonl')
    # xfc.xfc_utils.AddInputTxtToUse.predict_and_save(input_file=input_file,
    #                                   output_file=output_file, model=fc_model)

    # launch a shell command to compress folders in the AB/lime_explanations folder
    # os.system(f'cd {AB_path} && tar -czf lime_explanations.tar.gz lime_explanations')
    # explain the parameters of tar
    # c: create a new archive
    # z: compress the archive with gzip
    # f: use the given archive file
