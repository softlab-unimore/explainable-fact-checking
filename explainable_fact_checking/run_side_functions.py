import os

from explainable_fact_checking import FeverousModelAdapter, AddInputTxtToUse
from explainable_fact_checking import save_prediciton_without_evidence

if __name__ == "__main__":
    fc_model = FeverousModelAdapter()
    predictor = fc_model.predict
    AB_path = '/homes/bussotti/feverous_work/feverousdata/AB/'
    # input_file = os.path.join('/homes/bussotti/feverous_work/feverousdata/', 'feverous_dev_challenges_sentencesandtable.jsonl')
    # output_file = os.path.join(AB_path, 'feverous_dev_ST_01.jsonl')
    # input_file = '/homes/bussotti/feverous_work/feverousdata/feverous_dev_challenges_sentencesonly.jsonl'
    # output_file = os.path.join(AB_path, 'feverous_dev_SO_01.jsonl')
    for input_file_name in [

        'original_TO_01.jsonl',
        # 'ex_AB_00.jsonl',
        # 'feverous_dev_ST_01.jsonl',
        # 'feverous_dev_SO_01.jsonl'
    ]:
        input_file_path = os.path.join(AB_path, input_file_name)
        output_file = os.path.join(AB_path, input_file_name.replace('.jsonl', '_pred_only_claim.json'))
        save_prediciton_without_evidence(input_file_path, output_file=output_file, model=fc_model)

    # input_file = '/homes/bussotti/feverous_work/feverousdata/AB/ex_for_andrea_rightformat.jsonl'
    # output_file = os.path.join(AB_path, 'ex_AB_00.jsonl')
    # AddInputTxtToUse.predict_and_save(fc_model, input_file=input_file,
    #                                   output_file=output_file)

    # launch a shell command to compress folders in the AB/lime_explanations folder
    # os.system(f'cd {AB_path} && tar -czf lime_explanations.tar.gz lime_explanations')
    # explain the parameters of tar
    # c: create a new archive
    # z: compress the archive with gzip
    # f: use the given archive file
