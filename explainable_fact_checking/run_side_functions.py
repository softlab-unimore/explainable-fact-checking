import json
import os

from explainable_fact_checking import FeverousModelAdapter
import explainable_fact_checking as xfc

if __name__ == "__main__":
    fc_model = FeverousModelAdapter(model_path=xfc.experiment_definitions.C.JF_feverous_model['model_params']['model_path'][0])
    predictor = fc_model.predict
    AB_path = xfc.experiment_definitions.C.DATASET_DIR_FEVEROUS[0]


    # res_wrong = xfc.explanations_load.load_experiment_result_by_code('sk_f_jf_1.1', results_path=xfc.experiment_definitions.C.RESULTS_DIR)
    for input_file_name in [
        'feverous_train_challenges_withnoise.jsonl',
        #'feverous_dev_ST_01.jsonl',
        #'ex_AB_00.jsonl',
        #'original_TO_01.jsonl',
        #'feverous_dev_SO_01.jsonl'
    ]:
        input_file = os.path.join(AB_path, input_file_name)
        # output_file = os.path.join(AB_path, input_file_name.replace('.jsonl', '_evidence_type_map.pickle'))
        # with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        #     record_list = [json.loads(line) for line in f_in]
        #     evidence_type_map = xfc.xfc_utils.map_evidence_types(record_list)
        #     f_out.write(json.dumps(evidence_type_map) + '\n')

        # output_file = os.path.join(AB_path, input_file_name.replace('.jsonl', '_pred_only_claim.json'))
        # xfc.xfc_utils.save_prediciton_only_claim(input_file, output_file=output_file, model=fc_model)

        xfc.xfc_utils.AddInputTxtToUse.predict_and_save(input_file=input_file,
                                                        output_file=input_file, model=fc_model)

    # input_file = os.path.join(AB_path, 'original_TO_01.jsonl')
    # output_file = os.path.join(AB_path, 'original_TO_01_formatted.jsonl')

