import copy
import json
import os
import pickle
import jsonlines
import numpy as np
import itertools as it

from explainable_fact_checking.charts.explanations_loader import swapped_experiments
import explainable_fact_checking as xfc


if __name__ == '__main__':
    # SWAPPING THE ORDER OF THE CLASSES FOR SOME EXPERIMENTS THAT WERE TRAINED WITH A DIFFERENT ORDER
    for exp in swapped_experiments:
        # scan the directory. In each subfolder there is a json file with the params and a pkl file with the results
        # create a list of results containing in each item a pair of params and the results
        full_experiment_path = os.path.join(xfc.experiment_definitions.E.RESULTS_DIR, exp)
        old_path = full_experiment_path.replace(exp, exp + '_old')
        if os.path.exists(old_path):
            full_experiment_path = old_path
        else:
            # copy the full directory to a new directory with the suffix '_old'
            os.makedirs(old_path, exist_ok=True)
            os.system(f'cp -r {full_experiment_path}/* {old_path}')
            full_experiment_path = old_path

        assert pickle.load(open(list(os.scandir(list(os.scandir(old_path))[0]))[1], 'rb')) is not None, 'Results not loaded correctly'
        results_list = []
        for folder in os.scandir(full_experiment_path):
            if not folder.is_dir():
                continue
            params, results = None, None
            for file in os.scandir(folder):
                if file.name.endswith('.json'):
                    with open(file, 'r') as f:
                        params = json.load(f)
                elif file.name.endswith('.pkl'):
                    with open(file, 'rb') as f:
                        results = pickle.load(f)
            num_labels = params['model_params']['nb_label']
            if num_labels == 2:
                continue
            if num_labels == 3:
                universal_to_feverous = {1: 0, 2: 1, 0: 2}
                feverous_to_universal = {0: 1, 1: 2, 2: 0}

            elif num_labels == 2:
                universal_to_feverous = {i: i for i in range(7)}
                feverous_to_universal = {i: i for i in range(7)}
            else:
                universal_to_feverous = {1: 0, 2: 1, 0: 2} | {i: i for i in range(3, 7)}
                feverous_to_universal = {0: 1, 1: 2, 2: 0} | {i: i for i in range(3, 7)}
            if isinstance(results[0], dict):
                get_method = lambda exp_instance, key: exp_instance.get(key, None)
                set_method = lambda exp_instance, key, value: exp_instance.update({key: value})
            else:
                get_method = lambda exp_instance, key: getattr(exp_instance, key) if hasattr(exp_instance, key) else None
                set_method = lambda exp_instance, key, value: setattr(exp_instance, key, value)

            y_true = np.array([get_method(x, 'label') for x in results])
            y_pred_proba = np.array([get_method(x, 'predict_proba') for x in results])
            y_pred_int = np.argmax(y_pred_proba, axis=1)
            # all order permutations of the classes
            order_permutation = list(it.permutations(range(num_labels)))
            acc_dict = {}
            for order in order_permutation:
                y_pred_int_v2 = np.argmax(y_pred_proba[:, order], axis=1)
                acc = np.mean(y_true == y_pred_int_v2)
                acc_dict[order] = acc
            order_v1 = [universal_to_feverous[universal_to_feverous[i]] for i in range(num_labels)]
            order_v2 = [universal_to_feverous[i] for i in range(num_labels)]
            # order with max accuracy
            order = max(acc_dict, key=acc_dict.get)

            new_results = copy.deepcopy(results)
            for i, x in enumerate(new_results):
                xold = results[i]
                set_method(x, 'predict_proba', get_method(xold, 'predict_proba')[[order]][0])

                old_v = get_method(xold, 'intercept')
                new_intercept = {new_class: old_v[old_idx] for new_class, old_idx in enumerate(order)}
                set_method(x, 'intercept', new_intercept)

                old_v = get_method(xold, 'local_exp')
                new_local_exp = {new_class: old_v[old_idx] for new_class, old_idx in enumerate(order)}
                set_method(x, 'local_exp', new_local_exp)
            new_acc = np.mean(y_true == np.argmax(np.array([get_method(x, 'predict_proba') for x in new_results]), axis=1))
            if list(order) not in [order_v1, order_v2]:
                if order == (0, 1, 2):
                    print('Order already OK')
                    continue
                raise ValueError('Order not in the possible orders')

            assert new_acc == acc_dict[order], 'Accuracy not the same'
            # save the new results in a new folder with the same name as the original folder with the suffix '_swapped'
            new_folder = folder.path.replace('_old', '').replace(exp, exp + '_swapped')
            # delete new folder if it already exists
            if os.path.exists(new_folder):
                try:
                    for file in os.scandir(new_folder):
                        os.remove(file)
                    os.rmdir(new_folder, )
                except Exception as e:
                    print(f'Error deleting folder: {e}')
            assert not os.path.exists(new_folder), 'Folder not deleted correctly'

            new_folder = folder.path.replace('_old', '')
            os.makedirs(new_folder, exist_ok=True)
            with open(os.path.join(new_folder, 'params.json'), 'w') as f:
                json.dump(params, f)
            with open(os.path.join(new_folder, 'results.pkl'), 'wb') as f:
                pickle.dump(new_results, f)


# t = record.copy()
# del t['input_txt_to_use']
# predictor([t])


# [l for l in jsonlines.open(data_path + "/" + dev_test, "r")]


# Get length of the list of dictionaries
# [len(x['input_txt_to_use'].split('</s>'))-1 for x in restructured_records]

def convert_list_to_string(input_list):
    # Join the list elements into a single string with spaces in between
    return ' '.join(input_list)


# Example usage
input_list = [
    "Accompanying an array of images in a recent Facebook post , including photos of civil court filings, screenshots of tweets, and a picture of U.S. House Speaker Nancy Pelosi, is one explosive allegation: that the California Democrat was arrested.",
    "This post was flagged as part of Facebook\\u2019s efforts to combat false news and misinformation on its News Feed.",
    "(Read more about our partnership with Facebook .)",
    "There are screenshots of several tweets from one account in the Facebook post, including these: \"BREAKING While leaving the US Capitol nancy Pelosi has been intercepted by US Marshals and ARRESTED (!)",
    "under the direct order of President Trump.",
    "WH claims ripping up speech was violation of decorum provision of Rule XIX!\"",
    "\"Marshals holding Pelosi are NOT(!)",
    "allowing her a phone call or to contact her attorneys.",
    "Chuck Schumer is racing back to the US capitol.",
    "White House sources say that she will be transported to undisclosed location TONIGHT (!)",
    "to face MILITARY TRIBUNAL (!!!)\"",
    "First of all, Pelosi didn\\u2019t break the law when she tore up a copy of President Donald Trump\\u2019s State of the Union address on Feb. 4.",
    "Trump , among others , wrongly claimed it was illegal.",
    "And while some people have signed a Change.org petition calling for her arrest after she ripped up her copy of the president\\u2019s speech, no arrest has happened.",
    "We found no news coverage online about Pelosi being taken into custody.",
    "If she had been arrested, it would have drawn plenty of media attention."]

result = convert_list_to_string(input_list)
print(result)

{"evidence": [{"content": ["Algebraic logic_sentence_0", "Lindenbaum\u2013Tarski algebra_sentence_1",
                           "Lindenbaum\u2013Tarski algebra_sentence_6", "Lindenbaum\u2013Tarski algebra_sentence_3",
                           "Algebraic logic_cell_0_1_1", "Algebraic logic_cell_0_2_1", "Algebraic logic_cell_0_3_1",
                           "Algebraic logic_cell_0_4_1", "Algebraic logic_cell_0_5_1", "Algebraic logic_cell_0_6_1",
                           "Algebraic logic_cell_0_7_1", "Algebraic logic_cell_0_8_1", "Algebraic logic_cell_0_9_1"],
               "context": {"Algebraic logic_sentence_0": ["Algebraic logic_title"],
                           "Lindenbaum\u2013Tarski algebra_sentence_1": ["Lindenbaum\u2013Tarski algebra_title"],
                           "Lindenbaum\u2013Tarski algebra_sentence_6": ["Lindenbaum\u2013Tarski algebra_title"],
                           "Lindenbaum\u2013Tarski algebra_sentence_3": ["Lindenbaum\u2013Tarski algebra_title"],
                           "Algebraic logic_cell_0_1_1": ["Algebraic logic_title", "Algebraic logic_section_4",
                                                          "Algebraic logic_header_cell_0_0_1"],
                           "Algebraic logic_cell_0_2_1": ["Algebraic logic_title", "Algebraic logic_section_4",
                                                          "Algebraic logic_header_cell_0_0_1"],
                           "Algebraic logic_cell_0_3_1": ["Algebraic logic_title", "Algebraic logic_section_4",
                                                          "Algebraic logic_header_cell_0_0_1"],
                           "Algebraic logic_cell_0_4_1": ["Algebraic logic_title", "Algebraic logic_section_4",
                                                          "Algebraic logic_header_cell_0_0_1"],
                           "Algebraic logic_cell_0_5_1": ["Algebraic logic_title", "Algebraic logic_section_4",
                                                          "Algebraic logic_header_cell_0_0_1"],
                           "Algebraic logic_cell_0_6_1": ["Algebraic logic_title", "Algebraic logic_section_4",
                                                          "Algebraic logic_header_cell_0_0_1"],
                           "Algebraic logic_cell_0_7_1": ["Algebraic logic_title", "Algebraic logic_section_4",
                                                          "Algebraic logic_header_cell_0_0_1"],
                           "Algebraic logic_cell_0_8_1": ["Algebraic logic_title", "Algebraic logic_section_4",
                                                          "Algebraic logic_header_cell_0_0_1"],
                           "Algebraic logic_cell_0_9_1": ["Algebraic logic_title", "Algebraic logic_section_4",
                                                          "Algebraic logic_header_cell_0_0_1"]}}, {
                  "content": ["Algebraic logic_cell_0_1_0", "Algebraic logic_cell_0_2_0", "Algebraic logic_cell_0_3_0",
                              "Algebraic logic_cell_0_4_0", "Algebraic logic_cell_0_5_0", "Algebraic logic_cell_0_6_0",
                              "Algebraic logic_cell_0_7_0", "Algebraic logic_cell_0_8_0", "Algebraic logic_cell_0_9_0"],
                  "context": {"Algebraic logic_cell_0_1_0": ["Algebraic logic_title", "Algebraic logic_section_4",
                                                             "Algebraic logic_header_cell_0_0_0"],
                              "Algebraic logic_cell_0_2_0": ["Algebraic logic_title", "Algebraic logic_section_4",
                                                             "Algebraic logic_header_cell_0_0_0"],
                              "Algebraic logic_cell_0_3_0": ["Algebraic logic_title", "Algebraic logic_section_4",
                                                             "Algebraic logic_header_cell_0_0_0"],
                              "Algebraic logic_cell_0_4_0": ["Algebraic logic_title", "Algebraic logic_section_4",
                                                             "Algebraic logic_header_cell_0_0_0"],
                              "Algebraic logic_cell_0_5_0": ["Algebraic logic_title", "Algebraic logic_section_4",
                                                             "Algebraic logic_header_cell_0_0_0"],
                              "Algebraic logic_cell_0_6_0": ["Algebraic logic_title", "Algebraic logic_section_4",
                                                             "Algebraic logic_header_cell_0_0_0"],
                              "Algebraic logic_cell_0_7_0": ["Algebraic logic_title", "Algebraic logic_section_4",
                                                             "Algebraic logic_header_cell_0_0_0"],
                              "Algebraic logic_cell_0_8_0": ["Algebraic logic_title", "Algebraic logic_section_4",
                                                             "Algebraic logic_header_cell_0_0_0"],
                              "Algebraic logic_cell_0_9_0": ["Algebraic logic_title", "Algebraic logic_section_4",
                                                             "Algebraic logic_header_cell_0_0_0"]}}], "id": 7389,
 "claim": "Algebraic logic has five Logical system and Lindenbaum\u2013Tarski algebra which includes Physics algebra and Nodal algebra (provide models of propositional modal logics).",
 "label": "REFUTES", "annotator_operations": [{"operation": "start", "value": "start", "time": "0"},
                                              {"operation": "Now on", "value": "?search=", "time": "1.62"},
                                              {"operation": "search", "value": "Algebra Logic", "time": "55.915"},
                                              {"operation": "Now on", "value": "Algebra i Logika", "time": "58.458"},
                                              {"operation": "search", "value": "algebra logical system",
                                               "time": "75.851"},
                                              {"operation": "Now on", "value": "undefined", "time": "77.252"},
                                              {"operation": "search", "value": "Lindenbaum", "time": "95.919"},
                                              {"operation": "Now on", "value": "Lindenbaum", "time": "96.709"},
                                              {"operation": "search", "value": "lindenbaaum-tarski", "time": "112.075"},
                                              {"operation": "Now on", "value": "undefined", "time": "113"},
                                              {"operation": "hyperlink", "value": "Algebraic logic", "time": "126.503"},
                                              {"operation": "Now on", "value": "Algebraic logic", "time": "127.87"},
                                              {"operation": "Highlighting", "value": "Algebraic logic_sentence_0",
                                               "time": "139.588"},
                                              {"operation": "Page search", "value": "logical system", "time": "177.52"},
                                              {"operation": "hyperlink", "value": "Lindenbaum\u2013Tarski algebra",
                                               "time": "198.201"},
                                              {"operation": "Now on", "value": "Lindenbaum\u2013Tarski algebra",
                                               "time": "198.831"}, {"operation": "Highlighting",
                                                                    "value": "Lindenbaum\u2013Tarski algebra_sentence_1",
                                                                    "time": "229.376"}, {"operation": "Highlighting",
                                                                                         "value": "Lindenbaum\u2013Tarski algebra_sentence_6",
                                                                                         "time": "243.39"},
                                              {"operation": "Highlighting",
                                               "value": "Lindenbaum\u2013Tarski algebra_sentence_3", "time": "254.119"},
                                              {"operation": "back-button-clicked", "value": "back-button-clicked",
                                               "time": "309.012"},
                                              {"operation": "Now on", "value": "Algebraic logic", "time": "309.107"},
                                              {"operation": "search", "value": "five", "time": "319.153"},
                                              {"operation": "Now on", "value": "5 (disambiguation)", "time": "320.853"},
                                              {"operation": "back-button-clicked", "value": "back-button-clicked",
                                               "time": "326.901"},
                                              {"operation": "Now on", "value": "Algebraic logic", "time": "326.989"},
                                              {"operation": "Page search", "value": "five", "time": "331.075"},
                                              {"operation": "hyperlink", "value": "sentential logic",
                                               "time": "369.843"},
                                              {"operation": "Now on", "value": "Propositional calculus",
                                               "time": "370.894"},
                                              {"operation": "back-button-clicked", "value": "back-button-clicked",
                                               "time": "372.489"},
                                              {"operation": "Now on", "value": "Algebraic logic", "time": "372.578"},
                                              {"operation": "Highlighting", "value": "Algebraic logic_cell_0_1_0",
                                               "time": "374.293"},
                                              {"operation": "Highlighting", "value": "Algebraic logic_cell_0_2_0",
                                               "time": "375.079"},
                                              {"operation": "Highlighting", "value": "Algebraic logic_cell_0_3_0",
                                               "time": "375.978"},
                                              {"operation": "Highlighting", "value": "Algebraic logic_cell_0_4_0",
                                               "time": "376.906"},
                                              {"operation": "Highlighting", "value": "Algebraic logic_cell_0_5_0",
                                               "time": "377.955"},
                                              {"operation": "Highlighting", "value": "Algebraic logic_cell_0_6_0",
                                               "time": "378.868"},
                                              {"operation": "Highlighting", "value": "Algebraic logic_cell_0_7_0",
                                               "time": "379.584"},
                                              {"operation": "Highlighting", "value": "Algebraic logic_cell_0_8_0",
                                               "time": "380.466"},
                                              {"operation": "Highlighting", "value": "Algebraic logic_cell_0_9_0",
                                               "time": "381.396"},
                                              {"operation": "Highlighting", "value": "Algebraic logic_cell_0_1_1",
                                               "time": "440.289"},
                                              {"operation": "Highlighting", "value": "Algebraic logic_cell_0_2_1",
                                               "time": "440.919"},
                                              {"operation": "Highlighting", "value": "Algebraic logic_cell_0_3_1",
                                               "time": "441.889"},
                                              {"operation": "Highlighting", "value": "Algebraic logic_cell_0_4_1",
                                               "time": "442.479"},
                                              {"operation": "Highlighting", "value": "Algebraic logic_cell_0_5_1",
                                               "time": "443.149"},
                                              {"operation": "Highlighting", "value": "Algebraic logic_cell_0_6_1",
                                               "time": "444.967"},
                                              {"operation": "Highlighting", "value": "Algebraic logic_cell_0_7_1",
                                               "time": "446.181"},
                                              {"operation": "Highlighting", "value": "Algebraic logic_cell_0_8_1",
                                               "time": "447.299"},
                                              {"operation": "Highlighting", "value": "Algebraic logic_cell_0_9_1",
                                               "time": "449.401"},
                                              {"operation": "finish", "value": "finish", "time": "559.284"}],
 "challenge": "Multi-hop Reasoning",
 "input_txt_model": "Algebraic logic has five Logical system and Lindenbaum\u2013Tarski algebra which includes Physics algebra and Nodal algebra (provide models of propositional modal logics). </s>  In [[Mathematical_logic|mathematical logic]], algebraic logic is the reasoning obtained by manipulating equations with free variables. </s> Algebraic logic </s> [[Lindenbaum\u2013Tarski_algebra|Lindenbaum\u2013Tarski algebra]] is [[Modal_algebra|Modal algebra]] ; [[Lindenbaum\u2013Tarski_algebra|Lindenbaum\u2013Tarski algebra]] is [[Interior_algebra|Interior algebra]] ; [[Lindenbaum\u2013Tarski_algebra|Lindenbaum\u2013Tarski algebra]] is [[MV-algebra|MV-algebra]] ; [[Lindenbaum\u2013Tarski_algebra|Lindenbaum\u2013Tarski algebra]] is [[Cylindric_algebra|Cylindric algebra]] ; [[Lindenbaum\u2013Tarski_algebra|Lindenbaum\u2013Tarski algebra]] is [[Boolean_algebra|Boolean algebra]] ; [[Lindenbaum\u2013Tarski_algebra|Lindenbaum\u2013Tarski algebra]] is [[Heyting_algebra|Heyting algebra]] ; [[Lindenbaum\u2013Tarski_algebra|Lindenbaum\u2013Tarski algebra]] is [[Boolean-valued_model|Complete Boolean algebra]], [[Polyadic_algebra|polyadic algebra]], [[Predicate_functor_logic|predicate functor logic]] ; [[Lindenbaum\u2013Tarski_algebra|Lindenbaum\u2013Tarski algebra]] is [[Monadic_Boolean_algebra|Monadic Boolean algebra]] ; [[Lindenbaum\u2013Tarski_algebra|Lindenbaum\u2013Tarski algebra]] is [[Combinatory_logic|Combinatory logic]], [[Relation_algebra|relation algebra]]. </s> Logical system is [[Intuitionistic_logic|Intuitionistic]] propositional logic ; Logical system is First-order logic with [[Equality_(mathematics)|equality]] ; Logical system is [[First-order_logic|First-order logic]] ; Logical system is [[Clarence_Irving_Lewis|Lewis]]'s [[Modal_logic|S4]] ; Logical system is [[Set_theory|Set theory]] ; Logical system is [[\u0141ukasiewicz_logic|\u0141ukasiewicz logic]] ; Logical system is Lewis's [[S5_(modal_logic)|S5]], [[Monadic_predicate_logic|monadic predicate logic]] ; Logical system is Classical [[Sentential_logic|sentential logic]] ; Logical system is Modal logic [[Normal_modal_logic|K]]. </s>  In [[Mathematical_logic|mathematical logic]], the Lindenbaum\u2013Tarski algebra (or Lindenbaum algebra) of a [[Model_theory#Definition|logical theory]] T consists of the [[Equivalence_class|equivalence classes]] of [[Sentence_(mathematical_logic)|sentences]] of the theory (i.e., the [[Quotient|quotient]], under the [[Equivalence_relation|equivalence relation]] ~ defined such that p ~ q exactly when p and q are provably equivalent in T). </s>  The Lindenbaum\u2013Tarski algebra is considered the origin of the modern [[Algebraic_logic|algebraic logic]]. </s>  The Lindenbaum\u2013Tarski algebra is thus the [[Quotient_(universal_algebra)|quotient algebra]] obtained by factoring the algebra of formulas by this congruence relation.",
 "input_txt_to_use": "Algebraic logic has five Logical system and Lindenbaum\u2013Tarski algebra which includes Physics algebra and Nodal algebra (provide models of propositional modal logics). </s>  In [[Mathematical_logic|mathematical logic]], algebraic logic is the reasoning obtained by manipulating equations with free variables. </s> Algebraic logic </s> [[Lindenbaum\u2013Tarski_algebra|Lindenbaum\u2013Tarski algebra]] is [[Modal_algebra|Modal algebra]] ; [[Lindenbaum\u2013Tarski_algebra|Lindenbaum\u2013Tarski algebra]] is [[Interior_algebra|Interior algebra]] ; [[Lindenbaum\u2013Tarski_algebra|Lindenbaum\u2013Tarski algebra]] is [[MV-algebra|MV-algebra]] ; [[Lindenbaum\u2013Tarski_algebra|Lindenbaum\u2013Tarski algebra]] is [[Cylindric_algebra|Cylindric algebra]] ; [[Lindenbaum\u2013Tarski_algebra|Lindenbaum\u2013Tarski algebra]] is [[Boolean_algebra|Boolean algebra]] ; [[Lindenbaum\u2013Tarski_algebra|Lindenbaum\u2013Tarski algebra]] is [[Heyting_algebra|Heyting algebra]] ; [[Lindenbaum\u2013Tarski_algebra|Lindenbaum\u2013Tarski algebra]] is [[Boolean-valued_model|Complete Boolean algebra]], [[Polyadic_algebra|polyadic algebra]], [[Predicate_functor_logic|predicate functor logic]] ; [[Lindenbaum\u2013Tarski_algebra|Lindenbaum\u2013Tarski algebra]] is [[Monadic_Boolean_algebra|Monadic Boolean algebra]] ; [[Lindenbaum\u2013Tarski_algebra|Lindenbaum\u2013Tarski algebra]] is [[Combinatory_logic|Combinatory logic]], [[Relation_algebra|relation algebra]]. </s> Logical system is [[Intuitionistic_logic|Intuitionistic]] propositional logic ; Logical system is First-order logic with [[Equality_(mathematics)|equality]] ; Logical system is [[First-order_logic|First-order logic]] ; Logical system is [[Clarence_Irving_Lewis|Lewis]]'s [[Modal_logic|S4]] ; Logical system is [[Set_theory|Set theory]] ; Logical system is [[\u0141ukasiewicz_logic|\u0141ukasiewicz logic]] ; Logical system is Lewis's [[S5_(modal_logic)|S5]], [[Monadic_predicate_logic|monadic predicate logic]] ; Logical system is Classical [[Sentential_logic|sentential logic]] ; Logical system is Modal logic [[Normal_modal_logic|K]]. </s>  In [[Mathematical_logic|mathematical logic]], the Lindenbaum\u2013Tarski algebra (or Lindenbaum algebra) of a [[Model_theory#Definition|logical theory]] T consists of the [[Equivalence_class|equivalence classes]] of [[Sentence_(mathematical_logic)|sentences]] of the theory (i.e., the [[Quotient|quotient]], under the [[Equivalence_relation|equivalence relation]] ~ defined such that p ~ q exactly when p and q are provably equivalent in T). </s>  The Lindenbaum\u2013Tarski algebra is considered the origin of the modern [[Algebraic_logic|algebraic logic]]. </s>  The Lindenbaum\u2013Tarski algebra is thus the [[Quotient_(universal_algebra)|quotient algebra]] obtained by factoring the algebra of formulas by this congruence relation."}

{"claim_id": 0, "claim": "In a letter to Steve Jobs, Sean Connery refused to appear in an apple commercial.",
 "bm25_qau": [["Did Sean connery send a fake letter about real Steve jobs?",
               "Also, fake Sean Connery sent a letter to Real Steve Jobs.",
               "https://www.nbcnews.com/news/world/pre-caffeine-tech-apple-gossip-smart-pugs-flna122578"],
              ["What is the content of this letter?",
               "This is a letter Sean Connery wrote didn't write in response to Steve Jobs after being asked to appear in an Apple ad.",
               "https://www.businessinsider.com/james-bond-sean-connery-steve-jobs-apple-letter-2011-6"],
              ["What is the name of a fake letter that became a top twitter trend in 2020?",
               "Fake Sean Connery / Steve Jobs letter becomes top Twitter trending topic",
               "https://www.mi6-hq.com/news/index.php?itemid=9532"], ["Was it a real letter?",
                                                                      "Hilarious, though fictional, was this letter from Sean Connery to Steve Jobs released this morning on Scoopertino.",
                                                                      "https://www.splasmata.com/?m=201106"],
              ["Did Steve Job send a letter of refusal to a movie called James Bond?",
               "First, the bad news. Sean Connery never actually sent a typewritten letter to Steve Jobs in 1998 refusing to be in an Apple ad.",
               "https://www.cnet.com/culture/fake-sean-connery-letter-to-steve-jobs-goes-viral/"],
              ["Did Connery's letter to Jobs include a threat of legal action against Apple?",
               "Pingback: Did Sean Connery Write an Angry Letter to Steve Jobs? | wafflesatnoon.com",
               "https://web.archive.org/web/20201129141238/https://scoopertino.com/exposed-the-imac-disaster-that-almost-was/"],
              ["Did Connery's letter to Jobs include a threat of legal action against Apple?",
               "Pingback: Did Sean Connery Write an Angry Letter to Steve Jobs? | wafflesatnoon.com",
               "https://scoopertino.com/exposed-the-imac-disaster-that-almost-was/"],
              ["What is the difference between a university and a college of law?",
               "Pingback: Carta de Sean Connery a Steve Jobs — Tecnoculto",
               "https://web.archive.org/web/20201129141238/https://scoopertino.com/exposed-the-imac-disaster-that-almost-was/"],
              ["What is the difference between a university and a college of law?",
               "Pingback: Carta de Sean Connery a Steve Jobs — Tecnoculto",
               "https://scoopertino.com/exposed-the-imac-disaster-that-almost-was/"], [
                  "What is the name of a fictional character from the movie  'James Bond'  Question answer:  Question answer: Is there a name for a person who is not a member of parliament but who has a seat in parliament?",
                  "'I am f****** James Bond': Sean Connery letter to Steve Jobs rejecting offer to appear in Apple ad revealed to be fake",
                  "https://www.dailymail.co.uk/news/article-2006317/Sean-Connery-letter-Steve-Jobs-rejecting-offer-appear-Apple-ad-revealed-fake.html"],
              ["What is the name of the film?", "Sean Connery eating an apple on set of film Highlander in costume",
               "https://www.mediastorehouse.com/memory-lane-prints/mirror/0000to0099-00029/sean-connery-eating-apple-set-film-highlander-21300411.html"],
              ["What is the name of the film?", "Sean Connery eating an apple on set of film Highlander in costume",
               "https://www.mediastorehouse.co.uk/memory-lane-prints/mirror/0000to0099-00029/sean-connery-eating-apple-set-film-highlander-21300411.html"],
              [
                  "Was the alternative statement made by - Lebron James that Duterte was \"bigger\" in dealing with the coronavirus crisis than President Trump?",
                  "私のジェームズ・ボンド好きとアップル好きのせいで、私のメール箱は、1998 年に Sean Connery から Steve Jobs に送られたという Scoopertino の偽手紙に関するメールで溢れかえっている。これはこういうことだ。この手紙の画像コピーがインターネットを野火の如く流布している。しかし Scoopertino へのリンクは張られていない。ということはなりすましのイタズラはうまくいかなかったということだ。",
                  "https://maclalala2.wordpress.com/2011/06/24/%E3%81%9F%E3%81%8B%E3%81%8C%E3%82%B3%E3%83%B3%E3%83%94%E3%83%A5%E3%83%BC%E3%82%BF%E3%82%BB%E3%83%BC%E3%83%AB%E3%82%B9%E3%83%9E%E3%83%B3%E3%81%AE%E3%81%9F%E3%82%81%E3%81%AB%E3%82%B8%E3%82%A7%E3%83%BC/"],
              ["What is the total revenue for the year 2019?",
               "Pingback: JamesBondAuction.co.uk – James Bond 007 » Archive » Fake Sean Connery / Steve Jobs letter becomes top Twitter trending topic",
               "https://scoopertino.com/exposed-the-imac-disaster-that-almost-was/"],
              ["What is the total revenue for the year 2019?",
               "Pingback: JamesBondAuction.co.uk – James Bond 007 » Archive » Fake Sean Connery / Steve Jobs letter becomes top Twitter trending topic",
               "https://web.archive.org/web/20201129141238/https://scoopertino.com/exposed-the-imac-disaster-that-almost-was/"],
              ["Was it a costume apple or a real apple?",
               "Sean Connery eating an apple on set of film Highlander in costume Circa May 1985",
               "https://www.mediastorehouse.co.uk/memory-lane-prints/mirror/0000to0099-00029/sean-connery-eating-apple-set-film-highlander-21300411.html"],
              ["Was it a costume apple or a real apple?",
               "Sean Connery eating an apple on set of film Highlander in costume Circa May 1985",
               "https://www.mediastorehouse.co.uk/memory-lane-prints/mirror/0000to0099-00029/sean-connery-eating-apple-set-film-highlander-21300411.html"],
              ["Was it a costume apple or a real apple?",
               "Sean Connery eating an apple on set of film Highlander in costume Circa May 1985",
               "https://www.mediastorehouse.com/memory-lane-prints/mirror/0000to0099-00029/sean-connery-eating-apple-set-film-highlander-21300411.html"],
              ["Was it a costume apple or a real apple?",
               "Sean Connery eating an apple on set of film Highlander in costume Circa May 1985",
               "https://www.mediastorehouse.com/memory-lane-prints/mirror/0000to0099-00029/sean-connery-eating-apple-set-film-highlander-21300411.html"],
              ["What is the difference between a gun and a weapon?",
               "Pingback: Sean Connery writes Steve Jobs. - Science Fiction Fantasy Chronicles: forums",
               "https://scoopertino.com/exposed-the-imac-disaster-that-almost-was/"],
              ["What is the difference between a gun and a weapon?",
               "Pingback: Sean Connery writes Steve Jobs. - Science Fiction Fantasy Chronicles: forums",
               "https://web.archive.org/web/20201129141238/https://scoopertino.com/exposed-the-imac-disaster-that-almost-was/"],
              ["What is the real reason why Apple CEO Steve Job refused to work with James bond star Sean connery?",
               "Do you know who I am? A faked letter from James Bond star Sir Sean Connery firmly rejected an apparent advertising role from Apple chief Steve Jobs",
               "https://www.dailymail.co.uk/news/article-2006317/Sean-Connery-letter-Steve-Jobs-rejecting-offer-appear-Apple-ad-revealed-fake.html"],
              ["Is the letter real?",
               "An image of a purported 1998 letter from actor Sean Connery (famous for his portrayal of agent James Bond) to Apple CEO Steve Jobs, caustically rebuffing an offer to become a pitchman for Apple Computers, hit the Internet in June 2011.",
               "https://www.snopes.com/fact-check/false-sean-connery-letter-to-apple/"],
              ["Who was the last person to have an interview published?",
               "Ernest Hemingway, Sean Connery, Sigmund Freud, Steve Jobs, Padre Pio, Van Gogh, Giuseppe Verdi, George Clooney, Lenin, Cavour, Garibaldi…",
               "https://www.isupportstreetart.com/sigmund-freud-test-of-personality-by-chekos-opiemme/"],
              ["What does this interview have to with the question asked?",
               "Steve Jobs in an Interview with Fortune Magazine, 2000",
               "https://www.stephenfry.com/2011/10/steve-jobs/"],
              ["Who is the person who tweeted the letter from Steve Job to Sean Connnery?",
               "A 'letter' from Sean Connery to Steve Jobs was a top trending topic on Twitter today thanks to an unwitting tweet from a marketing executive who thought it was genuine.",
               "https://www.mi6-hq.com/news/index.php?itemid=9532"], [
                  "Did President Trump tell the truth about the fake letter to Apple CEO Steve Job Jobs Jobs  Question answer:  Yes, Trump said it was fake.?",
                  "Thanks to the confluence of my interests and the fact that it’s funny as hell, I’ve been inundated with email regarding Scoopertino’s fake 1998 letter from Sean Connery to Steve Jobs.",
                  "https://maclalala2.wordpress.com/2011/06/24/%E3%81%9F%E3%81%8B%E3%81%8C%E3%82%B3%E3%83%B3%E3%83%94%E3%83%A5%E3%83%BC%E3%82%BF%E3%82%BB%E3%83%BC%E3%83%AB%E3%82%B9%E3%83%9E%E3%83%B3%E3%81%AE%E3%81%9F%E3%82%81%E3%81%AB%E3%82%B8%E3%82%A7%E3%83%BC/"],
              ["What was Congressman Jason Kucinich's name before he became a congressman in 2016?",
               "In 1996, she married Jason Connery, son of Sean Connery, with whom she performed in Bullet to Beijing (1995).",
               "https://www.imdb.com/search/name/?birth_year=1967"], ["Is this a true story or a fake story?",
                                                                      "Another added: 'Sean Connery's letter to Steve Jobs? Well done satire site Scoopertino for fooling so many tweeps .'",
                                                                      "https://www.dailymail.co.uk/news/article-2006317/Sean-Connery-letter-Steve-Jobs-rejecting-offer-appear-Apple-ad-revealed-fake.html"],
              ["Is it appropriate to wear a kilt in a formal setting?",
               "Kellie Pickler, Sean Connery, Others Weigh In On Proper Kilt Etiquette",
               "https://www.mtv.com/news/mhhftp/kellie-pickler-sean-connery-others-weigh-in-on-proper-kilt-etiquette"],
              ["Did Sean connery denounce racism in the movie \"Dr. No\"?",
               "Anyway, here is a link to Sean Connery in his own words.....",
               "https://www.mumsnet.com/Talk/_chat/4066060-Sean-Connery-is-not-a-legend"],
              ["Was there any evidence that Jobs was aware of any of this before he agreed to take part in it?",
               "Though Steve had a thing for Sean Connery, the feeling was not mutual. Connery was appalled by the “advert” Jobs sent across the pond and declined to participate in the misadventure on at least three separate occasions.",
               "https://maclalala2.wordpress.com/2011/06/24/%E3%81%9F%E3%81%8B%E3%81%8C%E3%82%B3%E3%83%B3%E3%83%94%E3%83%A5%E3%83%BC%E3%82%BF%E3%82%BB%E3%83%BC%E3%83%AB%E3%82%B9%E3%83%9E%E3%83%B3%E3%81%AE%E3%81%9F%E3%82%81%E3%81%AB%E3%82%B8%E3%82%A7%E3%83%BC/"],
              ["Was there any evidence that Jobs was aware of any of this before he agreed to take part in it?",
               "Though Steve had a thing for Sean Connery, the feeling was not mutual. Connery was appalled by the “advert” Jobs sent across the pond and declined to participate in the misadventure on at least three separate occasions.",
               "https://scoopertino.com/exposed-the-imac-disaster-that-almost-was/"],
              ["Was there any evidence that Jobs was aware of any of this before he agreed to take part in it?",
               "Though Steve had a thing for Sean Connery, the feeling was not mutual. Connery was appalled by the “advert” Jobs sent across the pond and declined to participate in the misadventure on at least three separate occasions.",
               "https://web.archive.org/web/20201129141238/https://scoopertino.com/exposed-the-imac-disaster-that-almost-was/"],
              ["Is the prince of wales a British prince or a prince from another country?",
               "such an assembly to the duke, but James refused. In retaliation, mer-",
               "https://rockinst.org/wp-content/uploads/2017/10/New-York-State-Government-Second-Edition.pdf"],
              ["Was this a fake letter?",
               "Thousands of James Bond fans were today taken in by a spoof letter from Sean Conney to Apple boss Steve Jobs in which the film star launches a rant at the computer chief.",
               "https://www.dailymail.co.uk/news/article-2006317/Sean-Connery-letter-Steve-Jobs-rejecting-offer-appear-Apple-ad-revealed-fake.html"],
              [
                  "What was the special Christmas advertisement for Apple  Question answer:  Was the Apple ad for James bond?",
                  "Steve Jobs, a lifelong fan of James Bond (he'd originally wanted to name the revolutionary computer \"Double-O-Mac\"), instructed his agency to begin work on a special celebrity Christmas ad featuring 007 himself, Sean Connery — even though Connery had yet to be signed.",
                  "https://www.snopes.com/fact-check/false-sean-connery-letter-to-apple/"],
              ["What is this movie about?", "Starring Lorraine Bracco, Sean Connery, José Wilker",
               "https://tv.apple.com/us/movie/medicine-man/umc.cmc.4okbho8b3z0zwwsq45dju39vk"],
              ["What is the name of Apple’s cultivar of apples?",
               "The Macintosh was named after a type (or, more appropriately for a company run by Steve Jobs, a “cultivar”) of apple.",
               "https://www.techadvisor.com/article/725333/apple-a-z-everything-you-need-to-know-about-apple.html"],
              ["What is the significance of Sean connery's letter to Steve jobs?",
               "Dieser war damit nicht ganz einverstanden und das ging ihm offenbar gehörig auf den Saque. Also schrieb Sean Connery einen nicht ganz freundlichen Brief an Steve Jobs.",
               "https://www.kraftfuttermischwerk.de/blogg/james-bonds-brief-an-steve-jobs/"],
              ["Did Jobs order the James bond ad ad to start production before Sean connery was signed?",
               "Steve Jobs, a lifelong fan of James Bond (he’d originally wanted to name the revolutionary computer “Double-O-Mac”), instructed his agency to begin work on a special celebrity Christmas ad featuring 007 himself, Sean Connery — even though Connery had yet to be signed.",
               "https://scoopertino.com/exposed-the-imac-disaster-that-almost-was/"],
              ["Did Jobs order the James bond ad ad to start production before Sean connery was signed?",
               "Steve Jobs, a lifelong fan of James Bond (he’d originally wanted to name the revolutionary computer “Double-O-Mac”), instructed his agency to begin work on a special celebrity Christmas ad featuring 007 himself, Sean Connery — even though Connery had yet to be signed.",
               "https://web.archive.org/web/20201129141238/https://scoopertino.com/exposed-the-imac-disaster-that-almost-was/"],
              ["Did Jobs order the James bond ad ad to start production before Sean connery was signed?",
               "Steve Jobs, a lifelong fan of James Bond (he’d originally wanted to name the revolutionary computer “Double-O-Mac”), instructed his agency to begin work on a special celebrity Christmas ad featuring 007 himself, Sean Connery — even though Connery had yet to be signed.",
               "https://maclalala2.wordpress.com/2011/06/24/%E3%81%9F%E3%81%8B%E3%81%8C%E3%82%B3%E3%83%B3%E3%83%94%E3%83%A5%E3%83%BC%E3%82%BF%E3%82%BB%E3%83%BC%E3%83%AB%E3%82%B9%E3%83%9E%E3%83%B3%E3%81%AE%E3%81%9F%E3%82%81%E3%81%AB%E3%82%B8%E3%82%A7%E3%83%BC/"],
              [
                  "What What was was said said by Sean SeanConnery about Apple Apple  Question answer:  In the letter, which was sent to Jobs, Connernery wrote: “ItIt is a pleasure to write to you you to tell tell you that I have to to say that i i that it i am am ver?",
                  "Sean Connery has been in the news of late: First there was the “gay kiss” (more on that later), then The Donald (a.k.a. Donald Trump) announced he wanted Connery to open his Scottish golf complex, and now comes a letter – fake, but worth reading all the same – “From the Desk of Sean Connery,” telling Apple’s computer salesman Steve Jobs to get lost for good.",
                  "https://www.altfg.com/film/warren-beatty-howard-hughes/"], ["What does James bond mean to Apple?",
                                                                               "In what may be the most exciting James Bond/Apple crossover since the famous fake letter from Sean Connery to Steve Jobs, style icon James Bond cosplaying as Apple’s late CEO is perhaps the best compliment Apple can be paid as it continues to take on the fashion world.",
                                                                               "https://www.cultofmac.com/316087/what-do-steve-jobs-and-james-bond-have-in-common-turtlenecks-black-turtlenecks/"],
              ["What was the name of the actor who was knighted by Queen Elizabeth II in 2020?",
               "Though Connery became known as Sir Sean Connery, it was a bumpy road to his eventual knighting in 2000.",
               "https://www.looper.com/246825/the-untold-truth-of-sean-connery/"],
              ["How many hours did Jobs spend working on the Apple computer?",
               "According to resources including documented first-person interviews, TIME magazine and Walter Isaacson, Steve Jobs’ biographer, here is a mosaic of Steve Jobs’ sample day:",
               "https://owaves.com/day-plan/day-life-steve-jobs/"],
              ["Who was the actor who played the role of Jack Sparrow in Pirates Of The Caribbean?",
               "played by Sean Connery, Jack Nicholson, or Harrison ",
               "https://www.brooklaw.edu/-/media/Brooklaw/News-Events/Brooklyn-Law-Notes/Legacy-Issues/PDFs/LawNotesFall2012.pdf"],
              ["Who is Sean connery?", "Producer: Rhonda Tollefson, Michael Hertzberg, Sean Connery",
               "https://www.rottentomatoes.com/m/entrapment"],
              ["Who are the actors in this movie?", "Cast: Sean Connery, Ursula Andress, Joseph Wiseman",
               "https://www.afi.com/afis-100-years-100-heroes-villians/"],
              ["Who are the actors in this movie?", "Cast: Sean Connery, Ursula Andress, Joseph Wiseman",
               "https://www.afi.com/afis-100-years-100-heroes-villians/"], ["What was it called?",
                                                                            "A guy called Jony Ive, an Englishman, was hand-in-glove with Steve Jobs. The first apple instrument that was not white was the U2 black and red iPod.",
                                                                            "https://thecurrency.news/articles/27183/u2-and-me-paul-mcguinness-on-russian-oligarchs-in-the-riviera-cutting-deals-with-steve-jobs-and-taking-financial-advice-from-bono/"],
              [
                  "-------  --------- ------------  --------------------- ------------- --------------  ---------- -----------  ------  --------  -----  --  ---  ----  -  /  \\   /   \\ /    /     /      /        /       /         /          /           /            /?",
                  "In your natal chart, Sean Connery, the ten main planets are distributed as follows:",
                  "https://www.astrotheme.com/astrology/Sean_Connery"], ["Why did Jobs leave Apple in 1985?",
                                                                         "In 1977 Steve Jobs founded Apple together with Steve Wozniak, Ronald Wayne and Mike Markkula. In 1985 Jobs resigned from Apple after losing a struggle with the board of directors.",
                                                                         "https://creativecriminals.com/celebrities/apple/think-different"],
              ["Who is the longest screener in cinema history?",
               "Sean Connery, albeit his hirsute moobs have a longer screentime ",
               "https://commanderbond.net/wp-content/uploads/2013/12/The-007th-Minute-Ebook-Edition.pdf"],
              ["What are the elements, modes and house accentuations in this song?",
               "Elements, Modes and House Accentuations for Sean Connery",
               "https://www.astrotheme.com/astrology/Sean_Connery"],
              ["What is the reason why Sean connery didn't stop drinking wine?",
               "Sean Connery has said that he refused to give up his favourite wine despite doctors' calls for him to quit drinking.",
               "https://www.digitalspy.com/showbiz/a181395/connery-refused-to-quit-drinking/"], [
                  "had not received a response to his request for a meeting with the White House press secretary, Kayleigh McEnany, on the matter. Chenoault also wrote that she had asked the administration to \"clarify\" whether it had any plans to meet with him, but ha?",
                  "his complaints in a letter to the president.\" In the letter, Chennault said he ",
                  "https://history.army.mil/html/books/068/68-4/CMH_Pub_68-4.pdf"], ["What is the name of that movie?",
                                                                                     "appearance, a lot. Because Costner appeared with Sean Connery in The Untoucha-",
                                                                                     "https://monoskop.org/images/7/7b/Lovink_Geert_Rasch_Miriam_eds_Unlike_Us_Reader_Social_Media_Monopolies_and_Their_Alternatives.pdf"],
              ["What was Steve Jobs' full name?",
               "RIP Steve Jobs, thanks for everything. You have been an inspiration to my entrepreneurial career.",
               "https://news.ycombinator.com/item?id=3078128"],
              ["Was Joe's column published in 2008 about Jobs health and his criticisms of Apple and Steve Jobs?",
               "In 2008, Joe Nocera was working on a column about Steve Jobs' health, criticizing Jobs and Apple for keeping it a secret from investors.",
               "https://www.businessinsider.com/steve-jobs-jerk-2011-10"], [
                  "more than 100,000 people in person, but we can all be part of a virtual audience. Evidently, there is no such thing as a “virtual audience”, as it is not possible to have a conversation with a group of people who are not physically present in front?",
                  "or the late Apple founder Steve Jobs, few of us will ever talk to an audience of ",
                  "https://www.cag.edu.tr/uploads/site/lecturer-files/mary-guffey-essentials-of-business-communication-2016-yzss.pdf"],
              ["What does the character of Sean Connery's character mean to you?",
               "Dressed in his character's costume, Connery exudes an air of effortless charm and sophistication as he indulges in a crisp apple.",
               "https://www.mediastorehouse.com/memory-lane-prints/mirror/0000to0099-00029/sean-connery-eating-apple-set-film-highlander-21300411.html"],
              ["What does the character of Sean Connery's character mean to you?",
               "Dressed in his character's costume, Connery exudes an air of effortless charm and sophistication as he indulges in a crisp apple.",
               "https://www.mediastorehouse.co.uk/memory-lane-prints/mirror/0000to0099-00029/sean-connery-eating-apple-set-film-highlander-21300411.html"],
              ["What film was Sean connery in?", "In what film did Sean Connery sing Pretty Irish Girl",
               "https://www.scoutingpolaris.nl/downloads/spellen/10.000vragen.pdf"],
              ["Who was Robin williams and why did he refuse to play Jobs in his biopic?",
               "The next major stumbling block was in choosing an actor to read the script. Siltanen wanted Robin Williams but he refused to do any advertising, even after Jobs attempted to call him personally (his wife refused to put Jobs through).",
               "https://www.creativereview.co.uk/apple-think-different-slogan/"],
              ["What does the word “digital” mean in this context?",
               "in an open letter to the Department of Social Work, “In this technological age, when ",
               "https://wne.edu/university-archives/doc/WNE_History.pdf"],
              ["What is the name of a movie starring James Bond and Sean  Connery?",
               "rockets eaten whole, Sean Connery with a camera on his head, Fifty ",
               "https://commanderbond.net/wp-content/uploads/2013/12/The-007th-Minute-Ebook-Edition.pdf"],
              ["What was Steve Jobs' response to the question about the Apple Watch?",
               "Excerpts from an Oral History Interview with Steve Jobs",
               "https://americanhistory.si.edu/comphist/sj1.html"],
              ["What is the difference between a phone call and a text message?",
               "Steve Jobs got his start in business with another Steve, Steve Wozniak, building the blue boxes phone phreakers used to make free calls across the nation.",
               "https://www.investopedia.com/articles/fundamental-analysis/12/steve-jobs-apple-story.asp"],
              ["What is the name of this movie?",
               "Sean Connery writes me a letter and gives a phone call to Disney and says, Guys, you know what? I think I'm too old for this part.",
               "http://www.cigaraficionado.com/article/an-interview-with-arnon-milchan-6231"], ["Is this a good idea?",
                                                                                                "first glance appear to be a commodity, undifferentiated product, in an attempt to improve ",
                                                                                                "https://colbournecollege.weebly.com/uploads/2/3/7/9/23793496/essentials_of_marketing_3e.pdf"],
              ["Why did Apple change its name to Apple Inc. in 2001?",
               "In 1997, the year Steve Jobs returned as CEO, the company successfully managed to rebrand Apple as a product for independent thinkers.",
               "https://www.businessinsider.com/apple-history-through-advertising-40-years-anniversary-2016-3"],
              ["What is Steve Jobs' net worth in 2011?", "- ^ Walter Isaacson, Steve Jobs, Simon & Schuster, 2011",
               "https://en.wikipedia.org/wiki/IPod_advertising"], [
                  "What was the exact quote from Steve Jobs: \"I’ve been thinking about this for a long time. And I think it’s time for Apple to go public. It’s time to get out of the shadows and into the light. Because I don’t think Apple’s going anywhere else. Apple?",
                  "That’s precisely what we’ve done with Steve Jobs. Well, to an extent.",
                  "https://georgejziogas.medium.com/lets-stop-worshipping-steve-jobs-and-people-like-him-a99f2a7caa00"],
              ["What is the crime that the character is planning to commit?",
               "Connery. In Victorian England, a criminal plans to",
               "https://www.dominionpost.com/wp-content/uploads/paperpages/DP-2013-02-24.pdf"],
              ["What are the dominant planets, signs and houses for a person born on a given date?",
               "Dominants: Planets, Signs and Houses for Sean Connery",
               "https://www.astrotheme.com/astrology/Sean_Connery"],
              ["What is the difference between the Good Steve and the Bad Steve?",
               "When it comes to Steve Jobs, there's the \"Good Steve,\" and then, there's the \"Bad Steve,\" says biographer Walter Isaacson.",
               "https://www.businessinsider.com/steve-jobs-jerk-2011-10"], ["What did 007 say about Steve Jobs' death?",
                                                                            "One would think that the only thing 007 Sean Connery has in common with Apple co-founder Steve Jobs is a penchant for cool gadgets but this morning’s tweets proved otherwise.",
                                                                            "https://www.telegraph.co.uk/culture/film/jamesbond/8589096/Fake-Sean-Connery-letter-to-Steve-Jobs-becomes-Twitter-sensation.html"],
              ["Why did Jobs kill the clone program?",
               "In 1997 Steve Jobs returned to Apple, called the cloners “leeches” and killed the whole thing.",
               "https://www.techadvisor.com/article/725333/apple-a-z-everything-you-need-to-know-about-apple.html"],
              ["What is the name of Steve Jobs' father?",
               "With respect to Steve Jobs, his family, and the tremendous legacy he created.",
               "https://owaves.com/day-plan/day-life-steve-jobs/"], [
                  "Why is there a difference between the numbers of people who have died from COVID-19 in USA compared to other world countries?",
                  "Apparently Sean Connery is a bigger deal than Steve Jobs across the pond too. And apparently, this isn't the only difference in the lists from the sequel that is already hitting theaters around the world.",
                  "https://www.firstshowing.net/2014/check-out-steve-rogers-varying-to-do-lists-from-captain-america-2/"],
              ["What was Steve Jobs's goal?",
               "His goal, according to Walter Isaacson's biography \"Steve Jobs,\" was to build an enduring company that prioritized people.",
               "https://www.cnbc.com/2019/10/05/apple-ceo-steve-jobs-technology-is-nothing-heres-what-it-takes-to-achieve-great-success.html"],
              ["What is Bono's favorite music genre?",
               "And Bono formed a very deep friendship with Steve Jobs and had known him already before this. And Steve Jobs, a big music fan…",
               "https://thecurrency.news/articles/27183/u2-and-me-paul-mcguinness-on-russian-oligarchs-in-the-riviera-cutting-deals-with-steve-jobs-and-taking-financial-advice-from-bono/"],
              ["What is the difference between a commercial and an advertisement?",
               "forwarding to watch an appealing or novel commercial. In addition, longer commercials",
               "https://steladhima.files.wordpress.com/2014/03/consumer-behavior.pdf"],
              ["What was Steve Jobs' role in creating the Apple computer?",
               "- Steve Jobs Discovers the Macintosh Project, Mac History",
               "https://www.bahcall.com/tim-ferriss-garry-kasparov-and-the-secret-weapon-of-a-world-champion-chess-player/"],
              ["What are the symptoms of COVID-19 in animals?",
               "and entertainers such as Sammy Davis, Jr., Sean Connery, Dean Martin, ",
               "https://nibmehub.com/opac-service/pdf/read/Designing%20Clothes%20Culture%20and%20Organization%20of%20the%20Fashion%20Industry.pdf"],
              ["Why did Jobs write a letter to his iPhone customers in 2007?",
               "Jobs, S. 2007, September 6. \"Steve Jobs' Letter to iPhone Customers.\" The Wall Street Journal. ",
               "https://www.augie.edu/sites/default/files/u57/pdf/jaciel_subdocs/iPhone.pdf"],
              ["Who is Murray's character in Whiskey?",
               "Murray plays an out-of-luck American actor who goes to Japan to advertise whiskey, following in the footsteps of Mickey Rourke, Sammy Davis Junior and, again, Sean Connery.",
               "http://news.bbc.co.uk/2/hi/entertainment/3326137.stm"],
              ["What is the age of Sean connery?", "Sean Connery, Actor And The Original James Bond, Dies At 90",
               "https://www.npr.org/2020/10/31/521703453/sean-connery-actor-and-the-original-james-bond-dies-at-90"],
              ["What is the age of Sean connery?", "Sean Connery, Actor And The Original James Bond, Dies At 90",
               "https://www.npr.org/2020/10/31/521703453/sean-connery-actor-and-the-original-james-bond-dies-at-90"],
              ["Who are the people who interviewed Roger and Sean Moore and who are they?",
               "interviews with, and publicity about, Sean Connery and Roger Moore;",
               "https://files.eric.ed.gov/fulltext/ED355523.pdf"], ["When was Sean Connery's birth date?",
                                                                    "Sean married actress Diane Cilento in 1962 and they had Sean's only child, Jason Connery, born on January 11, 1963.",
                                                                    "https://www.imdb.com/name/nm0000125/"],
              ["When was Sean Connery's birth date?",
               "Sean married actress Diane Cilento in 1962 and they had Sean's only child, Jason Connery, born on January 11, 1963.",
               "https://www.imdb.com/name/nm0000125/"], ["What was this movie about?",
                                                         "Sam Neill, who appeared with Connery in The Hunt for Red October, tweeted: “Every day on set with Sean Connery was an object lesson in how to act on screen.",
                                                         "https://www.theguardian.com/film/2020/oct/31/sean-connery-james-bond-actor-dies-aged-90"],
              ["Who was the first person to die of COVID-19 in the United States?",
               "In a letter to his wife, a Confederate soldier who witnessed ",
               "https://www.wsfcs.k12.nc.us/cms/lib/NC01001395/Centricity/Domain/2407/A%20Pocket%20Style%20Manual%20-%20Diana%20Hacker.pdf"],
              ["How can I recognize a task in my apple’s apple?",
               "the teacher’s desk is an apple. The task, in other words, is to recognize an",
               "https://www.tribuneschoolchd.com/uploads/tms/files/1595167242-the-creative-mind-pdfdrive-com-.pdf"],
              ["What was the reason for this statement by Steve Jobs[2]?",
               "Steve Jobs, he stated, avoided using people in his ads because it was difficult to find an actor who appealed to everyone.[2]",
               "https://en.wikipedia.org/wiki/IPod_advertising"],
              ["What is the current status of Connery's health?", "For much more on Sean Connery, please scroll down.",
               "https://www.closerweekly.com/posts/sean-connery-movies-your-guide-to-the-actors-life-and-career/"], [
                  "How did Steve Job play hardball in iPhone  Question answer:  What does the term \"softball\" mean in relation to the iPhone?",
                  "Sharma, A., Wingfield, N., & Yuan, L. 2007, February 17. \"How Steve Jobs Played Hardball In iPhone ",
                  "https://www.augie.edu/sites/default/files/u57/pdf/jaciel_subdocs/iPhone.pdf"]]}
