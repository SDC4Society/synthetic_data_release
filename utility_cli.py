"""
Command-line interface for running utility evaluation
"""

import json
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from os import path
from numpy import mean
from numpy.random import choice, seed
import pandas as pd
from utils.utils import json_numpy_serialzer
from utils.logging import LOGGER
from sklearn.model_selection import train_test_split
from utils.parallel import create_model, create_utility_task, cleanup_resources
from utils.evaluation_framework import EvaluationEngine

def _deep_tuple(obj):
    """Recursively convert lists to tuples so nested configs are hashable."""
    if isinstance(obj, (list, tuple)):
        return tuple(_deep_tuple(x) for x in obj)
    return obj

from warnings import simplefilter
simplefilter('ignore', category=FutureWarning)
simplefilter('ignore', category=DeprecationWarning)

cwd = path.dirname(__file__)

SEED = 42


def utility_eval_gm_worker(model_config, rawTout, targets, targetIDs,
                           utility_task_configs, testRecords, testRecordIDs,
                           rawTest, metadata, runconfig):
    """Evaluate one generative model's utility across all targets.
    :return: tuple: (model_name, results_target dict, results_agg dict)
    """
    try:
        model = create_model(model_config, metadata)
        model.set_seed(SEED)
        utility_tasks = [create_utility_task(cfg, metadata) for cfg in utility_task_configs]
        for ut in utility_tasks:
            ut.set_seed(SEED)
        nSynT = runconfig['nSynT']
        sizeSynT = runconfig['sizeSynT']

        results_target = {}
        results_agg = {}

        model.fit(rawTout)
        synTwithoutTarget = [model.generate_samples(sizeSynT) for _ in range(nSynT)]

        for ut in utility_tasks:
            predErrorTargets = []
            predErrorAggr = []
            for syn in synTwithoutTarget:
                ut.train(syn)
                predErrorTargets.append(ut.evaluate(testRecords))
                predErrorAggr.append(ut.evaluate(rawTest))

            if predErrorTargets:
                results_target[(ut.__name__, 'OUT')] = {
                    'TestRecordID': testRecordIDs,
                    'Accuracy': list(mean(predErrorTargets, axis=0))
                }
                results_agg.setdefault(ut.__name__, []).append(('OUT', mean(predErrorAggr)))

        for tid in targetIDs:
            target = targets.loc[[tid]]
            rawTin = pd.concat([rawTout, target])
            model.fit(rawTin)
            synTwithTarget = [model.generate_samples(sizeSynT) for _ in range(nSynT)]

            for ut in utility_tasks:
                predErrorTargets = []
                predErrorAggr = []
                for syn in synTwithTarget:
                    ut.train(syn)
                    predErrorTargets.append(ut.evaluate(testRecords))
                    predErrorAggr.append(ut.evaluate(rawTest))

                if predErrorTargets:
                    results_target[(ut.__name__, tid)] = {
                        'TestRecordID': testRecordIDs,
                        'Accuracy': list(mean(predErrorTargets, axis=0))
                    }
                    results_agg.setdefault(ut.__name__, []).append((tid, mean(predErrorAggr)))

        return (model.__name__, results_target, results_agg)
    except Exception as e:
        LOGGER.error(f"Utility evaluation failed for model {model_config[0]}: {e}")
        return (model_config[0], {}, {})
    finally:
        if 'model' in locals():
            del model
        cleanup_resources()


def utility_eval_san_worker(model_config, rawTout, targets, targetIDs,
                            utility_task_configs, testRecords, testRecordIDs,
                            rawTest, metadata, runconfig):
    """Evaluate one sanitiser's utility across all targets.
    :return: tuple: (model_name, results_target dict, results_agg dict)
    """
    try:
        model = create_model(model_config, metadata)
        model.set_seed(SEED)
        attack_metadata = model.metadata
        utility_tasks = [create_utility_task(cfg, attack_metadata) for cfg in utility_task_configs]
        for ut in utility_tasks:
            ut.set_seed(SEED)
        nSynT = runconfig['nSynT']

        results_target = {}
        results_agg = {}

        sanOut = model.sanitise(rawTout)

        for ut in utility_tasks:
            predErrorTargets = []
            predErrorAggr = []
            for _ in range(nSynT):
                ut.train(sanOut)
                predErrorTargets.append(ut.evaluate(testRecords))
                predErrorAggr.append(ut.evaluate(rawTest))

            if predErrorTargets:
                results_target[(ut.__name__, 'OUT')] = {
                    'TestRecordID': testRecordIDs,
                    'Accuracy': list(mean(predErrorTargets, axis=0))
                }
                results_agg.setdefault(ut.__name__, []).append(('OUT', mean(predErrorAggr)))

        for tid in targetIDs:
            target = targets.loc[[tid]]
            rawTin = pd.concat([rawTout, target])
            sanIn = model.sanitise(rawTin)

            for ut in utility_tasks:
                predErrorTargets = []
                predErrorAggr = []
                for _ in range(nSynT):
                    ut.train(sanIn)
                    predErrorTargets.append(ut.evaluate(testRecords))
                    predErrorAggr.append(ut.evaluate(rawTest))

                if predErrorTargets:
                    results_target[(ut.__name__, tid)] = {
                        'TestRecordID': testRecordIDs,
                        'Accuracy': list(mean(predErrorTargets, axis=0))
                    }
                    results_agg.setdefault(ut.__name__, []).append((tid, mean(predErrorAggr)))

        return (model.__name__, results_target, results_agg)
    except Exception as e:
        LOGGER.error(f"Utility evaluation failed for sanitiser {model_config[0]}: {e}")
        return (model_config[0], {}, {})
    finally:
        if 'model' in locals():
            del model
        cleanup_resources()


def main():
    engine = EvaluationEngine(description="Command-line interface for running utility evaluation")
    engine.setup(cwd)

    runconfig = engine.runconfig
    rawPop = engine.rawPop
    metadata = engine.metadata
    dname = engine.dname
    args = engine.args

    ########################
    #### GAME INPUTS #######
    ########################
    # Train test split
    if 'dataFilter' in runconfig and runconfig['dataFilter']:
        rawTrain = rawPop.query(runconfig['dataFilter']['train'])
        rawTest = rawPop.query(runconfig['dataFilter']['test'])
    else:
        specified_targets = runconfig.get('Targets') or []
        specified_test_records = runconfig.get('TestRecords') or []
        reserved_ids = set(specified_targets) | set(specified_test_records)
        if reserved_ids:
            mask_targets = rawPop.index.isin(specified_targets)
            mask_test = rawPop.index.isin(specified_test_records)
            reserved_train = rawPop[mask_targets & ~mask_test]
            reserved_test = rawPop[mask_test & ~mask_targets]
            rest = rawPop[~mask_targets & ~mask_test]
            rest_train, rest_test = train_test_split(rest, test_size=0.5, random_state=SEED)
            rawTrain = pd.concat([reserved_train, rest_train])
            rawTest = pd.concat([reserved_test, rest_test])
        else:
            # Default to 50/50 random split if no filter is provided
            rawTrain, rawTest = train_test_split(rawPop, test_size=0.5, random_state=SEED)

    # Pick targets
    targetIDs = choice(list(rawTrain.index), size=runconfig['nTargets'], replace=False).tolist()

    # If specified: Add specific target records
    if runconfig['Targets'] is not None:
        targetIDs.extend(runconfig['Targets'])

    targets = rawTrain.loc[targetIDs, :]

    # Drop targets from population
    rawTrainWoTargets = rawTrain.drop(targetIDs)

    # Get test target records
    testRecordIDs = choice(list(rawTest.index), size=runconfig['nTargets'], replace=False).tolist()

    # If specified: Add specific target records
    if runconfig['TestRecords'] is not None:
        testRecordIDs.extend(runconfig['TestRecords'])

    testRecords = rawTest.loc[testRecordIDs, :]

    # Build serializable utility task configs
    utility_task_configs = []
    for taskName, paramsList in runconfig['utilityTasks'].items():
        for params in paramsList:
            utility_task_configs.append((taskName, *params))

    ##################################
    ######### EVALUATION #############
    ##################################
    # Build utility task names
    ut_names = []
    for cfg in utility_task_configs:
        ut = create_utility_task(cfg, metadata)
        ut_names.append(ut.__name__)

    resultsTargetUtility = {ut_name: {'Raw': {}} for ut_name in ut_names}
    resultsAggUtility = {ut_name: {'Raw': {'TargetID': [], 'Accuracy': []}} for ut_name in ut_names}

    for nr in range(runconfig['nIter']):
        rIdx = choice(list(rawTrainWoTargets.index), size=runconfig['sizeRawT'], replace=False).tolist()
        rawTout = rawTrain.loc[rIdx]

        ###############
        ## RAW EVALUATION (keep serial - small computation)
        ###############
        LOGGER.info('Start: Utility evaluation on Raw...')

        for ut_cfg in utility_task_configs:
            ut = create_utility_task(ut_cfg, metadata)
            ut.set_seed(SEED)

            resultsTargetUtility[ut.__name__]['Raw'][nr] = {}

            predErrorTargets = []
            predErrorAggr = []
            for _ in range(runconfig['nSynT']):
                ut.train(rawTout)
                predErrorTargets.append(ut.evaluate(testRecords))
                predErrorAggr.append(ut.evaluate(rawTest))

            resultsTargetUtility[ut.__name__]['Raw'][nr]['OUT'] = {
                'TestRecordID': testRecordIDs,
                'Accuracy': list(mean(predErrorTargets, axis=0))
            }
            resultsAggUtility[ut.__name__]['Raw']['TargetID'].append('OUT')
            resultsAggUtility[ut.__name__]['Raw']['Accuracy'].append(mean(predErrorAggr))

        for tid in targetIDs:
            target = targets.loc[[tid]]
            rawIn = pd.concat([rawTout, target])

            for ut_cfg in utility_task_configs:
                ut = create_utility_task(ut_cfg, metadata)
                ut.set_seed(SEED)

                predErrorTargets = []
                predErrorAggr = []
                for _ in range(runconfig['nSynT']):
                    ut.train(rawIn)
                    predErrorTargets.append(ut.evaluate(testRecords))
                    predErrorAggr.append(ut.evaluate(rawTest))

                resultsTargetUtility[ut.__name__]['Raw'][nr][tid] = {
                    'TestRecordID': testRecordIDs,
                    'Accuracy': list(mean(predErrorTargets, axis=0))
                }
                resultsAggUtility[ut.__name__]['Raw']['TargetID'].append(tid)
                resultsAggUtility[ut.__name__]['Raw']['Accuracy'].append(mean(predErrorAggr))

        LOGGER.info('Finished: Utility evaluation on Raw.')

        ###############
        ## PARALLEL MODEL EVALUATION
        ###############
        gm_tasks = [
            (cfg, rawTout, targets, targetIDs,
             utility_task_configs, testRecords, testRecordIDs, rawTest, metadata, runconfig)
            for cfg in engine.gm_configs
        ]
        san_tasks = [
            (cfg, rawTout, targets, targetIDs,
             utility_task_configs, testRecords, testRecordIDs, rawTest, metadata, runconfig)
            for cfg in engine.san_configs
        ]

        all_results = engine.run_parallel_evaluation(
            eval_gm_worker=utility_eval_gm_worker,
            eval_san_worker=utility_eval_san_worker,
            san_tasks=san_tasks,
            gm_tasks=gm_tasks,
            iter_idx=nr,
            desc_prefix="eval"
        )

        for model_name, results_target, results_agg in all_results:
            for (ut_name, tid_or_out), result_dict in results_target.items():
                if ut_name not in resultsTargetUtility:
                    resultsTargetUtility[ut_name] = {}
                if model_name not in resultsTargetUtility[ut_name]:
                    resultsTargetUtility[ut_name][model_name] = {}
                if nr not in resultsTargetUtility[ut_name][model_name]:
                    resultsTargetUtility[ut_name][model_name][nr] = {}
                resultsTargetUtility[ut_name][model_name][nr][tid_or_out] = result_dict

            for ut_name, entries in results_agg.items():
                if ut_name not in resultsAggUtility:
                    resultsAggUtility[ut_name] = {}
                if model_name not in resultsAggUtility[ut_name]:
                    resultsAggUtility[ut_name][model_name] = {'TargetID': [], 'Accuracy': []}
                for tid_or_out, accuracy in entries:
                    resultsAggUtility[ut_name][model_name]['TargetID'].append(tid_or_out)
                    resultsAggUtility[ut_name][model_name]['Accuracy'].append(accuracy)

    engine.dump_results(resultsTargetUtility, prefix="ResultsUtilTargets")
    engine.dump_results(resultsAggUtility, prefix="ResultsUtilAgg")


if __name__ == "__main__":
    main()
