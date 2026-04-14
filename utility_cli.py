"""
Command-line interface for running utility evaluation
"""

import json
import os

from os import mkdir, path
from numpy import mean
from numpy.random import choice, seed
import pandas as pd
from utils.utils import json_numpy_serialzer
from utils.logging import LOGGER
from utils.parallel import create_model, create_utility_task
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
    model = create_model(model_config, metadata)
    utility_tasks = [create_utility_task(cfg, metadata) for cfg in utility_task_configs]
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

            results_target[(ut.__name__, tid)] = {
                'TestRecordID': testRecordIDs,
                'Accuracy': list(mean(predErrorTargets, axis=0))
            }
            results_agg.setdefault(ut.__name__, []).append((tid, mean(predErrorAggr)))

    return (model.__name__, results_target, results_agg)


def utility_eval_san_worker(model_config, rawTout, targets, targetIDs,
                            utility_task_configs, testRecords, testRecordIDs,
                            rawTest, metadata, runconfig):
    """Evaluate one sanitiser's utility across all targets.
    :return: tuple: (model_name, results_target dict, results_agg dict)
    """
    model = create_model(model_config, metadata)
    attack_metadata = model.get_output_metadata(metadata)
    utility_tasks = [create_utility_task(cfg, attack_metadata) for cfg in utility_task_configs]
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

            results_target[(ut.__name__, tid)] = {
                'TestRecordID': testRecordIDs,
                'Accuracy': list(mean(predErrorTargets, axis=0))
            }
            results_agg.setdefault(ut.__name__, []).append((tid, mean(predErrorAggr)))

    return (model.__name__, results_target, results_agg)


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
    rawTrain = rawPop.query(runconfig['dataFilter']['train'])
    rawTest = rawPop.query(runconfig['dataFilter']['test'])

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
