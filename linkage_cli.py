"""
Command-line interface for running privacy evaluation with respect to the risk of linkability
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

import json

from os import mkdir, path
from numpy.random import choice, seed
from argparse import ArgumentParser
from pandas import DataFrame
import pandas as pd
from utils.datagen import load_s3_data_as_df, load_local_data_as_df
from utils.utils import json_numpy_serialzer
from utils.logging import LOGGER
from utils.constants import *
from utils.parallel import create_model, is_generative_model, run_parallel_models

from feature_sets.independent_histograms import HistogramFeatureSet
from feature_sets.model_agnostic import NaiveFeatureSet, EnsembleFeatureSet
from feature_sets.bayes import CorrelationsFeatureSet

from sanitisation_techniques.sanitiser import SanitiserNHS

from generative_models.ctgan import CTGAN
from generative_models.pate_gan import PATEGAN
from generative_models.data_synthesiser import (IndependentHistogram,
                                                BayesianNet,
                                                PrivBayes)

from attack_models.mia_classifier import (MIAttackClassifierRandomForest,
                                          generate_mia_shadow_data,
                                          generate_mia_anon_data)

from warnings import simplefilter
simplefilter('ignore', category=FutureWarning)
simplefilter('ignore', category=DeprecationWarning)

cwd = path.dirname(__file__)


SEED = 42


def linkage_attack_worker(model_config, tid, target, rawA, metadata, runconfig):
    """Train MIA attacks for one (target, model) pair."""
    model = create_model(model_config, metadata)
    trained_attacks = {}

    if is_generative_model(model):
        synA, labelsA = generate_mia_shadow_data(
            model, target, rawA,
            runconfig['sizeRawT'], runconfig['sizeSynT'],
            runconfig['nShadows'], runconfig['nSynA'])

        for Feature in [NaiveFeatureSet(model.datatype),
                        HistogramFeatureSet(model.datatype, metadata),
                        CorrelationsFeatureSet(model.datatype, metadata)]:
            Attack = MIAttackClassifierRandomForest(metadata, Feature)
            Attack.train(synA, labelsA)
            trained_attacks[Feature.__name__] = Attack
    else:
        sanA, labelsA = generate_mia_anon_data(
            model, target, rawA,
            runconfig['sizeRawT'],
            runconfig['nShadows'] * runconfig['nSynA'])

        for Feature in [NaiveFeatureSet(DataFrame),
                        HistogramFeatureSet(DataFrame, metadata,
                                           nbins=model.histogram_size, quids=model.quids),
                        CorrelationsFeatureSet(DataFrame, metadata, quids=model.quids),
                        EnsembleFeatureSet(DataFrame, metadata,
                                          nbins=model.histogram_size,
                                          quasi_id_cols=model.quids)]:
            Attack = MIAttackClassifierRandomForest(metadata=metadata, FeatureSet=Feature, quids=model.quids)
            Attack.train(sanA, labelsA)
            trained_attacks[Feature.__name__] = Attack

    return (tid, model.__name__, trained_attacks)


def linkage_eval_worker(model_config, rawTout, targets, targetIDs,
                        attacks_for_model, metadata, runconfig):
    """Evaluate one model across all targets for one game iteration."""
    model = create_model(model_config, metadata)
    nSynT = runconfig['nSynT']
    sizeSynT = runconfig['sizeSynT']
    per_target_results = {}

    if is_generative_model(model):
        model.fit(rawTout)
        synTwithoutTarget = [model.generate_samples(sizeSynT) for _ in range(nSynT)]
        synLabelsOut = [LABEL_OUT for _ in range(nSynT)]

        for tid in targetIDs:
            target = targets.loc[[tid]]
            rawTin = pd.concat([rawTout, target])
            model.fit(rawTin)
            synTwithTarget = [model.generate_samples(sizeSynT) for _ in range(nSynT)]
            synLabelsIn = [LABEL_IN for _ in range(nSynT)]

            synT = synTwithoutTarget + synTwithTarget
            synTlabels = synLabelsOut + synLabelsIn

            per_target_results[tid] = {}
            for feature, Attack in attacks_for_model[tid].items():
                attackerGuesses = Attack.attack(synT)
                per_target_results[tid][feature] = {
                    'Secret': synTlabels,
                    'AttackerGuess': attackerGuesses
                }
    else:
        sanOut = model.sanitise(rawTout)
        for tid in targetIDs:
            target = targets.loc[[tid]]
            rawTin = pd.concat([rawTout, target])
            sanIn = model.sanitise(rawTin)

            sanT = [sanOut, sanIn]
            sanTLabels = [LABEL_OUT, LABEL_IN]

            per_target_results[tid] = {}
            for feature, Attack in attacks_for_model[tid].items():
                attackerGuesses = Attack.attack(sanT, attemptLinkage=True, target=target)
                per_target_results[tid][feature] = {
                    'Secret': sanTLabels,
                    'AttackerGuess': attackerGuesses
                }

    return (model.__name__, per_target_results)


def main():
    argparser = ArgumentParser()
    datasource = argparser.add_mutually_exclusive_group()
    datasource.add_argument('--s3name', '-S3', type=str, choices=['adult', 'census', 'credit', 'alarm', 'insurance'], help='Name of the dataset to run on')
    datasource.add_argument('--datapath', '-D', type=str, help='Relative path to cwd of a local data file')
    argparser.add_argument('--runconfig', '-RC', default='runconfig_mia.json', type=str, help='Path relative to cwd of runconfig file')
    argparser.add_argument('--outdir', '-O', default='tests', type=str, help='Path relative to cwd for storing output files')
    argparser.add_argument('--workers', '-W', type=int, default=None,
                           help='Number of parallel workers (default: CPU count)')
    args = argparser.parse_args()

    # Load runconfig
    with open(path.join(cwd, args.runconfig)) as f:
        runconfig = json.load(f)
    print('Runconfig:')
    print(runconfig)

    # Load data
    if args.s3name is not None:
        rawPop, metadata = load_s3_data_as_df(args.s3name)
        dname = args.s3name
    else:
        rawPop, metadata = load_local_data_as_df(path.join(cwd, args.datapath))
        dname = args.datapath.split('/')[-1]

    print(f'Loaded data {dname}:')
    print(rawPop.info())

    # Make sure outdir exists
    if not path.isdir(args.outdir):
        mkdir(args.outdir)

    seed(SEED)

    ########################
    #### GAME INPUTS #######
    ########################
    # Pick targets
    targetIDs = choice(list(rawPop.index), size=runconfig['nTargets'], replace=False).tolist()

    # If specified: Add specific target records
    if runconfig['Targets'] is not None:
        targetIDs.extend(runconfig['Targets'])

    targets = rawPop.loc[targetIDs, :]

    # Drop targets from population
    rawPopDropTargets = rawPop.drop(targetIDs)

    # Init adversary's prior knowledge
    rawAidx = choice(list(rawPopDropTargets.index), size=runconfig['sizeRawA'], replace=False).tolist()
    rawA = rawPop.loc[rawAidx, :]

    # Build serializable model configs for parallel execution
    all_model_configs = []
    if 'generativeModels' in runconfig.keys():
        for gm, paramsList in runconfig['generativeModels'].items():
            for params in paramsList:
                all_model_configs.append((gm, *params))

    if 'sanitisationTechniques' in runconfig.keys():
        for name, paramsList in runconfig['sanitisationTechniques'].items():
            for params in paramsList:
                all_model_configs.append((name, *params))

    ###################################
    #### ATTACK TRAINING #############
    ##################################
    print('\n---- Attack training ----')

    attack_tasks = [
        (cfg, tid, targets.loc[[tid]], rawA, metadata, runconfig)
        for tid in targetIDs
        for cfg in all_model_configs
    ]
    attack_results = run_parallel_models(
        linkage_attack_worker, attack_tasks, max_workers=args.workers)

    attacks = {}
    for tid, model_name, trained in attack_results:
        attacks.setdefault(tid, {})[model_name] = trained

    ##################################
    ######### EVALUATION #############
    ##################################
    # Build model_name lookup from configs
    _model_names = {}
    for cfg in all_model_configs:
        m = create_model(cfg, metadata)
        _model_names[cfg] = m.__name__

    resultsTargetPrivacy = {tid: {name: {} for name in _model_names.values()} for tid in targetIDs}

    print('\n---- Start the game ----')
    for nr in range(runconfig['nIter']):
        print(f'\n--- Game iteration {nr + 1} ---')
        rIdx = choice(list(rawPopDropTargets.index), size=runconfig['sizeRawT'], replace=False).tolist()
        rawTout = rawPopDropTargets.loc[rIdx]

        eval_tasks = [
            (cfg, rawTout, targets, targetIDs,
             {tid: attacks[tid][_model_names[cfg]] for tid in targetIDs},
             metadata, runconfig)
            for cfg in all_model_configs
        ]
        eval_results = run_parallel_models(
            linkage_eval_worker, eval_tasks, max_workers=args.workers)

        for model_name, per_target in eval_results:
            for tid, feature_results in per_target.items():
                resultsTargetPrivacy[tid][model_name][nr] = feature_results

    outfile = f"ResultsMIA_{dname}"
    LOGGER.info(f"Write results to {path.join(f'{args.outdir}', f'{outfile}')}")

    with open(path.join(f'{args.outdir}', f'{outfile}.json'), 'w') as f:
        json.dump(resultsTargetPrivacy, f, indent=2, default=json_numpy_serialzer)


if __name__ == "__main__":
    main()