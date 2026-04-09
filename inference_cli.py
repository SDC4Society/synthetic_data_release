"""
Command-line interface for running privacy evaluation under an attribute inference adversary
"""

import json

from os import mkdir, path
from numpy.random import choice, seed
from argparse import ArgumentParser
import pandas as pd
from utils.datagen import load_s3_data_as_df, load_local_data_as_df
from utils.utils import json_numpy_serialzer
from utils.logging import LOGGER
from utils.constants import *
from utils.parallel import create_model, is_generative_model, run_parallel_models

def _deep_tuple(obj):
    """Recursively convert lists to tuples so nested configs are hashable."""
    if isinstance(obj, (list, tuple)):
        return tuple(_deep_tuple(x) for x in obj)
    return obj

from generative_models.ctgan import CTGAN
from generative_models.data_synthesiser import IndependentHistogram, BayesianNet, PrivBayes
from generative_models.pate_gan import PATEGAN
from sanitisation_techniques.sanitiser_nhs import SanitiserNHS
from attack_models.reconstruction import LinRegAttack, RandForestAttack

from warnings import simplefilter
simplefilter('ignore', category=FutureWarning)
simplefilter('ignore', category=DeprecationWarning)

cwd = path.dirname(__file__)

SEED = 42


def inference_eval_gm_worker(model_config, rawTout, targets, targetIDs,
                             sensitive_attrs, metadata, runconfig):
    """Evaluate one generative model for inference attack across all targets.
    :return: tuple: (model_name, {(tid, sa): result_dict})
    """
    model = create_model(model_config, metadata)
    nSynT = runconfig['nSynT']
    sizeSynT = runconfig['sizeSynT']

    attacks = {}
    for sa, atype in sensitive_attrs.items():
        if atype == 'LinReg':
            attacks[sa] = LinRegAttack(sensitiveAttribute=sa, metadata=metadata)
        elif atype == 'Classification':
            attacks[sa] = RandForestAttack(sensitiveAttribute=sa, metadata=metadata)

    results = {}

    model.fit(rawTout)
    synTwithoutTarget = [model.generate_samples(sizeSynT) for _ in range(nSynT)]

    for sa, Attack in attacks.items():
        for tid in targetIDs:
            results[(tid, sa)] = {
                'AttackerGuess': [], 'ProbCorrect': [],
                'TargetPresence': [LABEL_OUT for _ in range(nSynT)]
            }

        for syn in synTwithoutTarget:
            Attack.train(syn)
            for tid in targetIDs:
                target = targets.loc[[tid]]
                targetAux = target.loc[[tid], Attack.knownAttributes]
                targetSecret = target.loc[tid, Attack.sensitiveAttribute]

                guess = Attack.attack(targetAux)
                pCorrect = Attack.get_likelihood(targetAux, targetSecret)

                results[(tid, sa)]['AttackerGuess'].append(guess)
                results[(tid, sa)]['ProbCorrect'].append(pCorrect)

    for tid in targetIDs:
        target = targets.loc[[tid]]
        rawTin = pd.concat([rawTout, target])

        model.fit(rawTin)
        synTwithTarget = [model.generate_samples(sizeSynT) for _ in range(nSynT)]

        for sa, Attack in attacks.items():
            targetAux = target.loc[[tid], Attack.knownAttributes]
            targetSecret = target.loc[tid, Attack.sensitiveAttribute]

            for syn in synTwithTarget:
                Attack.train(syn)

                guess = Attack.attack(targetAux)
                pCorrect = Attack.get_likelihood(targetAux, targetSecret)

                results[(tid, sa)]['AttackerGuess'].append(guess)
                results[(tid, sa)]['ProbCorrect'].append(pCorrect)
                results[(tid, sa)]['TargetPresence'].append(LABEL_IN)

    return (model.__name__, results)


def inference_eval_san_worker(model_config, rawTout, targets, targetIDs,
                              sensitive_attrs, metadata, runconfig):
    """Evaluate one sanitiser for inference attack across all targets.
    :return: tuple: (model_name, {(tid, sa): result_dict})
    """
    model = create_model(model_config, metadata)
    attack_metadata = model.get_output_metadata(metadata)

    attacks = {}
    for sa, atype in sensitive_attrs.items():
        if atype == 'LinReg':
            attacks[sa] = LinRegAttack(sensitiveAttribute=sa, metadata=attack_metadata, quids=model.quids)
        elif atype == 'Classification':
            attacks[sa] = RandForestAttack(sensitiveAttribute=sa, metadata=attack_metadata, quids=model.quids)

    results = {}

    sanOut = model.sanitise(rawTout)

    for sa, Attack in attacks.items():
        Attack.train(sanOut)
        for tid in targetIDs:
            target = targets.loc[[tid]]
            targetAux = target.loc[[tid], Attack.knownAttributes]
            targetSecret = target.loc[tid, Attack.sensitiveAttribute]

            guess = Attack.attack(targetAux, attemptLinkage=True, data=sanOut)
            pCorrect = Attack.get_likelihood(targetAux, targetSecret, attemptLinkage=True, data=sanOut)

            results[(tid, sa)] = {
                'AttackerGuess': [guess],
                'ProbCorrect': [pCorrect],
                'TargetPresence': [LABEL_OUT]
            }

    for tid in targetIDs:
        target = targets.loc[[tid]]
        rawTin = pd.concat([rawTout, target])
        sanIn = model.sanitise(rawTin)

        for sa, Attack in attacks.items():
            targetAux = target.loc[[tid], Attack.knownAttributes]
            targetSecret = target.loc[tid, Attack.sensitiveAttribute]

            Attack.train(sanIn)

            guess = Attack.attack(targetAux, attemptLinkage=True, data=sanIn)
            pCorrect = Attack.get_likelihood(targetAux, targetSecret, attemptLinkage=True, data=sanIn)

            results[(tid, sa)]['AttackerGuess'].append(guess)
            results[(tid, sa)]['ProbCorrect'].append(pCorrect)
            results[(tid, sa)]['TargetPresence'].append(LABEL_IN)

    return (model.__name__, results)


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

    ##################################
    ######### EVALUATION #############
    ##################################
    # Build model_name lookup from configs
    _model_names = {}
    for cfg in all_model_configs:
        m = create_model(cfg, metadata)
        _model_names[_deep_tuple(cfg)] = m.__name__

    # Separate gm and san configs
    gm_configs = [(cfg, _model_names[_deep_tuple(cfg)]) for cfg in all_model_configs
                  if is_generative_model(create_model(cfg, metadata))]
    san_configs = [(cfg, _model_names[_deep_tuple(cfg)]) for cfg in all_model_configs
                   if not is_generative_model(create_model(cfg, metadata))]

    resultsTargetPrivacy = {
        tid: {sa: {name: {} for name in list(_model_names.values()) + ['Raw']}
              for sa in runconfig['sensitiveAttributes']}
        for tid in targetIDs
    }

    for nr in range(runconfig['nIter']):
        rIdx = choice(list(rawPopDropTargets.index), size=runconfig['sizeRawT'], replace=False).tolist()
        rawTout = rawPopDropTargets.loc[rIdx]

        ###############
        ## RAW ATTACKS (keep serial - small computation)
        ###############
        attacks = {}
        for sa, atype in runconfig['sensitiveAttributes'].items():
            if atype == 'LinReg':
                attacks[sa] = LinRegAttack(sensitiveAttribute=sa, metadata=metadata)
            elif atype == 'Classification':
                attacks[sa] = RandForestAttack(sensitiveAttribute=sa, metadata=metadata)

        for sa, Attack in attacks.items():
            Attack.train(rawTout)
            for tid in targetIDs:
                target = targets.loc[[tid]]
                targetAux = target.loc[[tid], Attack.knownAttributes]
                targetSecret = target.loc[tid, Attack.sensitiveAttribute]

                guess = Attack.attack(targetAux, attemptLinkage=True, data=rawTout)
                pCorrect = Attack.get_likelihood(targetAux, targetSecret, attemptLinkage=True, data=rawTout)

                resultsTargetPrivacy[tid][sa]['Raw'][nr] = {
                    'AttackerGuess': [guess],
                    'ProbCorrect': [pCorrect],
                    'TargetPresence': [LABEL_OUT]
                }

        for tid in targetIDs:
            target = targets.loc[[tid]]
            rawTin = pd.concat([rawTout, target])

            for sa, Attack in attacks.items():
                targetAux = target.loc[[tid], Attack.knownAttributes]
                targetSecret = target.loc[tid, Attack.sensitiveAttribute]

                guess = Attack.attack(targetAux, attemptLinkage=True, data=rawTin)
                pCorrect = Attack.get_likelihood(targetAux, targetSecret, attemptLinkage=True, data=rawTin)

                resultsTargetPrivacy[tid][sa]['Raw'][nr]['AttackerGuess'].append(guess)
                resultsTargetPrivacy[tid][sa]['Raw'][nr]['ProbCorrect'].append(pCorrect)
                resultsTargetPrivacy[tid][sa]['Raw'][nr]['TargetPresence'].append(LABEL_IN)

        ###############
        ## PARALLEL MODEL EVALUATION
        ###############
        gm_tasks = [
            (cfg, rawTout, targets, targetIDs,
             runconfig['sensitiveAttributes'], metadata, runconfig)
            for cfg, _ in gm_configs
        ]
        san_tasks = [
            (cfg, rawTout, targets, targetIDs,
             runconfig['sensitiveAttributes'], metadata, runconfig)
            for cfg, _ in san_configs
        ]

        all_results = []
        if gm_tasks:
            all_results.extend(run_parallel_models(
                inference_eval_gm_worker, gm_tasks, max_workers=args.workers,
                desc=f"GM eval {nr+1}/{runconfig['nIter']}"))
        if san_tasks:
            all_results.extend(run_parallel_models(
                inference_eval_san_worker, san_tasks, max_workers=args.workers,
                desc=f"San eval {nr+1}/{runconfig['nIter']}"))

        for model_name, results in all_results:
            for (tid, sa), result_dict in results.items():
                resultsTargetPrivacy[tid][sa][model_name][nr] = result_dict

    outfile = f"ResultsMLEAI_{dname}"
    LOGGER.info(f"Write results to {path.join(f'{args.outdir}', f'{outfile}')}")

    with open(path.join(f'{args.outdir}', f'{outfile}.json'), 'w') as f:
        json.dump(resultsTargetPrivacy, f, indent=2, default=json_numpy_serialzer)

if __name__ == "__main__":
    main()